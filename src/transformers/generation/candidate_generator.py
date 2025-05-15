# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
import torch
from .logger import print_tensor

from ..utils import is_sklearn_available


if is_sklearn_available():
    from sklearn.metrics import roc_curve

from ..cache_utils import DynamicCache
from ..pytorch_utils import isin_mps_friendly
from .logits_process import LogitsProcessorList, MinLengthLogitsProcessor


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from ..tokenization_utils_base import PreTrainedTokenizerBase
    from .configuration_utils import GenerationConfig


class CandidateGenerator:
    """Abstract base class for all candidate generators that can be applied during assisted generation."""

    def get_candidates(self, input_ids: torch.LongTensor) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
        """
        Fetches the candidates to be tried for the current input.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)

        Return:
            `torch.LongTensor` of shape `(batch_size, candidate_length)` containing the candidate sequences to be
            assessed by the model and, optionally, a `torch.FloatTensor` of shape `(batch_size, candidate_length,
            vocabulary_size)` containing the logits associated to each candidate.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can call `get_candidates`."
        )

    def update_candidate_strategy(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int):
        """
        Updates the candidate generation strategy based on the outcomes.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, candidate_length, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            num_matches (`int`):
                The number of matches between the candidate sequences and the model predictions.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can call "
            "`update_candidate_strategy`."
        )


class AssistedCandidateGenerator(CandidateGenerator):
    """
    `CandidateGenerator` class to be used for assisted generation and speculative decoding. This class generates
    candidates through the use of a smaller model. Read the following blog post for more information:
    https://huggingface.co/blog/assisted-generation

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        assistant_model (`PreTrainedModel`):
            The model to be used for generating candidates. This model should be smaller than the main model.
        generation_config (`~generation.GenerationConfig`, *optional*):
            The generation configuration to be used as base parametrization for the generation call.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        model_kwargs (`Dict`):
            The keyword arguments that will be passed to the main model, and are used as base inputs for the assistant
            model as well.
        inputs_tensor (`torch.Tensor`, *optional*):
            The model input tensor. In encoder-decoder models, this is the encoder input.
    """

    def __init__(
        self,
        input_ids: torch.LongTensor,
        assistant_model: "PreTrainedModel",
        generation_config: "GenerationConfig",
        model_kwargs: Dict,
        inputs_tensor: Optional[torch.Tensor] = None,
        logits_processor: "LogitsProcessorList" = None,
    ):
        # Make sure all data at the same device as assistant model
        device = assistant_model.device
        input_ids = input_ids.to(device)
        if inputs_tensor is not None:
            inputs_tensor = inputs_tensor.to(device)

        # Prepare the assistant and the starting number of candidate tokens
        self.assistant_model = assistant_model
        self.num_assistant_tokens = assistant_model.generation_config.num_assistant_tokens
        self.assistant_confidence_threshold = assistant_model.generation_config.assistant_confidence_threshold

        # Set eos in assistant same as in target model
        self.assistant_model.generation_config.eos_token_id = generation_config.eos_token_id

        # Prepare the kwargs for the assistant model
        assistant_kwargs = {}
        for key, value in model_kwargs.items():  # deepcopy crashes if we attempt to copy encoder outputs with grads
            if key not in ("encoder_outputs", "assistant_encoder_outputs", "past_key_values"):
                assistant_kwargs[key] = (
                    value.detach().to(device) if isinstance(value, torch.Tensor) else copy.deepcopy(value)
                )

        # Remove potential default "num_logits_to_keep" key
        if "num_logits_to_keep" in assistant_kwargs.keys() and not assistant_model._supports_num_logits_to_keep():
            del assistant_kwargs["num_logits_to_keep"]

        if "assistant_encoder_outputs" in model_kwargs:
            assistant_kwargs["encoder_outputs"] = model_kwargs["assistant_encoder_outputs"]
        elif assistant_model.config.is_encoder_decoder:
            inputs_tensor, model_input_name, assistant_kwargs = assistant_model._prepare_model_inputs(
                inputs_tensor, assistant_model.generation_config.bos_token_id, assistant_kwargs
            )
            assistant_kwargs = assistant_model._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, assistant_kwargs, model_input_name, assistant_model.generation_config
            )
        elif "encoder_outputs" in model_kwargs:
            assistant_kwargs["encoder_outputs"] = model_kwargs["encoder_outputs"]
        self.assistant_kwargs = assistant_kwargs

        # Prepare assistant model's keys of inputs
        if assistant_model.config.is_encoder_decoder:
            # both are encoder-decoder
            self.input_ids_key = "decoder_input_ids"
        elif "encoder_outputs" in assistant_kwargs:
            # special case for encoder-decoder with decoder-only assistant (like DistilWhisper)
            self.input_ids_key = "input_ids"
            self.assistant_kwargs["attention_mask"] = self.assistant_kwargs.get(
                "decoder_attention_mask",
                torch.ones((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.long),
            )
        else:
            # both are decoder-only
            self.input_ids_key = "input_ids"

        # Prepare generation-related options.
        self.logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        self.generation_config = copy.deepcopy(generation_config)

        self.generation_config.return_dict_in_generate = True
        self.generation_config.output_scores = True
        self.generation_config.assistant_confidence_threshold = self.assistant_confidence_threshold
        # this flag allow us set the confidence stopping criteria for assistant model generation.
        self.generation_config.is_assistant = True

        # avoid unnecessary warnings that min_length is larger than max_new_tokens
        # remove the `MinLengthLogitsProcessor` if exists (NOTE: no need to check for `MinNewTokensLogitsProcessor`)
        self.main_model_min_length = self.generation_config.min_length
        self.generation_config.min_length = 0
        self.generation_config.min_new_tokens = None
        for processor in self.logits_processor:
            if isinstance(processor, MinLengthLogitsProcessor):
                raise ValueError(
                    "Passing `MinLengthLogitsProcessor` when using `assisted_generation is disabled. "
                    "Please pass in `min_length` into `.generate()` instead"
                )

        # We need to roll back the cache in assisted generation, only DynamicCache is supported
        self.generation_config.cache_implementation = None

        if (
            is_sklearn_available()
            and self.assistant_model.generation_config.assistant_confidence_threshold
            and type(self) is AssistedCandidateGenerator
        ):
            self.probs = []
            self.matches = []

    def get_candidates(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.LongTensor]) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
        """
        Fetches the candidates to be tried for the current input.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)

        Return:
            `torch.LongTensor` of shape `(batch_size, candidate_length)` containing the candidate sequences to be
            assessed by the model and a `torch.FloatTensor` of shape `(batch_size, candidate_length,
            vocabulary_size)` containing the logits associated to each candidate.
        """
        input_ids = input_ids.to(self.assistant_model.device)

        # Don't generate more than `max_length - 1` candidates since the target model generates one extra token.
        new_cur_len = input_ids.shape[-1]
        max_new_tokens = min(int(self.num_assistant_tokens), self.generation_config.max_length - new_cur_len - 1)
        min_new_tokens = max(min(max_new_tokens, self.main_model_min_length - new_cur_len), 0)
        if max_new_tokens == 0:
            return input_ids, None

        # 1. If it is not the first round of candidate generation, prepare the inputs based on the input_ids length
        # (which implicitly contains the number of accepted candidates from the previous round)
        has_past_key_values = self.assistant_kwargs.get("past_key_values", None) is not None
        if has_past_key_values:
            new_cache_size = new_cur_len - 1
            self.assistant_kwargs["past_key_values"] = _crop_past_key_values(
                self.assistant_model, self.assistant_kwargs["past_key_values"], new_cache_size - 1
            )  # the assistant does not have the token after the last match, hence the -1

        if attention_mask is None:
            self.assistant_kwargs = _prepare_attention_mask(
                self.assistant_kwargs, new_cur_len, self.assistant_model.config.is_encoder_decoder
            )
        else:
            self.assistant_kwargs["attention_mask"] = attention_mask
        self.assistant_kwargs = _prepare_token_type_ids(self.assistant_kwargs, new_cur_len)

        # 2. Forecast next N tokens using the assistant model.
        assistant_generation_kwargs = {
            self.input_ids_key: input_ids,
            "min_new_tokens": min_new_tokens,
            "max_new_tokens": max_new_tokens,
            "generation_config": self.generation_config,
            "logits_processor": self.logits_processor,
        }
        
        # print(f"assistant_generation_kwargs:")
        # for key, value in assistant_generation_kwargs.items():
        #     print(f"    {key}:", end=" ")
        #     if key == "generation_config":
        #         print("")
        #         print(f"        use_cache: {value.use_cache}")
        #         print(f"        cache_implementation: {value.cache_implementation}")
        #         print(f"        cache_config: {value.cache_config}")
        #         print(f"        return_legacy_cache: {value.return_legacy_cache}")
        #     else:
        #         if value.__class__.__name__ == 'int':
        #             print(value)
        #         elif value.__class__.__name__ == 'Tensor':
        #             print(value.shape)
        #             print_tensor(value, n_padding=6)
        #         else:
        #             print(value.__class__.__name__)
        
        # print(f"self.assistant_kwargs:")
        # for key, value in self.assistant_kwargs.items():
        #     print(f"    {key}:", end=" ")
        #     if value.__class__.__name__ == 'int':
        #         print(value)
        #     elif value.__class__.__name__ == 'Tensor':
        #         print(value.shape)
        #         print_tensor(value, n_padding=6)
        #     else:
        #         print(value.__class__.__name__)
        
        assistant_output = self.assistant_model.generate(**assistant_generation_kwargs, **self.assistant_kwargs)

        # print(f"self.assistant_output:")
        # for key, value in assistant_output.items():
        #     print(f"    {key}:", end=" ")
        #     if value.__class__.__name__ == 'int':
        #         print(value)
        #     elif value.__class__.__name__ == 'Tensor':
        #         print(value.shape)
        #     else:
        #         print(value.__class__.__name__)

        # 3. Update variables for the next round of candidate generation
        # self.assistant_kwargs["past_key_values"] = assistant_output.past_key_values

        if (
            is_sklearn_available()
            and self.assistant_model.generation_config.assistant_confidence_threshold
            and type(self) is AssistedCandidateGenerator
        ):
            scores_tensor = torch.cat(assistant_output.scores, dim=0)
            scores_softmax = torch.softmax(scores_tensor, dim=-1)
            ids = assistant_output.sequences[-1, -len(assistant_output.scores) :]
            p = scores_softmax[range(len(ids)), ids]
            self.probs.extend(p.tolist())

        # 4. Prepare variables for output
        candidate_logits = torch.stack(assistant_output.scores, dim=1)
        candidate_ids = assistant_output.sequences
        return candidate_ids, candidate_logits

    def update_candidate_strategy(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int):
        """
        Updates the candidate generation strategy based on the outcomes.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, candidate_length, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            num_matches (`int`):
                The number of matches between the candidate sequences and the model predictions.
        """
        # Adjust the max number of assistant tokens to use in the next iteration. This is a simple heuristic,
        # probably can be improved -- we want to balance the benefits of getting assistant tokens correct with the
        # cost of forecasting incorrect assistant tokens.
        if self.assistant_model.generation_config.num_assistant_tokens_schedule in {
            "heuristic",
            "heuristic_transient",
        }:
            # len(scores[0])-1 is the number of candidates according to the target tokenizer.
            if num_matches == len(scores[0]) - 1:
                self.num_assistant_tokens += 2.0
            else:
                self.num_assistant_tokens = max(1.0, self.num_assistant_tokens - 1.0)

        # The assistant's confidence threshold is adjusted throughout the speculative iterations to reduce the number of unnecessary draft and target forward passes. The costs are estimated based on the ROC curve, which considers the probability of the draft token and its match with the target. A cost of 25% is assigned to false positives and 75% to false negatives.
        # This adaptation is not compatible with UAG, as it relies on the number of matched tokens based on the draft vocabulary, which is unavailable in UAG.
        if (
            is_sklearn_available()
            and self.assistant_model.generation_config.assistant_confidence_threshold
            and type(self) is AssistedCandidateGenerator
        ):
            # update self.matches
            # self.matches.extend([1] * num_matches)
            if len(self.probs) > len(self.matches):
                self.matches.append(0)

            # update self.probs
            excess_length = len(self.probs) - len(self.matches)
            if excess_length > 0:
                del self.probs[-excess_length:]

            if (
                len(self.probs) > 5 and {0, 1}.issubset(self.matches)
            ):  # require at least 5 samples to calculate the ROC curve and at least one positive and one negative sample
                fpr, tpr, thresholds = roc_curve(self.matches, self.probs)
                fnr = 1 - tpr

                # Calculate the cost for each threshold
                costs = fpr + 3 * fnr

                # Find the threshold that minimizes the cost
                optimal_threshold_index = np.argmin(costs)
                best_threshold = thresholds[optimal_threshold_index]

                self.assistant_model.generation_config.assistant_confidence_threshold = best_threshold


class AssistedCandidateGeneratorDifferentTokenizers(AssistedCandidateGenerator):
    """
    `CandidateGenerator` class to be used for Universal Assisted Generation (UAD): assisted generation with different tokenizers
    for the assistant and main models. This class generates candidates through the use of a smaller
    model.

    The main model input tokens are re-encoded into assistant model tokens, then candidate tokens are generated in the assistant encoding, which are
    in turn re-encoded into main model candidate tokens. Validation then proceeds as explained above.
    The re-encoding steps involve decoding token ids into text and then encoding the text using a different tokenizer.
    Since re-encoding the tokens may result in tokenization discrepancies, UAD finds the longest common subsequence between the source and target encodings,
    to ensure the new tokens include the correct prompt suffix.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        assistant_model (`PreTrainedModel`):
            The model to be used for generating candidates. This model should be smaller than the main model.
        target_tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for the target model.
        assistant_tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for the assistant model.
        generation_config (`~generation.GenerationConfig`, *optional*):
            The generation configuration to be used as base parametrization for the generation call.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        model_kwargs (`Dict`):
            The keyword arguments that will be passed to the main model, and are used as base inputs for the assistant
            model as well.
        inputs_tensor (`torch.Tensor`, *optional*):
            The model input tensor. In encoder-decoder models, this is the encoder input.
    """

    def __init__(
        self,
        input_ids: torch.LongTensor,
        assistant_model: "PreTrainedModel",
        target_tokenizer: "PreTrainedTokenizerBase",
        assistant_tokenizer: "PreTrainedTokenizerBase",
        generation_config: "GenerationConfig",
        model_kwargs: Dict,
        inputs_tensor: Optional[torch.Tensor] = None,
        logits_processor: "LogitsProcessorList" = None,
    ):
        super().__init__(input_ids, assistant_model, generation_config, model_kwargs, inputs_tensor, logits_processor)

        self.target_tokenizer = target_tokenizer
        self.assistant_tokenizer = assistant_tokenizer
        self.prev_assistant_ids = None
        self.prev_assistant_output_ids = None
        self.prev_new_assistant_ids = None
        self.prev_padding_length = None
        self.target_lookbehind = assistant_model.generation_config.target_lookbehind
        self.assistant_lookbehind = assistant_model.generation_config.assistant_lookbehind

    @staticmethod
    def _get_longest_diag_dict(input_matrix, nonzero_idx):
        """
        Calculates the length of the longest diagonal sequence in a given matrix.
        Args:
            input_matrix (torch.Tensor): The input matrix.
            nonzero_idx (torch.Tensor): The indices of the non-zero elements in the matrix.
        Returns:
            dict: A dictionary where the keys are the indices of the non-zero elements and the values are the lengths of the longest diagonal sequences starting from those indices.
        """

        visited = set()
        diags = {}
        for idx in nonzero_idx:
            start_idx = torch.clone(idx)
            tuple_start_idx = tuple(start_idx.tolist())

            if tuple_start_idx in visited:
                continue

            visited.add(tuple_start_idx)
            cur_diag_len = 1
            start_idx += 1
            while start_idx[0] < input_matrix.shape[0] and start_idx[1] < input_matrix.shape[1]:
                tuple_start_idx = tuple(start_idx.tolist())
                visited.add(tuple_start_idx)

                if input_matrix[start_idx[0], start_idx[1]] == 1:
                    cur_diag_len += 1
                    start_idx += 1
                else:
                    break

            diags[idx] = cur_diag_len
        return diags

    @staticmethod
    def _get_longest_diag_index(input_matrix):
        """
        Returns the start index and length of the longest diagonal in the given input.
        Args:
            input_matrix (numpy.ndarray): The input matrix.
        Returns:
            tuple: A tuple containing the start index and length of the longest diagonal.
        """

        diags = AssistedCandidateGeneratorDifferentTokenizers._get_longest_diag_dict(
            input_matrix, input_matrix.nonzero()
        )
        diags_values = list(diags.values())
        diags_keys = list(diags.keys())
        best_diag = np.argmax(diags_values)
        diag_start_index = diags_keys[best_diag]
        diag_start_length = diags_values[best_diag]
        return diag_start_index, diag_start_length

    @staticmethod
    def _get_tokens_diag(prompt, prompt_plus_new_tokens):
        """
        Input:
            prompt: 2D array of shape (batch_size, prompt_length), represents the original prompt tokens
            prompt_plus_new_tokens: 2D array of shape (batch_size, prompt_length), represents the suffix of the original prompt, with additional new tokens.
        Output:
            discrepancy_length: int, represents the number of tokens that need to be replaced from prompt
            new_tokens_only: 2D array of shape (batch_size, new_token_length), represents the new tokens that are not in prompt
            discrepancy_only: 2D array of shape (batch_size, discrepancy_length), represents the new tokens that are in prompt but not in prompt_plus_new_tokens
        """
        compare_mat = prompt_plus_new_tokens.T == prompt
        if not torch.is_tensor(compare_mat):
            compare_mat = torch.tensor(compare_mat)

        compare_mat_int = compare_mat.to(int)

        if not compare_mat_int.any().item():
            # empty intersection between prompt and prompt_plus_new_tokens
            return None, None, None

        longest_location, longest_diag_length = AssistedCandidateGeneratorDifferentTokenizers._get_longest_diag_index(
            compare_mat_int
        )
        # print(f"longest_location: {longest_location}; longest_diag_length: {longest_diag_length}")
        new_token_start_index = longest_location[0] + longest_diag_length
        discrepancy_with_old = longest_location[1] + longest_diag_length
        discrepancy_length = (prompt.shape[1] - discrepancy_with_old).item()
        new_tokens_only = prompt_plus_new_tokens[:, new_token_start_index + discrepancy_length :]
        discrepancy_only = prompt_plus_new_tokens[
            :, new_token_start_index : new_token_start_index + discrepancy_length
        ]
        return discrepancy_length, new_tokens_only, discrepancy_only
    
    @staticmethod
    def _get_tokens_diag_bs(prompt, prompt_plus_new_tokens):
        """
        _get_tokens_diag for batch_size > 1
        Input:
            prompt: 2D array of shape (batch_size, prompt_length), represents the original prompt tokens
            prompt_plus_new_tokens: 2D array of shape (batch_size, prompt_length), represents the suffix of the original prompt, with additional new tokens.
        Output:
            discrepancy_length: int, represents the number of tokens that need to be replaced from prompt
            new_tokens_only: 2D array of shape (batch_size, new_token_length), represents the new tokens that are not in prompt
            discrepancy_only: 2D array of shape (batch_size, discrepancy_length), represents the new tokens that are in prompt but not in prompt_plus_new_tokens
        """
        bs = prompt.shape[0]
        discrepancy_length_list = []
        new_tokens_only_list = []
        discrepancy_only_list = []
        for batch_index in range(bs):
            discrepancy_length, new_tokens_only, discrepancy_only = AssistedCandidateGeneratorDifferentTokenizers._get_tokens_diag(
                prompt[batch_index:batch_index+1,:], prompt_plus_new_tokens[batch_index:batch_index+1,:]
            )
            # TODO handle edge case where return None, None, None
            discrepancy_length_list.append(discrepancy_length)
            new_tokens_only_list.append(new_tokens_only)
            discrepancy_only_list.append(discrepancy_only)
        # min_new_num = min([new_tokens_only.shape[1] for new_tokens_only in new_tokens_only_list])
        # new_tokens_only_list = [new_tokens_only[:,:min_new_num] for new_tokens_only in new_tokens_only_list]
        
        # new_tokens_only_tensor = torch.cat(new_tokens_only_list, dim=0)
        
        # return discrepancy_length_list, new_tokens_only_tensor, discrepancy_only_list
        return discrepancy_length_list, new_tokens_only_list, discrepancy_only_list
    
    @staticmethod
    def _get_tokens_diag_bs_2(prompt, prompt_plus_new_tokens, pad_token):
        """_get_tokens_diag for batch_size > 1
        
        Input:
            prompt: 2D array of shape (batch_size, prompt_length), represents the original prompt tokens
            prompt_plus_new_tokens: 2D array of shape (batch_size, prompt_length), represents the suffix of the original prompt, with additional new tokens.
        Output:
            new_tokens_only: 2D array of shape (batch_size, new_token_length), represents the new tokens that are not in prompt
        """
        _, new_tokens_only_list, _ = AssistedCandidateGeneratorDifferentTokenizers._get_tokens_diag_bs(
            prompt, prompt_plus_new_tokens
        )
        
        max_new_num = max([new_tokens_only.shape[1] for new_tokens_only in new_tokens_only_list])
        new_tokens_only_list_padding = []
        for new_tokens_only in new_tokens_only_list:
            if new_tokens_only.shape[1] < max_new_num:
                new_tokens_only_list_padding.append(
                    torch.cat([torch.ones((1, max_new_num - new_tokens_only.shape[1]), dtype=new_tokens_only.dtype, device=new_tokens_only.device) * pad_token, new_tokens_only], dim=-1)
                )
            else:
                new_tokens_only_list_padding.append(new_tokens_only)
        
        return torch.cat(new_tokens_only_list_padding, dim=0)

    def convert_source_tokens_to_target_tokens(
        self,
        input_ids,
        source_tokenizer,
        destination_tokenizer,
    ):
        """
        Convert token IDs from one tokenizer to another.
        Args:
            input_ids: The input token IDs.
            source_tokenizer: The source tokenizer.
            destination_tokenizer: The destination tokenizer.
        Returns:
            The converted token IDs.
        """
        text = source_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        # print(f"source_text: {text}")
        # for one_text in text:
        #     dest_id = destination_tokenizer(one_text, add_special_tokens=True, return_tensors="pt")["input_ids"]
        #     print(f"dest_id: {dest_id}")
        dest_ids = destination_tokenizer(text, add_special_tokens=True, return_tensors="pt", padding=True, padding_side="left")["input_ids"]
        return dest_ids.to(input_ids.device)
    
    
    def _prepare_assistant_inputs(self, input_ids, convert_kwargs):
        """
        Prepare the assistant inputs by converting the source tokens to the assistant tokens.
        Args:
            input_ids: The input token IDs.
            convert_kwargs: The keyword arguments for the conversion function.
        Returns:
            The assistant input token IDs.
        """
        remove_from_pkv = 0
        # input_ids contains all target prompt input ids and some new target input ids
        start_index_in_target_window = input_ids.shape[1] - self.target_lookbehind

        new_assistant_ids = self.convert_source_tokens_to_target_tokens(
            input_ids[:, start_index_in_target_window:], **convert_kwargs
        )
        prompt_use_length = new_assistant_ids.shape[1]
        prompt_use = self.prev_assistant_ids[:, -prompt_use_length:]
        
        # print(f"prompt_use: {prompt_use}")
        # print(f"new_assistant_ids: {new_assistant_ids}")
        
        discrepancy_length_list, new_tokens_only_list, discrepancy_only_list = (
            AssistedCandidateGeneratorDifferentTokenizers._get_tokens_diag_bs(prompt_use, new_assistant_ids)
        )
        # assistant_input_ids = self.prev_assistant_ids
        # print(f"discrepancy_length_list: {discrepancy_length_list}")
        # print(f"new_tokens_only_list: {new_tokens_only_list}")
        # print(f"discrepancy_only_list: {discrepancy_only_list}")
        
        assistant_input_ids_list = [self.prev_assistant_ids[idx:idx+1, :] for idx in range(self.prev_assistant_ids.shape[0])]
        
        new_assistant_input_ids_list = []
        idx = 0
        max_len = 0
        for discrepancy_length, new_tokens_only, discrepancy_only, assistant_input_ids in \
            zip(discrepancy_length_list, new_tokens_only_list, discrepancy_only_list, assistant_input_ids_list):
            if new_tokens_only is not None:
                if discrepancy_length > 0 and discrepancy_only.shape[1] > 0:
                    if discrepancy_length == discrepancy_only.shape[1]:
                        assistant_input_ids[:, -discrepancy_length:] = discrepancy_only

                    elif discrepancy_length > discrepancy_only.shape[1]:
                        discrepancy_length_diff = discrepancy_length - discrepancy_only.shape[1]
                        assistant_input_ids = assistant_input_ids[:, :-discrepancy_length_diff]
                        assistant_input_ids[:, -discrepancy_only.shape[1] :] = discrepancy_only

                    remove_from_pkv = max(remove_from_pkv, discrepancy_length)

                if new_tokens_only.shape[1] > 0:
                    assistant_input_ids = torch.cat([assistant_input_ids, new_tokens_only], dim=-1)
            else:
                # edge case: in case of no intersection between prompt and new_assistant_ids
                assistant_input_ids = torch.cat([assistant_input_ids, new_assistant_ids[idx:idx+1,:]], dim=-1)
            idx += 1
            
            max_len = max(max_len, assistant_input_ids.shape[1])
            new_assistant_input_ids_list.append(assistant_input_ids)
        
        tmp_list = []
        for assistant_input_ids in new_assistant_input_ids_list:
            if assistant_input_ids.shape[1] < max_len:
                tmp_list.append(
                    torch.cat([torch.ones(1, max_len-assistant_input_ids.shape[1], dtype=assistant_input_ids.dtype, device=assistant_input_ids.device)*self.assistant_tokenizer.pad_token_id, assistant_input_ids], dim=-1)
                )
            else:
                tmp_list.append(assistant_input_ids)
        assistant_input_ids = torch.cat(tmp_list, dim=0)
        
        return assistant_input_ids, remove_from_pkv, discrepancy_length_list, new_tokens_only_list
    
    @staticmethod
    def longest_common_prefix(t1, t2):
        assert t1.dim() == 1
        assert t2.dim() == 1
        min_len = min(t1.shape[0], t2.shape[0])
        for i in range(min_len):
            if t1[i] != t2[i]:
                return i
        return min_len
    
    def shape_kv_caches(self, past_key_values, discrepancy_length_list, new_tokens_only_list):
        prev_new_token_kv = self.prev_new_assistant_ids.shape[-1] - 1
        bs = len(discrepancy_length_list)
        remove_from_pkv_list = []
        for bs_idx in range(bs):
            discrepancy_length = discrepancy_length_list[bs_idx]
            new_tokens_only = new_tokens_only_list[bs_idx]
            prev_new_ids = self.prev_new_assistant_ids[bs_idx]
            if discrepancy_length > 0:
                # case 1, the last prompts has changed, so all new kv should be removed
                remove_from_pkv = prev_new_token_kv + discrepancy_length
            else:
                accepted_len = AssistedCandidateGeneratorDifferentTokenizers.longest_common_prefix(
                    new_tokens_only.squeeze(dim=0), prev_new_ids
                )
                remove_from_pkv = max(0, prev_new_token_kv - accepted_len)
            remove_from_pkv_list.append(remove_from_pkv)
            
        remove_len = min(remove_from_pkv_list)
        if remove_len > 0:
            past_key_values = _crop_past_key_values(self.assistant_model, past_key_values, max_length=-remove_len)
            
        shift = torch.tensor([x - remove_len for x in remove_from_pkv_list], dtype=torch.int, device=self.assistant_model.device)
        past_key_values = _crop_past_key_values(self.assistant_model, past_key_values, shift=shift)
        
        self.prev_padding_length = [x+y for x, y in zip(remove_from_pkv_list, self.prev_padding_length)]
        left_cut = min(self.prev_padding_length)
        if left_cut > 0:
            past_key_values = _crop_past_key_values(self.assistant_model, past_key_values, left_cut=left_cut)
            self.prev_padding_length = [x-left_cut for x in self.prev_padding_length]
            
        # print(f"remove_len: {remove_len}; shift: {shift}; left_cut: {left_cut}")
        
        return past_key_values
    
    def get_padding_length(self, assistant_input_ids):
        pad_token_id = self.assistant_tokenizer.pad_token_id
        bs = assistant_input_ids.shape[0]
        padding_length_list = []
        for bs_idx in range(bs):
            padding_length = 0
            for idx in range(assistant_input_ids.shape[1]):
                if assistant_input_ids[bs_idx, idx] == pad_token_id:
                    padding_length += 1
                else:
                    break
            padding_length_list.append(padding_length)
        return padding_length_list
    
    def get_candidates(self, input_ids: torch.LongTensor) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
        """
        Fetches the candidates to be tried for the current input.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)

        Return:
            `torch.LongTensor` of shape `(batch_size, candidate_length)` containing the candidate sequences to be
            assessed by the model and a `torch.FloatTensor` of shape `(batch_size, candidate_length,
            vocabulary_size)` containing the logits associated to each candidate.
        """
        print("******In get_candidates*******")
        max_new_tokens = int(self.num_assistant_tokens)
        if max_new_tokens == 0:
            return input_ids, None
        
        input_ids = input_ids.to(self.assistant_model.device)
        convert_kwargs = {
            "source_tokenizer": self.target_tokenizer,
            "destination_tokenizer": self.assistant_tokenizer,
        }
        remove_from_pkv = 0
        
        all_assistant_ids = self.convert_source_tokens_to_target_tokens(
            input_ids, **convert_kwargs
        )
        # print(f"input_ids in assistant model: {all_assistant_ids}")
        # for i in range(all_assistant_ids.shape[0]):
        #     print(f"input_text {i} in assistant model: {[self.assistant_tokenizer.decode(all_assistant_ids[i], skip_special_tokens=True)]}")
        
        # Since re-encoding the tokens may result in tokenization discrepancies, we use 2 look behind values
        # (one for each conversion) which mark where to start looking for the overlap between the
        # source and target encodings, to ensure the new tokens include the correct prompt suffix.
        if self.prev_assistant_ids is not None and input_ids.shape[1] > self.target_lookbehind:
            # # input_ids contains all target prompt input ids and some new target input ids
            # start_index_in_target_window = input_ids.shape[1] - self.target_lookbehind

            # new_assistant_ids = self.convert_source_tokens_to_target_tokens(
            #     input_ids[:, start_index_in_target_window:], **convert_kwargs
            # )
            # prompt_use_length = new_assistant_ids.shape[1]
            # prompt_use = self.prev_assistant_ids[:, -prompt_use_length:]

            # discrepancy_length, new_tokens_only, discrepancy_only = (
            #     AssistedCandidateGeneratorDifferentTokenizers._get_tokens_diag(prompt_use, new_assistant_ids)
            # )
            # assistant_input_ids = self.prev_assistant_ids

            # if new_tokens_only is not None:
            #     if discrepancy_length > 0 and discrepancy_only.shape[1] > 0:
            #         if discrepancy_length == discrepancy_only.shape[1]:
            #             assistant_input_ids[:, -discrepancy_length:] = discrepancy_only

            #         elif discrepancy_length > discrepancy_only.shape[1]:
            #             discrepancy_length_diff = discrepancy_length - discrepancy_only.shape[1]
            #             assistant_input_ids = assistant_input_ids[:, :-discrepancy_length_diff]
            #             assistant_input_ids[:, -discrepancy_only.shape[1] :] = discrepancy_only

            #         remove_from_pkv = discrepancy_length

            #     if new_tokens_only.shape[1] > 0:
            #         assistant_input_ids = torch.cat([assistant_input_ids, new_tokens_only], dim=-1)
            # else:
            #     # edge case: in case of no intersection between prompt and new_assistant_ids
            #     assistant_input_ids = torch.cat([assistant_input_ids, new_assistant_ids], dim=-1)
            assistant_input_ids, remove_from_pkv, discrepancy_length_list, new_tokens_only_list = \
                self._prepare_assistant_inputs(input_ids, convert_kwargs)
        else:
            assistant_input_ids = self.convert_source_tokens_to_target_tokens(input_ids, **convert_kwargs)
            self.prev_padding_length = self.get_padding_length(assistant_input_ids)
            
        # print(f"assistant_input_ids: {assistant_input_ids}")
        # print(f"assistant_input_text: {self.assistant_tokenizer.batch_decode(assistant_input_ids, skip_special_tokens=True)}")
        
        self.prev_assistant_ids = assistant_input_ids
        new_cur_len = assistant_input_ids.shape[-1]
        min_new_tokens = max(min(max_new_tokens, self.main_model_min_length - new_cur_len), 0)
        # print(f"max_new_tokens: {max_new_tokens}; min_new_tokens: {min_new_tokens}")

        # 1. If it is not the first round of candidate generation, prepare the inputs based on the input_ids length
        # (which implicitly contains the number of accepted candidates from the previous round)
        has_past_key_values = self.assistant_kwargs.get("past_key_values", None) is not None
        if has_past_key_values:
            # padding 后kv cache怎么对齐？
            new_cache_size = new_cur_len - 1 - remove_from_pkv
            past_kv = self.assistant_kwargs.get("past_key_values", None)
            # print(f"before crop seen_tokens: {past_kv[0][0].shape}")
            self.assistant_kwargs["past_key_values"] = self.shape_kv_caches(
                past_kv, discrepancy_length_list, new_tokens_only_list
            )
            # self.assistant_kwargs["past_key_values"] = _crop_past_key_values(
            #     self.assistant_model, self.assistant_kwargs["past_key_values"], new_cache_size - 1
            # )  # the assistant does not have the token after the last match, hence the -1
            # past_kv = self.assistant_kwargs.get("past_key_values", None)
            # print(f"after crop seen_tokens: {past_kv[0][0].shape}")

            self.assistant_kwargs = _prepare_attention_mask(
                self.assistant_kwargs, new_cur_len, self.assistant_model.config.is_encoder_decoder
            )
            self.assistant_kwargs = _prepare_token_type_ids(self.assistant_kwargs, new_cur_len)
            
            
        # 1.5 prepare attention mask
        self.assistant_kwargs["attention_mask"] = torch.ones(assistant_input_ids.shape, device=assistant_input_ids.device)
        for bs_idx in range(len(self.prev_padding_length)):
            self.assistant_kwargs["attention_mask"][bs_idx, :self.prev_padding_length[bs_idx]] = 0

        # 2. Forecast next N tokens using the assistant model.
        assistant_generation_kwargs = {
            self.input_ids_key: assistant_input_ids,
            "min_new_tokens": min_new_tokens,
            "max_new_tokens": max_new_tokens,
            "generation_config": self.generation_config,
            "logits_processor": self.logits_processor,
        }
        # self.assistant_kwargs.pop("attention_mask", None)
        # self.assistant_kwargs.pop("past_key_values", None)
        
        # print(f"type generation_config: {type(self.generation_config)}")
        # print(f"assistant_generation_kwargs keys: {assistant_generation_kwargs.keys()}")
        # print(f"assistant_kwargs keys: {self.assistant_kwargs.keys()}")
        
        assistant_output = self.assistant_model.generate(**assistant_generation_kwargs, **self.assistant_kwargs)
        self.prev_assistant_output_ids = assistant_output.sequences
        # print(f"assistant_output.sequences: {assistant_output.sequences}")
        # for i in range(assistant_output.sequences.shape[0]):
        #     print(f"assistant_output.sequences text {i}: {[self.assistant_tokenizer.decode(assistant_output.sequences[i], skip_special_tokens=True)]}")
            
        new_assistant_ids_length = assistant_output.sequences.shape[1] - assistant_input_ids.shape[1]
        self.prev_new_assistant_ids = assistant_output.sequences[:, -new_assistant_ids_length:]
        # print(f"prev_new_assistant_ids: {self.prev_new_assistant_ids}")
        # for i in range(self.prev_new_assistant_ids.shape[0]):
        #     print(f"prev_new_assistant_ids text {i}: {[self.assistant_tokenizer.decode(self.prev_new_assistant_ids[i], skip_special_tokens=True)]}")
        
        num_prev_assistant = self.prev_assistant_ids.shape[1]
        start_assistant_look_index = num_prev_assistant - self.assistant_lookbehind
        if start_assistant_look_index < 0:
            start_assistant_look_index = 0

        new_target_ids_from_window = self.convert_source_tokens_to_target_tokens(
            assistant_output.sequences[:, start_assistant_look_index:],
            source_tokenizer=self.assistant_tokenizer,
            destination_tokenizer=self.target_tokenizer,
        )
        target_prompt_use_length = new_target_ids_from_window.shape[1]

        target_prompt_use = input_ids[:, -target_prompt_use_length:]
        
        # _, target_new_tokens_only, _ = AssistedCandidateGeneratorDifferentTokenizers._get_tokens_diag(
        #     target_prompt_use, new_target_ids_from_window
        # )
        target_new_tokens_only = AssistedCandidateGeneratorDifferentTokenizers._get_tokens_diag_bs_2(
            target_prompt_use, new_target_ids_from_window, pad_token=self.target_tokenizer.pad_token_id
        )
        # print(f"target_new_tokens_only: {target_new_tokens_only}")

        new_target_ids = input_ids

        if target_new_tokens_only is not None:
            if target_new_tokens_only.shape[1] > 0:
                new_target_ids = torch.cat([new_target_ids, target_new_tokens_only], dim=-1)
        else:
            # edge case: in case of no intersection between prompt and new_target_ids
            new_target_ids = torch.cat([new_target_ids, new_target_ids_from_window], dim=-1)

        if hasattr(self.generation_config, "max_length"):
            new_target_ids = new_target_ids[:, : self.generation_config.max_length]

        # 3. Update variables for the next round of candidate generation
        self.assistant_kwargs["past_key_values"] = assistant_output.past_key_values
        
        print("********************************")

        # 4. Prepare variables for output
        if input_ids.shape[1] >= new_target_ids.shape[1]:
            return input_ids, None

        return new_target_ids, None


class PromptLookupCandidateGenerator(CandidateGenerator):
    """
    `CandidateGenerator` class to be used for prompt lookup generation. This class generates candidates by looking up
    likely continuations in the provided prompt (input_ids) itself.
    Read the following blog post for more information: https://github.com/apoorvumang/prompt-lookup-decoding

    Args:
        max_matching_ngram_size (`int`):
            The maximum ngram size to be considered for matching in the prompt
        num_output_tokens (`int`):
            The number of tokens to be output as candidate tokens.
        max_length (`int`):
            The number of total maximum tokens that can be generated. For decoder-only models that includes the prompt length.
            Defaults to 20, which is the max length used as default in generation config.
    """

    def __init__(
        self,
        eos_token_id: torch.Tensor = None,
        num_output_tokens: int = 10,
        max_matching_ngram_size: int = None,
        max_length: int = 20,
    ):
        self.num_output_tokens = num_output_tokens
        self.max_matching_ngram_size = max_matching_ngram_size if max_matching_ngram_size else 2
        self.max_length = max_length
        self.eos_token_id = eos_token_id

        if self.max_matching_ngram_size <= 0 or self.num_output_tokens <= 0:
            raise ValueError("Invalid max_matching_ngram_size or num_output_tokens")

    def get_candidates(self, input_ids: torch.LongTensor) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
        """
        Fetches the candidates to be tried for the current input.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)

        Return:
            `torch.LongTensor` of shape `(num_candidates, candidate_length)`: The candidate sequences to be tried.
        """
        input_length = input_ids.size(1)

        # Don't generate more than `max_length - 1` candidates since the target model generates one extra token.
        if self.max_length == input_length + 1:
            return input_ids, None

        chosen_ids = None
        match_found = False
        for ngram_size in range(min(self.max_matching_ngram_size, input_length - 1), 0, -1):
            # Create sliding windows of size ngram_size
            windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)

            # Convert ngram to a tensor for comparison
            ngram_tensor = input_ids[0, -ngram_size:]

            # Find where the windows match the ngram
            matches = (windows == ngram_tensor).all(dim=2)

            # Get the indices of matches
            match_indices = matches.nonzero(as_tuple=True)[1]

            # Iterate through match indices to find a valid continuation
            for idx in match_indices:
                start_idx = idx + ngram_size
                end_idx = start_idx + self.num_output_tokens
                end_idx = min(end_idx, input_length, self.max_length)

                if start_idx < end_idx:
                    chosen_ids = input_ids[0, start_idx:end_idx]
                    match_found = True

                    # remove remaining candidate ids if an "eos" token is found, otherwise the target model may
                    # accept eos and the rest as valid, thus not stopping generation after "eos"
                    # NOTE: below code is written based on the fact that assisted decoding supports only bs=1
                    mask = isin_mps_friendly(chosen_ids, self.eos_token_id)
                    match_indices_eos = torch.nonzero(mask)
                    if match_indices_eos.numel() > 0:
                        first_eos_index = match_indices_eos[0].item()
                        chosen_ids = chosen_ids[:first_eos_index]
                    break
            if match_found:
                break

        if chosen_ids is None or len(chosen_ids) == 0:
            # In case we didn't find a match return the input sequence unchanged, reverts back to autoregressive decoding
            return input_ids, None

        # Now need extend input_ids with chosen_ids
        chosen_ids = chosen_ids.unsqueeze(0)
        candidate_input_ids = torch.cat((input_ids, chosen_ids), dim=1)
        # assisted_generation expects logits as well, but we don't have those here, so returning None
        return candidate_input_ids, None

    def update_candidate_strategy(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int):
        """
        Updates the candidate generation strategy based on the outcomes.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, candidate_length, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            num_matches (`int`):
                The number of matches between the candidate sequences and the model predictions.
        """
        # Currently does nothing
        return


class EarlyExitCandidateGenerator(AssistedCandidateGenerator):
    """
    `CandidateGenerator` class to be used for assisted generation and speculative decoding. This class generates
    candidates through the use of **the model itself**, exiting early. Can only be used with models that support early
    exit, e.g., `facebook/layerskip-llama3.2-1B`.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        assistant_model (`PreTrainedModel`):
            The original model. This model must support early exit (i.e. is trained to compute logits in earlier
            layers).
        generation_config (`~generation.GenerationConfig`, *optional*):
            The generation configuration to be used as base parametrization for the generation call.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        model_kwargs (`Dict`):
            The keyword arguments that will be passed to the main model, and are used as base inputs for the assistant
            model as well.
        inputs_tensor (`torch.Tensor`, *optional*):
            The model input tensor. In encoder-decoder models, this is the encoder input.
    """

    def __init__(
        self,
        input_ids: torch.LongTensor,
        assistant_model: "PreTrainedModel",
        generation_config: "GenerationConfig",
        model_kwargs: Dict,
        inputs_tensor: Optional[torch.Tensor] = None,
        logits_processor: "LogitsProcessorList" = None,
    ):
        super().__init__(
            input_ids=input_ids,
            assistant_model=assistant_model,
            generation_config=generation_config,
            model_kwargs=model_kwargs,
            inputs_tensor=inputs_tensor,
            logits_processor=logits_processor,
        )
        # We have to move early exit out of the generation config, otherwise the assistant will also call `generate`
        # with early exit
        self.assistant_early_exit = self.generation_config.assistant_early_exit
        self.generation_config.assistant_early_exit = None

    def get_candidates(self, input_ids: torch.LongTensor) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
        # Temporarily sets the number of hidden layers to the early exit value
        base_model = getattr(self.assistant_model, self.assistant_model.base_model_prefix)
        original_num_hidden_layers = base_model.config.num_hidden_layers
        base_model.config.num_hidden_layers = self.assistant_early_exit
        candidate_ids, candidate_logits = super().get_candidates(input_ids)
        base_model.config.num_hidden_layers = original_num_hidden_layers
        return candidate_ids, candidate_logits


def _crop_past_key_values(model, past_key_values, max_length=None, n_matches=None, left_cut=None, shift=None):
    """Crops the past key values up to a certain maximum length."""
    new_past = []
    if model.config.is_encoder_decoder:
        for idx in range(len(past_key_values)):
            if left_cut is None:
                k_cache = past_key_values[idx][0][:, :, :max_length, :]
                v_cache = past_key_values[idx][1][:, :, :max_length, :]
            else:
                k_cache = past_key_values[idx][0][:, :, left_cut:, :]
                v_cache = past_key_values[idx][1][:, :, left_cut:, :]

            if n_matches is not None:
                for batch_idx in range(len(n_matches)):
                    num_roll_left = n_matches.max() - n_matches[batch_idx]
                    if num_roll_left > 0:
                        # TODO(PVP) - check mem usage
                        # k_cache[batch_idx].index_copy_(1, torch.arange(num_roll_left, maximum_length, device=k_cache.device), k_cache[batch_idx][:, :-num_roll_left].clone())
                        # v_cache[batch_idx].index_copy_(1, torch.arange(num_roll_left, maximum_length, device=v_cache.device), v_cache[batch_idx][:, :-num_roll_left].clone())
                        k_cache[batch_idx][:, num_roll_left:] = k_cache[batch_idx][:, :-num_roll_left].clone()
                        v_cache[batch_idx][:, num_roll_left:] = v_cache[batch_idx][:, :-num_roll_left].clone()

            new_past.append(
                (
                    # past_key_values[idx][0][:, :, :max_length, :],
                    # past_key_values[idx][1][:, :, :max_length, :],
                    k_cache,
                    v_cache,
                    past_key_values[idx][2],
                    past_key_values[idx][3],
                )
            )
        past_key_values = tuple(new_past)
    # gptbigcode is special and stores kv in shape (batch_size, seq_len, dim), if it's a multi_query model
    elif "gptbigcode" in model.__class__.__name__.lower() or (
        model.config.architectures is not None and "gptbigcode" in model.config.architectures[0].lower()
    ):
        if model.config.multi_query:
            for idx in range(len(past_key_values)):
                past_key_values[idx] = past_key_values[idx][:, :max_length, :]
        else:
            for idx in range(len(past_key_values)):
                past_key_values[idx] = past_key_values[idx][:, :, :max_length, :]
    elif isinstance(past_key_values, DynamicCache):
        if max_length is not None:
            past_key_values.crop(max_length)
        elif shift is not None:
            # TODO maybe performance can be improved by using torch.roll instead of index_copy
            batch_size = shift.shape[0]
            for bs_idx in range(batch_size):
                if shift[bs_idx] <= 0: continue
                for layer_index in range(len(past_key_values.key_cache)):
                    past_key_values.key_cache[layer_index][bs_idx, :, shift[bs_idx]:, :] = \
                        past_key_values.key_cache[layer_index][bs_idx, :, :-shift[bs_idx], :].clone()
                    past_key_values.key_cache[layer_index][bs_idx, :, :shift[bs_idx], :] = 0
                    past_key_values.value_cache[layer_index][bs_idx, :, shift[bs_idx]:, :] = \
                        past_key_values.value_cache[layer_index][bs_idx, :, :-shift[bs_idx], :].clone()
                    past_key_values.value_cache[layer_index][bs_idx, :, :shift[bs_idx], :] = 0
        elif left_cut is not None:
            for layer_index in range(len(past_key_values.key_cache)):
                past_key_values.key_cache[layer_index] = past_key_values.key_cache[layer_index][:, :, left_cut:, :]
                past_key_values.value_cache[layer_index] = past_key_values.value_cache[layer_index][:, :, left_cut:, :]
            past_key_values._seen_tokens -= left_cut
    elif past_key_values is not None:
        if max_length is not None:
            for idx in range(len(past_key_values)):
                if past_key_values[idx] != ([], []):
                    new_past.append(
                        (
                            past_key_values[idx][0][:, :, :max_length, :],
                            past_key_values[idx][1][:, :, :max_length, :],
                        )
                    )
                else:
                    new_past.append((past_key_values[idx][0], past_key_values[idx][1]))
        elif shift is not None:
            bs = shift.shape[0]
            for idx in range(len(past_key_values)):
                if past_key_values[idx] != ([], []):
                    for bs_idx in range(bs):
                        if shift[bs_idx] > 0:
                            past_key_values[idx][0][bs_idx, :, shift[bs_idx]:, :] = past_key_values[idx][0][bs_idx, :, :-shift[bs_idx], :].clone()
                            past_key_values[idx][0][bs_idx, :, :shift[bs_idx], :] = 0
                            past_key_values[idx][1][bs_idx, :, shift[bs_idx]:, :] = past_key_values[idx][1][bs_idx, :, :-shift[bs_idx], :].clone()
                            past_key_values[idx][1][bs_idx, :, :shift[bs_idx], :] = 0
                new_past.append((past_key_values[idx][0], past_key_values[idx][1]))
        elif left_cut is not None:
            for idx in range(len(past_key_values)):
                if past_key_values[idx] != ([], []):
                    new_past.append(
                        (
                            past_key_values[idx][0][:, :, left_cut:, :],
                            past_key_values[idx][1][:, :, left_cut:, :],
                        )
                    )
                else:
                    new_past.append((past_key_values[idx][0], past_key_values[idx][1]))
        past_key_values = tuple(new_past)
    return past_key_values


def _prepare_attention_mask(model_kwargs: Dict[str, Any], new_length: int, is_encoder_decoder: bool) -> Dict[str, Any]:
    """Expands or crops the model's mask for decoding purposes, to the defined length"""

    mask_key = "decoder_attention_mask" if is_encoder_decoder else "attention_mask"
    if mask_key not in model_kwargs:
        return model_kwargs

    mask = model_kwargs[mask_key]
    mask_length_diff = new_length - mask.shape[1]

    if mask_length_diff < 0:
        model_kwargs[mask_key] = mask[:, :mask_length_diff]
    elif mask_length_diff > 0:
        model_kwargs[mask_key] = torch.cat([mask, mask.new_ones((mask.shape[0], mask_length_diff))], dim=-1)

    # Handle cross attention models
    if "cross_attention_mask" in model_kwargs:
        # Mllama case
        cross_mask = model_kwargs["cross_attention_mask"]
        if mask_length_diff < 0:
            model_kwargs["cross_attention_mask"] = cross_mask[:, :mask_length_diff]
        elif mask_length_diff > 0:
            new_mask = cross_mask[:, -1:, :, :].repeat(1, mask_length_diff, 1, 1)
            model_kwargs["cross_attention_mask"] = torch.cat([cross_mask, new_mask], dim=1)
    elif "image_attention_mask" in model_kwargs:
        # IDEFICS case
        cross_mask = model_kwargs["image_attention_mask"]
        if mask_length_diff < 0:
            model_kwargs["image_attention_mask"] = cross_mask[:, :mask_length_diff]
        elif mask_length_diff > 0:
            new_mask = cross_mask[:, -1:, :].repeat(1, mask_length_diff, 1)
            model_kwargs["image_attention_mask"] = torch.cat([cross_mask, new_mask], dim=1)

    return model_kwargs


def _prepare_token_type_ids(model_kwargs: Dict[str, Any], new_length: int) -> Dict[str, Any]:
    """Expands or crops the model's token_type_ids for decoding purposes, to the defined length"""
    if "token_type_ids" not in model_kwargs or model_kwargs["token_type_ids"] is None:
        return model_kwargs

    token_type_ids = model_kwargs["token_type_ids"]
    final_token_type = token_type_ids[:, -1].unsqueeze(-1)
    type_length_diff = new_length - token_type_ids.shape[1]

    if type_length_diff < 0:
        token_type_ids = token_type_ids[:, :type_length_diff]
    elif type_length_diff > 0:
        token_type_copies = final_token_type.repeat(1, type_length_diff)
        model_kwargs["token_type_ids"] = torch.cat([model_kwargs["token_type_ids"], token_type_copies], dim=-1)
    return model_kwargs
