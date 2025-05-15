import argparse
import os

from mixtral_greedy import FiddlerMixtral


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mixtral-8x7B-v0.1",
        help="Model path. default `mistralai/Mixtral-8x7B-v0.1`.",
    )
    parser.add_argument(
        "--cpu-offload",
        type=int,
        default=1,
        choices=[0, 1],
        help="0: exeute at GPU (baseline), 1: offload to CPU.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="University of Washington is",
        help="Input text to generate.",
    )
    parser.add_argument(
        "--n-token",
        type=int,
        default=32,
        help="Number of tokens to generate.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for generation. Default is 1.",
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="openai_humaneval",
        help="Dataset for generation. Default is None.",
    )
    
    args = parser.parse_args()
    model = FiddlerMixtral(args)
    prefill_time, decode_time, hit_rate = model.generate(
        args.input, output_token=args.n_token,
        DATASET=args.dataset,
        GENERATION_BATCH_SIZE=args.batch_size
    )
    print(
        f"prefill_time: {prefill_time}, decode_time: {decode_time}, hit_rate: {hit_rate}"
    )
