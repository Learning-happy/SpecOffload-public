from typing import Optional

import torch
import time

def print_tensor(tensor: torch.Tensor, n_padding: int = 0, n_start_elem: int = 2, n_end_elem: int = 2, n_start_line: int = 2, n_end_line: int = 2, elem_format: str = "") -> None:
    def formatted_elem(elem: torch.Tensor) -> str:
        return f"{elem.item():{elem_format}}"
    def formatted_line(line: torch.Tensor) -> str:
        if line.shape[0] <= n_start_elem + n_end_elem:
            return "[ " + ", ".join(
                    [formatted_elem(elem) for elem in line]
                ) + " ]"
        else:
            return "[ " + ", ".join(
                    [formatted_elem(elem) for elem in line[:n_start_elem]]
                    + ["..."]
                    + [formatted_elem(elem) for elem in line[-n_end_elem:]]
                ) + " ]"
    
    if tensor.dim() == 0:
        print(" " * n_padding,
              "tensor()", sep="")
    elif tensor.dim() == 1:
        print(" " * n_padding,
              "tensor(",
              formatted_line(tensor),
              ")", sep="")
    elif tensor.dim() == 2:
        if tensor.shape[0] <= n_start_line + n_end_line:
            print(" " * n_padding,
                  "tensor([",
                  f",\n{' ' * (n_padding + 8)}".join(
                      [formatted_line(line) for line in tensor]
                  ),
                  "])", sep="")
        else:
            print(" " * n_padding,
                  "tensor([",
                  f",\n{' ' * (n_padding + 8)}".join(
                      [formatted_line(line) for line in tensor[:n_start_line]]
                      + ["..."]
                      + [formatted_line(line) for line in tensor[-n_end_line:]]
                  ),
                  "])", sep="")
    else:
        print(" " * n_padding, "Tensor with more than 2 dimensions")

def print_tensor_match(tensor: torch.Tensor, n_matches: torch.Tensor, n_padding: int = 0, n_start_elem: int = 10, n_start_line: int = 8, elem_format: str = "5d", match_color: str="green") -> None:
    def formatted_elem(elem: torch.Tensor, is_match: bool, match_color_control: str) -> str:
        if is_match:
            return f"{match_color_control}{elem.item():{elem_format}}\033[0m"
        else:
            return f"{elem.item():{elem_format}}"
    def formatted_line(line: torch.Tensor, n_match: int, match_color_control: str) -> str:
        return "[ " + ", ".join(
                [formatted_elem(line[i], i < n_match, match_color_control) for i in range(line.shape[0])]
            ) + " ]"
    
    if match_color:
        match_color = match_color.lower()
    if match_color == 'red':
        match_color_control = '\033[31m'
    elif match_color == 'green':
        match_color_control = '\033[32m'
    elif match_color == 'yellow':
        match_color_control = '\033[33m'
    elif match_color == 'blue':
        match_color_control = '\033[34m'
    else:
        match_color_control = ""
    
    if tensor.dim() == 2:
        tensor_cropped = tensor[:n_start_line, :n_start_elem]
        print(" " * n_padding,
              "tensor([",
              f",\n{' ' * (n_padding + 8)}".join(
                  [formatted_line(line, n_match, match_color_control) for line, n_match in zip(tensor_cropped, n_matches)]
              ),
              "])", sep="")
    else:
        print(" " * n_padding, "Invalid tensor comparison dimensions")
    
def print_log(
    *args,
    prefix: Optional[str] = None,
    thread_id: Optional[int] = None,
    iteration: Optional[int] = None,
    color: Optional[str] = None,
    background: Optional[str] = None,
    **kwargs
) -> None:
    if color:
        color = color.lower()
    if color == 'red':
        color_code = '31'
    elif color == 'green':
        color_code = '32'
    elif color == 'yellow':
        color_code = '33'
    elif color == 'blue':
        color_code = '34'
    else:
        color_code = None
    
    if background:
        background = background.lower()
    if background == 'white':
        background_code = '47'
    else:
        background_code = None
    
    if background_code:
        if color_code:
            color_code = color_code + ';' + background_code
        else:
            color_code = background_code
    
    if color_code:
        print(f"\033[{color_code}m", end="")
    
    if prefix:
        print(f"[{prefix}]", end=" ")
    if thread_id is not None:
        print(f"Th.{thread_id}", end="")
        print(":" if iteration is None else "", end=" ")
    if iteration is not None:
        print(f"Iter.{iteration}:", end=" ")
    
    end = kwargs.pop("end", "\n")
    kwargs["end"] = ""
    print(*args, **kwargs)
    
    if color_code:
        print("\033[0m", end="")
    print("", end=end)

class LogConfig:
    def __init__(
        self,
        timer: bool = False,
        layer_timer: bool = False,
        input_ids_shape: bool = False,
        input_ids: bool = False,
        generation_process: bool = False,
        generation_statistics: bool = False,
        attention_mask: bool = False,
        kv_cache: bool = False,
        model_inputs: bool = False,
        n_show_output: int = 4,
    ):
        self.timer = timer
        self.layer_timer = layer_timer
        self.input_ids_shape = input_ids_shape
        self.input_ids = input_ids
        self.generation_process = generation_process
        self.generation_statistics = generation_statistics
        self.attention_mask = attention_mask
        self.kv_cache = kv_cache
        self.model_inputs = model_inputs
        self.n_show_output = n_show_output
        
        self.time_base = None
    
    def start_timer(self):
        self.time_base = time.time()
        
    def get_time(self):
        if self.time_base is None:
            self.time_base = time.time()
        return time.time() - self.time_base