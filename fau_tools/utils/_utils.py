import typing
import os
import time

import torch.nn as nn


def exit_with_error(): raise SystemExit(1)


def ensure_file_postfix(file_path: str, postfix: str) -> str:
  """To ensure `file_path` end with extension name `postfix`."""
  if postfix[0] != '.': postfix = f".{postfix}"
  return f"{os.path.splitext(file_path)[0]}{postfix}"


def create_folder(path: str) -> str|None:
  """
  Create a folder with `path`; if `path` is exists, it will add postfix automatically.
  If create successfully, will return folder path; else will return `None`.
  """
  if not os.path.exists(path): os.makedirs(path)
  else:
    post_num = 1
    while os.path.exists(f"{path}_{post_num}"): post_num += 1
    else:
      path = f"{path}_{post_num}"
      os.makedirs(path)

  if not os.path.isdir(path):
    from ._color_print import cprint
    cprint(f"Error: Create experiment folder named {path} failed.", color="red")
    return None

  return path


def calc_dataloader_sample_num(data_loader: nn.Module):
  total_dataset_sample_num = len(data_loader.dataset)
  if not data_loader.drop_last: return total_dataset_sample_num

  batch_size = min(len(data_loader.dataset), data_loader.batch_size)
  if total_dataset_sample_num % batch_size == 0: return total_dataset_sample_num
  return batch_size * (len(data_loader) - 1)


def calc_time(function: typing.Callable):
  """A decorator to display the running time of a function."""
  def wrapper(*args, **kwargs):
    print('-' * 15, "BEGIN", function.__name__, '-' * 15)
    BEGIN = time.time()
    res = function(*args, **kwargs)
    END = time.time()
    COST_TIME = time_to_human(END - BEGIN)
    print(f"{function.__name__} cost: {COST_TIME}")
    print('-' * 15, " END ", function.__name__, '-' * 15)
    return res

  wrapper.__name__ = function.__name__  # to keep the function origin name
  return wrapper


def time_to_human(time: float|int) -> str:
  """
  Convert time in seconds to the human-friendly time display.

  Parameters
  ----------
  time : time in seconds

  Returns
  -------
  A string with format HH:mm:ss;
  but if the time is more than one day, will be "MTOD";
  but if the time is less than one second, will be "minor".

  """
  time = int(time)
  second = time % 60; time //= 60
  minute = time % 60; time //= 60
  hour = time

  if hour >= 24: return "MTOD"  # more than one day

  if hour > 0: return f"{hour:02d}:{minute:02d}:{second:02d}"
  if minute > 0: return f"{minute:02d}:{second:02d}"
  if second > 0: return f"{second:02d}s"

  return "minor"


# ------------------------------------------------------------
# --------------- Auto calculate feature size
# ------------------------------------------------------------
# TODO: integrate to a class and test.
def __get_value_in_height_and_width(value: int|tuple[int, int], value_name: str) -> tuple[int, int]:
  if isinstance(value, tuple): return value
  if isinstance(value, int): return value, value

  raise TypeError(f"The type of {value_name} requires int|tuple, but got {type(value)}.")


def _calc_value_after_layer(x: int, k_size: int, stride: int, padding: int) -> int:
  return (x - k_size + 2 * padding) // stride + 1


def calc_feature_size(channel: int, height: int, width: int, sequential: nn.Sequential) -> int:
  """
  Calculate the number of neurons of the convolutional layer to fully connected layer.

  Parameters
  ----------
  channel : the channel of input image
  height : the height of input image
  width : the width of input image
  sequential : the convolutional layers sequential function

  Returns
  -------
  An integer, indicating the number of neurons.

  """
  # TODO: use `isinstance()` function.
  CONV, POOL = "torch.nn.modules.conv", "torch.nn.modules.pooling"
  for op in sequential:
    if op.__module__ == CONV:
      # get basic parameters
      in_channel, out_channel = op.in_channels, op.out_channels
      k_size, stride, padding = op.kernel_size, op.stride, op.padding

      # illegal channel
      if in_channel != channel: raise ValueError(f"Got {channel=}, but {in_channel=} in Conv2d.")

      # get values in height and width
      k_size_h, k_size_w = __get_value_in_height_and_width(k_size, "kernel_size")
      stride_h, stride_w = __get_value_in_height_and_width(stride, "stride")
      padding_h, padding_w = __get_value_in_height_and_width(padding, "padding")

      # calculate
      channel = out_channel
      height = _calc_value_after_layer(height, k_size_h, stride_h, padding_h)
      width = _calc_value_after_layer(width, k_size_w, stride_w, padding_w)
    elif op.__module__ == POOL:
      k_size, stride, padding = op.kernel_size, op.stride, op.padding

      # get values in height and width
      k_size_h, k_size_w = __get_value_in_height_and_width(k_size, "kernel_size")
      stride_h, stride_w = __get_value_in_height_and_width(stride, "stride")
      padding_h, padding_w = __get_value_in_height_and_width(padding, "padding")

      height = _calc_value_after_layer(height, k_size_h, stride_h, padding_h)
      width = _calc_value_after_layer(width, k_size_w, stride_w, padding_w)

  return channel * height * width
