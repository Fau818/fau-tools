import functools
import os
import time
import typing
from collections.abc import Sized

import torch
import torch.utils.data as tdata
from torch import nn

from ._color_print import cprint

__all__ = [
  "calc_dataloader_sample_num",
  "calc_feature_size",
  "calc_time",
  "create_folder",
  "ensure_file_postfix",
  "exit_with_error",
  "time_to_human",
]


# `calc_time` hands back the signature it was given rather than a bare `(*args, **kwargs)`.
_P = typing.ParamSpec("_P")
_R = typing.TypeVar("_R")


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
  if os.path.exists(path):
    post_num = 1
    while os.path.exists(f"{path}_{post_num}"): post_num += 1
    path = f"{path}_{post_num}"
  os.makedirs(path)

  if not os.path.isdir(path):
    cprint(f"Error: Create experiment folder in {path} failed.", color="red")
    return None

  return path


def calc_dataloader_sample_num(data_loader: tdata.DataLoader) -> int:
  """Calculate the number of samples the loader actually yields."""
  # With `drop_last`, the tail that cannot fill a whole batch never reaches the model.
  if data_loader.drop_last:
    assert data_loader.batch_size is not None  # a DataLoader rejects `drop_last` without automatic batching
    return len(data_loader) * data_loader.batch_size

  dataset = data_loader.dataset
  assert isinstance(dataset, Sized), "an iterable-style dataset has no sample count"
  return len(dataset)


def calc_time(function: typing.Callable[_P, _R]) -> typing.Callable[_P, _R]:
  """Display the running time of the decorated function."""
  @functools.wraps(function)
  def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
    print('-' * 15, "BEGIN", function.__name__, '-' * 15)
    BEGIN = time.time()
    res = function(*args, **kwargs)
    END = time.time()
    COST_TIME = time_to_human(END - BEGIN)
    print(f"{function.__name__} cost: {COST_TIME}")
    print('-' * 15, " END ", function.__name__, '-' * 15)
    return res

  return wrapper


def time_to_human(time: float) -> str:
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


# ════════════════════════════════════════════════════════════
# ═══════════════ Auto Calculate Feature Size ════════════════
# ════════════════════════════════════════════════════════════

def calc_feature_size(channel: int, height: int, width: int, sequential: nn.Sequential) -> int:
  """
  Calculate the number of neurons of the convolutional layer to fully connected layer.

  Runs one forward pass over a zero tensor; `sequential` is put in eval mode for it and
  restored afterwards.

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
  # Every layer already knows its own output shape. Re-deriving it per layer type meant
  # silently skipping whatever the code did not recognize, such as `ConvTranspose2d`.
  parameter = next(sequential.parameters(), None)
  dummy = torch.zeros(
    1, channel, height, width,
    device=parameter.device if parameter is not None else None,
    dtype=parameter.dtype if parameter is not None else None,
  )

  # In train mode the pass would feed the dummy into BatchNorm's running statistics.
  was_training = sequential.training
  sequential.eval()
  try:
    with torch.no_grad(): return sequential(dummy).numel()
  finally:
    if was_training: sequential.train()
