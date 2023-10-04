import torch
from torch import nn
import torch.utils.data as tdata
from clearml import Task

from fau_tools.data_structure import ModelManager, TimeManager, TrainRecorder


# ------------------------------------------------------------
# --------------- Function --- training
# ------------------------------------------------------------
def __show_progress(now: int, total: int, loss: float=None, accuracy: float=None, time_manager: TimeManager=None) -> None: ...
def __stop_training(epoch: int, model_manager: ModelManager, threshold: int) -> bool: ...

def calc_accuracy(model: nn.Module, test_loader: tdata.DataLoader, device: torch.device=None) -> float: ...

def torch_train(
  model: nn.Module, train_loader: tdata.DataLoader, test_loader: tdata.DataLoader,
  optimizer: torch.optim.Optimizer, loss_function: nn.Module,
  *,
  total_epoch: int=100, early_stop: int=None,
  name: str=None, save_model: bool=True,
  clearml_task: Task=None,
  device: torch.device=None
) -> None: ...


# ------------------------------------------------------------
# --------------- Function --- plot
# ------------------------------------------------------------
def load_record(file_path: str) -> tuple[list, list]: ...
def draw_plot(*args: list, legend_names: list=None, x_name: str=None, y_name: str=None, percent: bool=False) -> None: ...


# ------------------------------------------------------------
# --------------- Function --- Loading model
# ------------------------------------------------------------
def load_model(model: nn.Module, file_path: str, device: torch.device=None) -> None: ...
