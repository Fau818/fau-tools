import torch
from torch import nn
import torch.utils.data as tdata

from Fau_tools.data_structure import TimeManager


# ------------------------------------------------------------
# --------------- Function --- training
# ------------------------------------------------------------

def __show_progress(now: int, total: int, loss: float=None, accuracy: float=None, time_manager: TimeManager=None) -> None: ...

def calc_accuracy(model: nn.Module, test_loader: tdata.DataLoader, DEVICE: torch.device=None) -> float: ...

def torch_train(model: nn.Module, train_loader: tdata.DataLoader, test_loader: tdata.DataLoader,
                optimizer: torch.optim.Optimizer, loss_function: nn.Module, EPOCH: int=100,
				name: str=None, save_model: bool=True, DEVICE: torch.device=None) -> None: ...



# ------------------------------------------------------------
# --------------- Function --- plot
# ------------------------------------------------------------

def load_record(file_name: str) -> tuple[list, list]: ...

def draw_plot(*args: list, legend_names: list=None, x_name: str=None, y_name: str=None, percent: bool=False) -> None: ...



# ------------------------------------------------------------
# --------------- Function --- Loading model
# ------------------------------------------------------------

def load_model(model: nn.Module, file_path: str, DEVICE: torch.device=None) -> None: ...
