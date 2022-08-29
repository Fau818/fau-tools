import torch
from torch import nn
import torch.utils.data as tdata


# ------------------------------------------------------------
# --------------- Data Structure
# ------------------------------------------------------------

class ModelManager:
	def __init__(self):
		self.loss: float = ...
		self.accuracy: float = ...
		self.model: nn.Module = ...

	def update(self, model: nn.Module, loss: float, accuracy: float) -> None: ...

	def save(self, file_name: str, only_param: bool=True) -> None: ...



class TrainRecorder:
	def __init__(self):
		self.loss_list: list = ...
		self.accuracy_list: list = ...

	def update(self, loss_value: float, accuracy: float) ->None: ...

	def save(self, file_name: str) -> None: ...







# ------------------------------------------------------------
# --------------- Function --- training
# ------------------------------------------------------------

def show_progress(now: int, total: int, loss: float=None, accuracy: float=None) -> None: ...

def calc_accuracy(model: nn.Module, test_loader: tdata.DataLoader, DEVICE: torch.device=None) -> float: ...

def torch_train(model: nn.Module, train_loader: tdata.DataLoader, test_loader: tdata.DataLoader,
                optimizer: torch.optim.Optimizer, loss_function: nn.Module, EPOCH: int=100,
                DEVICE: torch.device=None, SAVE_NAME: str=None) -> None: ...





# ------------------------------------------------------------
# --------------- Function --- plot
# ------------------------------------------------------------

def load_record(file_name: str) -> (list, list): ...

def draw_plot(*args: list, legend_names: list=None, x_name: str=None, y_name: str=None, percent: bool=False) -> None: ...
