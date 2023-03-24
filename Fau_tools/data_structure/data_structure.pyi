import torch
from torch import nn


class ModelManager:
  def __init__(self):
    self.loss: float
    self.accuracy: float
    self.model: nn.Module
    self.epoch: int

  def update(self, model: nn.Module, loss: float, accuracy: float, epoch: int) -> None: ...

  def save(self, file_name: str, only_param: bool=True) -> None: ...

  @staticmethod
  def load(model: nn.Module, file_path: str, DEVICE: torch.device=None) -> None: ...

  def get_postfix(self) -> str: ...



class TrainRecorder:
  def __init__(self):
    self.loss_list: list
    self.accuracy_list: list
    self.precision_list: list
    self.recall_list: list
    self.f1_list: list

  def update(self, loss_value: float, accuracy: float, precision: float, recall: float, f1: float) -> None: ...

  def save(self, file_name: str) -> None: ...



class TimeManager:
  def __init__(self):
    self.time_list: list
    self.elapsed_time: float = 0

  def time_tick(self) -> None: ...

  def get_average_time(self) -> float: ...

  def get_elapsed_time(self) -> float: ...
