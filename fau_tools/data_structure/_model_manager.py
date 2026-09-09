import copy
import os

import torch
from torch import nn

import fau_tools.utils as utils


class ModelManager:
  """Manage the model."""

  def __init__(self):
    self.model: nn.Module|None = None
    self.loss: float|None      = None
    self.accuracy: float|None  = None
    self.epoch: int|None       = None


  @classmethod
  def _class_notify(cls, content, level):
    """Report class notice."""
    utils.notify(cls.__name__, content=content, level=level)


  def update(self, model: nn.Module, loss: float, accuracy: float, epoch: int):
    """
    Update to the best model.

    Parameters
    ----------
    model    : current model
    loss     : current loss value
    accuracy : current accuracy rate
    epoch    : current epoch

    """
    if self.accuracy is None or self.accuracy < accuracy:
      self.loss, self.accuracy = loss, accuracy
      self.model = copy.deepcopy(model)  # a reference would let the later epochs overwrite the best weights
      self.epoch = epoch


  def save(self, file_path: str, only_param: bool=True):
    """
    Save the selected(optimal) model.

    Parameters
    ----------
    file_path  : the name of the saved model
    only_param : whether only save the parameters of the model

    """
    assert self.model is not None, "no model has been recorded yet"

    file_path = utils.ensure_file_postfix(file_path, ".pth")
    if only_param: torch.save(self.model.state_dict(), file_path)
    else: torch.save(self.model, file_path)

    if os.path.exists(file_path):
      self._class_notify(f"Save best model to {file_path} successfully!", level="success")
    else:
      self._class_notify("Save best model error.", level="error")


  @staticmethod
  def load(model: nn.Module, file_path: str, device: str|torch.device|None=None):
    """
    Load the trained model that saved only parameters.

    Parameters
    ----------
    model     : the structure of the model.
    file_path : the path of the trained model.
    device    : the calculating device used in pytorch; if None, will be determined automatically

    Returns
    -------
    After this method, the model will be loaded on `device` with the evaluation mode.

    """
    device = utils.parse_device(device)
    model.load_state_dict(torch.load(file_path, map_location=device))
    model.eval()


  def get_postfix(self) -> str:
    assert self.accuracy is not None, "no model has been recorded yet"
    return f"{round(self.accuracy * 10000)}"  # 87.65% -> 8765


  def get_best_epoch(self) -> int:
    """Return the 1-based epoch that produced the best model."""
    assert self.epoch is not None, "no model has been recorded yet"
    return self.epoch + 1


  def report(self, training_epoch: int):
    """Report the best model."""
    assert self.accuracy is not None, "no model has been recorded yet"
    self._class_notify(f"After {training_epoch + 1} training epochs, the best model at the {self.get_best_epoch()} epoch with {self.accuracy:.2%} accuracy.", level="info")
