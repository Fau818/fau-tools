import os

import torch
import torch.nn as nn

from fau_tools import utils


class ModelManager:
  """Manage the model."""

  def __init__(self):
    self.model: nn.Module = None
    self.loss: float      = None
    self.accuracy: float  = None
    self.epoch: int       = None


  @classmethod
  def _class_notify(cls, content, notify_type):
    """Report class notice."""
    utils.notify(cls.__name__, content=content, notify_type=notify_type)


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
      self.model = model
      self.epoch = epoch


  def save(self, file_path: str, only_param: bool=True):
    """
    Save the selected(optimal) model.

    Parameters
    ----------
    file_path  : the name of the saved model
    only_param : whether only save the parameters of the model

    """
    file_path = utils.ensure_file_postfix(file_path, ".pth")
    if only_param: torch.save(self.model.state_dict(), rf"{file_path}")
    else: torch.save(self.model, rf"{file_path}")

    if os.path.exists(file_path):
      self._class_notify(f"Save best model to {file_path} successfully!", notify_type="success")
    else:
      self._class_notify(f"Save best model error.", notify_type="error")


  @staticmethod
  def load(model: nn.Module, file_path: str, device: str|torch.device=None):
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
    device = utils.device.parse_device(device)
    model.load_state_dict(torch.load(file_path, device))
    model.eval()


  def get_postfix(self): return f"{round(self.accuracy * 10000)}"  # 87.65%  ->  8765


  def report(self, training_epoch: int):
    """Report the best model."""
    self._class_notify(f"After {training_epoch + 1} training epochs, the best model at the {self.epoch + 1} epoch with {self.accuracy:.2%} accuracy.", notify_type="info")
