import os

from fau_tools import utils


class ScalarRecorder:
  """Record the process of training."""

  def __init__(self):
    self.loss_list, self.accuracy_list = list(), list()


  @classmethod
  def _class_notify(cls, content, notify_type):
    """Report class notice."""
    utils.notify(cls.__name__, content=content, notify_type=notify_type)


  def update(self, loss_value: float, accuracy: float):
    """
    Update the training recorder.

    Parameters
    ----------
    loss_value : the current loss value
    accuracy   : the current accuracy rate

    """
    self.loss_list.append(loss_value)
    self.accuracy_list.append(accuracy)


  def save(self, file_path: str):
    """
    Save the training process.

    Parameters
    ----------
    file_path : the name of the process file

    Returns
    -------
    Generate a csv_file; there are some columns recorded the values variation during training

    """
    file_path = utils.ensure_file_postfix(file_path, ".csv")
    with open(rf"{file_path}", "w") as file:
      col_list = ", ".join(("loss", "accuracy")) + "\n"
      file.write(col_list)
      for loss, accuracy in zip(self.loss_list, self.accuracy_list):
        line = f"{loss:.6f}, {accuracy:.6f}\n"
        file.write(line)

    if os.path.exists(file_path):
      file_name = os.path.basename(file_path)
      self._class_notify(f"Save training process file named {file_name} successfully!", notify_type="success")
    else:
      self._class_notify(f"Save training process file error.", notify_type="error")
