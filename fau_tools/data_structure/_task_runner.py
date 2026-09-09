import os
from datetime import datetime, timezone
from typing import final

import torch
import torch.utils.data as tdata
from torch import nn

import fau_tools.utils as utils

from ._model_manager import ModelManager
from ._scalar_recorder import ScalarRecorder
from ._time_manager import TimeManager


@final
class TaskRunner:
  def __init__(
    self,
    model: nn.Module,
    train_loader: tdata.DataLoader, test_loader: tdata.DataLoader,
    loss_function: nn.Module, optimizer: torch.optim.Optimizer,
    total_epoch: int,
    *,
    patience: int|None=None,
    exp_path: str|None=None, save_model: bool=True,
    clearml_task=None,
    device: str|torch.device|None=None
  ):
    """
    Classification Task Runner.

    Parameters
    ----------
    model         : the model needs to be trained
    train_loader  : train data loader
    test_loader   : test data loader
    loss_function : loss function
    optimizer     : optimizer
    total_epoch   : the total epoch of training
    patience      : the early stop threshold; set `None` to disable
    exp_path      : the experiment path for saving files
    save_model    : whether to save the trained model (needed to set `exp_path`)
    clearml_task  : a `clearml.Task` instance to set up clearml logger; set `None` to disable
    device        : the device used in pytorch; if is `None`, will be determined automatically

    """
    # ─── Set parameters to attributes ───────────────────────
    self.model = model
    self.train_loader, self.test_loader = train_loader, test_loader
    self.loss_function, self.optimizer = loss_function, optimizer

    self.total_epoch = total_epoch
    self.patience = patience

    self.exp_path = exp_path
    self.save_model = save_model

    self.clearml_task = clearml_task if self._check_clearml_task(clearml_task) else None

    self.device, self.device_name = utils.parse_device(device, return_name=True)

    # ─── Attributes ─────────────────────────────────────────
    self.train_sample_num = utils.calc_dataloader_sample_num(self.train_loader)
    self.test_sample_num  = utils.calc_dataloader_sample_num(self.test_loader)

    # ─── Module attributes ──────────────────────────────────
    self.model_manager   = ModelManager()
    self.scalar_recorder = ScalarRecorder()
    self.time_manager    = TimeManager()

    # ─── Scalars ────────────────────────────────────────────
    self.cur_epoch = 0  # NOTE: Start from zero.
    # NOTE: Consider use ModelManager to manage them.
    # The init loss value should be None, and use a function to update it.
    self.train_loss         = 0.0
    self.train_average_loss = 0.0
    # self.test_loss          = 0.0
    # self.test_average_loss  = 0.0
    self.test_accuracy      = 0.0


  @classmethod
  def _class_notify(cls, content, level):
    """Report class notice."""
    utils.notify(cls.__name__, content=content, level=level)


  def _check_clearml_task(self, clearml_task):
    """Determine whether to enable ClearML task."""
    if clearml_task is None: return False

    # Check the clearml module.
    try:
      from clearml import Task  # pyright: ignore[reportMissingImports]
    except ImportError:
      self._class_notify("Run `from clearml import Task` error.", level="error")
      return False

    # Check object type.
    if not isinstance(clearml_task, Task):
      self._class_notify("TypeError: `clearml_task` should be the `clearml.Task` type.", level="error")
      return False

    self._class_notify("ClearML task is enabled.", level="info")
    return True


  def _before_train(self):
    # Show Info
    self._class_notify(f"Training in {self.device_name} device.", level="info")
    if self.train_loader.batch_size == 1:
      self._class_notify("The batch_size is set to 1; if the net uses BN will raise an error.", level="warn")

    self.model.to(self.device)
    self.model.train()
    self.time_manager.tick_time()


  def _before_epoch(self, epoch):
    self.cur_epoch  = epoch
    self.train_loss = 0.0


  def _before_iteration(self): ...


  def _train_step(self, features: torch.Tensor, targets: torch.Tensor):
    # Forward
    features, targets = features.to(self.device), targets.to(self.device)
    outputs: torch.Tensor = self.model(features)
    loss: torch.Tensor    = self.loss_function(outputs, targets)
    # `loss` is averaged over the batch, so weight it by the batch size to keep
    # `train_loss` a sum over samples; the last batch may hold fewer of them.
    self.train_loss += loss.item() * features.size(0)

    # Backward
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()


  def _after_iteration(self): ...


  def _after_epoch(self) -> bool:
    self.time_manager.tick_time()

    self.train_average_loss = self.train_loss / self.train_sample_num
    self.test_accuracy      = self.calc_accuracy(self.test_loader)
    self._show_progress_bar()

    self._update_model()
    self._record_scalars()
    self._report_scalars_to_clearml()
    return self._stop_train()  # early stop


  def _after_train(self):
    # A run without any finished epoch (e.g. `total_epoch` is 0) leaves the manager empty.
    if self.model_manager.epoch is None:
      self._class_notify("No epoch was finished; there is no best model to report or save.", level="warn")
    else:
      self.model_manager.report(self.cur_epoch)
      if self.exp_path is not None: self._save_files()

    # The task has to be closed even when nothing was saved.
    if self.clearml_task is not None: self.clearml_task.close()


  def _save_files(self):
    """Save the experiment files into a folder named after `exp_path`, then upload them to ClearML."""
    folder_path   = f"{self.exp_path}_{self.model_manager.get_postfix()}"
    folder_path   = utils.create_folder(folder_path)
    if folder_path is None: return

    model_path    = f"{folder_path}/best.pth"
    scalar_path   = f"{folder_path}/scalars.csv"
    exp_info_path = f"{folder_path}/exp_info.txt"

    self._save_experiment(exp_info_path)
    self.scalar_recorder.save(scalar_path)
    if self.save_model: self.model_manager.save(model_path)
    if self.clearml_task is not None:
      self.clearml_task.upload_artifact("experiment information", exp_info_path)
      self.clearml_task.upload_artifact("scalars variation", scalar_path)


  def calc_accuracy(self, data_loader: tdata.DataLoader) -> float:
    total_positive_num = 0

    # Calculate positive number
    was_training = self.model.training  # this method is public: leave the mode as the caller had it
    self.model.eval()
    with torch.no_grad():
      for features, targets in data_loader:
        features, targets         = features.to(self.device), targets.to(self.device)
        outputs: torch.Tensor     = self.model(features)
        predictions: torch.Tensor = outputs.argmax(1)
        # loss: torch.Tensor = self.loss_function(outputs, targets)
        total_positive_num += predictions.eq(targets).sum().item()
    if was_training: self.model.train()

    accuracy = total_positive_num / utils.calc_dataloader_sample_num(data_loader)
    return round(accuracy, 6)


  def _show_progress_bar(self):
    # Bar settings
    finish_char, unfinish_char, char_length = '█', ' ', 30

    percent = (self.cur_epoch + 1) / self.total_epoch
    finish_char_num   = int(percent * char_length)
    unfinish_char_num = char_length - finish_char_num

    progress_section = utils.cprint(
      f"|{finish_char * finish_char_num}{unfinish_char * unfinish_char_num}| {percent:6.2%}",
      color="magenta", show=False
    )

    average_time, elapsed_time = self.time_manager.get_average_time(), self.time_manager.get_elapsed_time()
    total_time = self.total_epoch * average_time
    elapsed_time = utils.time_to_human(elapsed_time)
    total_time   = utils.time_to_human(total_time)

    time_section     = utils.cprint(f"[{elapsed_time}<{total_time}]", color="cyan", show=False)
    loss_section     = utils.cprint(f"loss: {self.train_average_loss:.6f}", color="red", show=False)
    accuracy_section = utils.cprint(f"accuracy: {self.test_accuracy:.2%}", color="green", show=False)

    progress_bar = f"{progress_section}  {time_section}  {loss_section}  {accuracy_section}"
    print(progress_bar)


  def _update_model(self):
    self.model_manager.update(self.model, self.train_average_loss, self.test_accuracy, self.cur_epoch)


  def _record_scalars(self):
    self.scalar_recorder.update(self.train_average_loss, self.test_accuracy)


  def _report_scalars_to_clearml(self):
    if self.clearml_task is None: return
    clearml_logger = self.clearml_task.get_logger()
    clearml_logger.report_scalar("loss", "train/loss", self.train_average_loss, self.cur_epoch)
    clearml_logger.report_scalar("metric", "test/accuracy", self.test_accuracy, self.cur_epoch)
    # A custom optimizer is not required to keep an "lr" entry in its param groups.
    param_groups = getattr(self.optimizer, "param_groups", ())
    if param_groups and "lr" in param_groups[0]:
      clearml_logger.report_scalar("learning_rate", "lr", param_groups[0]["lr"], self.cur_epoch)


  def _stop_train(self):
    # There is nothing to early-stop on before the first model is recorded.
    if self.patience is None or self.model_manager.epoch is None: return False

    gap = self.cur_epoch - self.model_manager.epoch
    return gap >= self.patience


  def _save_experiment(self, file_path: str):
    file_path = utils.ensure_file_postfix(file_path, ".txt")
    with open(file_path, "w") as file:
      # Save experiment datetime
      file.write(f"experiment end time: {datetime.now(timezone.utc).astimezone()}\n\n")
      # Save loss function and optimizer
      file.write(f"optimizer:\n{self.optimizer}\n")
      file.write(f"{'-' * 20}\n")
      file.write(f"loss function:\n{self.loss_function}\n")
      file.write(f"{'-' * 20}\n")
      # Save total epoch and batch size
      file.write(f"total_epoch: {self.total_epoch}\n")
      file.write(f"trained_epoch: {self.cur_epoch + 1}\n")  # smaller than total_epoch when early stopped
      file.write(f"batch size: {self.train_loader.batch_size}\n")
      file.write(f"{'-' * 20}\n")
      # Save data number
      file.write(f"train_data_number: {self.train_sample_num}\n")
      file.write(f"test_data_number : {self.test_sample_num}\n")
      # save best information
      file.write(f"{'-' * 20}\n")
      file.write(f"The best model in the {self.model_manager.get_best_epoch()} epoch.\n")
      # save time
      file.write(f"{'-' * 20}\n")
      cost_time = utils.time_to_human(self.time_manager.get_elapsed_time())
      file.write(f"Training platform: {self.device_name}\n")
      file.write(f"Training cost: {cost_time}\n")

    if os.path.exists(file_path):
      self._class_notify(f"Save experiment information file to {file_path} successfully!", level="success")
    else:
      self._class_notify("Save experiment information file error.", level="error")


  @utils.calc_time
  def train(self):
    self._before_train()

    for epoch in range(self.total_epoch):
      self._before_epoch(epoch)  # calc data load time

      for features, targets in self.train_loader:
        self._before_iteration()  # calc train cost time
        self._train_step(features, targets)
        self._after_iteration()

      stop_train_flag = self._after_epoch()
      if stop_train_flag is True:
        self._class_notify(f"Early stop: The model has gone through {self.patience} epochs without being optimized.", level="info")
        break

    self._after_train()
