import os

import torch
import torch.utils.data as tdata
from torch import nn

import fau_tools.utils as utils

__all__ = ["calc_evaluation_indicators", "draw_plot", "load_record"]


def calc_evaluation_indicators(
  model: nn.Module, data_loader: tdata.DataLoader, device: str|torch.device|None=None,
) -> tuple[float, float, float, float]:
  """
  Calculate the accuracy, precision, recall and f1 score of a model over a dataset.

  The three macro scores follow scikit-learn's convention: a class whose denominator is zero
  scores 0, and the average runs over the classes that appear in the targets or the predictions.
  `model` is moved to `device` and is left in the mode it arrived in.

  Parameters
  ----------
  model : the model to evaluate
  data_loader : the data loader holding the samples to evaluate on
  device : the device used in pytorch; if is `None`, will be determined automatically

  Returns
  -------
  The (accuracy, precision, recall, f1) indicators, each rounded to 6 decimal places.

  """
  device = utils.parse_device(device)
  model.to(device)
  was_training = model.training
  model.eval()

  # How many classes there are is whatever the model scores, known only after a forward pass.
  class_num, collected = 0, []
  with torch.no_grad():
    for features, targets in data_loader:
      features, targets = features.to(device), targets.to(device)
      outputs: torch.Tensor = model(features)
      class_num = max(class_num, outputs.size(1))
      collected.append((targets, outputs.argmax(1)))

  if was_training: model.train()
  if not collected: raise ValueError("`data_loader` yielded no batch to evaluate.")

  # Accumulating one matrix over the whole set, rather than averaging per-batch scores, is what
  # makes a macro average mean anything: the latter drifts with the batch size.
  targets = torch.cat([batch_targets for batch_targets, _ in collected])
  predictions = torch.cat([batch_predictions for _, batch_predictions in collected])
  confusion = torch.zeros(class_num, class_num, dtype=torch.long, device=device)
  confusion.index_put_((targets, predictions), torch.ones_like(targets), accumulate=True)

  # A macro average runs over the classes scikit-learn would use: those seen in either side.
  seen = torch.zeros(class_num, dtype=torch.bool, device=device)
  seen[targets.unique()] = True
  seen[predictions.unique()] = True

  # float64 keeps the numbers identical to scikit-learn's; the matrix is tiny either way.
  true_positive = confusion.diag().double()
  predicted, actual = confusion.sum(0).double(), confusion.sum(1).double()
  zero = torch.zeros_like(true_positive)

  precision = torch.where(predicted > 0, true_positive / predicted.clamp(min=1), zero)
  recall    = torch.where(actual > 0, true_positive / actual.clamp(min=1), zero)
  total     = precision + recall
  f1        = torch.where(total > 0, 2 * precision * recall / total.clamp(min=1e-12), zero)

  accuracy = true_positive.sum().item() / confusion.sum().item()
  return (round(accuracy, 6), round(precision[seen].mean().item(), 6),
          round(recall[seen].mean().item(), 6), round(f1[seen].mean().item(), 6))


def load_record(file_path: str) -> tuple[list[float], list[float]]:
  """
  Load the training record.

  Parameters
  ----------
  file_path : the record file path

  Returns
  -------
  (loss_list, accuracy_list)

  Raises
  ------
  ValueError : File path is illegal.

  """
  if os.path.splitext(file_path)[1].lower() != ".csv": raise ValueError(f"Expected a `.csv` file, but got `{file_path}`.")

  import pandas as pd
  csv = pd.read_csv(file_path, skipinitialspace=True)
  loss_list     = csv["loss"].tolist()
  accuracy_list = csv["accuracy"].tolist()
  return loss_list, accuracy_list


def draw_plot(*args, legend_names: list[str]|None=None, x_name: str|None=None, y_name: str|None=None, percent: bool=False):
  """
  Display a comparison of multiple models on a single plot.

  For example, you can draw the accuracy of multiple models in a plot.
  Notes: Please manually use 'plt.show()'.

  Parameters
  ----------
  args         : the list of `values`; `values`: loss values or accuracy rates ...
  legend_names : if the legend is required, please pass a list of names in order of the args
  x_name       : set the name for the x-axis
  y_name       : set the name for the y-axis
  percent      : display the values of the y-axis as a percentage

  """
  import matplotlib.pyplot as plt
  from matplotlib import ticker

  if legend_names is not None and len(args) != len(legend_names):
    raise ValueError("The length of legend is not equal to the number of args.")

  plt.figure()

  # Draw plot
  plt_list = []
  for cur in args:
    cur_plt, = plt.plot(range(1, len(cur) + 1), cur)  # unpack
    plt_list.append(cur_plt)

  # Add effects
  if legend_names is not None: plt.legend(handles=plt_list, labels=legend_names)
  if x_name is not None: plt.xlabel(x_name)
  if y_name is not None: plt.ylabel(y_name)
  if percent:
    plt.ylim(0, 1)
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))

  # plt.show()  # Note: This will lead to show the figure one by one.
