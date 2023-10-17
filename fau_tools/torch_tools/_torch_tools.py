def load_record(file_path: str) -> tuple[list[float], list[float]]:
  """
  Load the traning record.

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
  if len(file_path) < 4: ValueError("The file name is too short! (Missing postfix)")

  if file_path[-4:] == '.csv':
    import pandas as pd
    csv = pd.read_csv(file_path, skipinitialspace=True)
    loss_list     = csv["loss"].tolist()
    accuracy_list = csv["accuracy"].tolist()
    return loss_list, accuracy_list
  else: raise ValueError("The file name postfix is illegal.")


def draw_plot(*args, legend_names: list[str]=None, x_name: str=None, y_name: str=None, percent: bool=False):
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
  plt_list = list() if legend_names is not None else None
  for cur in args:
    cur_plt, = plt.plot(range(1, len(cur) + 1), cur)  # unpack
    if legend_names is not None: plt_list.append(cur_plt)

  # Add effects
  if legend_names is not None: plt.legend(handles=plt_list, labels=legend_names)
  if x_name is not None: plt.xlabel(x_name)
  if y_name is not None: plt.ylabel(y_name)
  if percent:
    plt.ylim(0, 1)
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))

  # plt.show()  # Note: This will lead to show the figure one by one.
