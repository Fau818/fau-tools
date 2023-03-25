import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib import ticker
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve

from Fau_tools import utility
from Fau_tools.data_structure import ModelManager, TimeManager, TrainRecorder
from Fau_tools.utility import cprint



# ------------------------------------------------------------
# --------------- Function --- training
# ------------------------------------------------------------

def __show_progress(now, total, loss=None, accuracy=None, time_manager=None):
  """
  Show the training process bar.

  Parameters
  ----------
  now : the current epoch (start from zero)
  total : the total epoch (EPOCH)
  loss : current loss value; if None, will not be displayed.
  accuracy : current accuracy; if None, will not be displayed.
  time_manager : for showing the training time process; if None, will not be displayed.

  """
  now += 1  # remap 0 -> 1
  FINISH, UNFINISH = '█', ' '
  N = 30  # the length
  PERCENT = now / total

  # for showing blocks
  finish = int(PERCENT * N) * FINISH
  unfinish = (N - len(finish)) * UNFINISH
  show = f"|{finish}{unfinish}| {PERCENT:.2%}"

  if time_manager:  # for showing time process:
    average_time, elapsed_time = time_manager.get_average_time(), time_manager.get_elapsed_time()
    total_time = total * average_time

    elapsed_time = utility.time_to_human(elapsed_time)
    total_time = utility.time_to_human(total_time)

    show += cprint(f"  [{elapsed_time}<{total_time}]", "cyan", ret=True)

  if loss: show += cprint(f"  loss: {loss:.6f}", "red", ret=True)
  if accuracy: show += cprint(f"  accuracy: {accuracy:.2%}", "green", ret=True)

  print(show)



def __stop_training(epoch, model_manager, threshold):
  """
  Judge whether satisfy early stop.

  Parameters
  ----------
  epoch : current epoch
  model_manager : the model manager
  threshold : early_stop threshold

  Returns
  -------
  Boolean value, indicating whether should stop training.

  """
  gap = epoch - model_manager.epoch
  return gap >= threshold



def calc_accuracy(model, test_loader, DEVICE=None):
  """
  Calculate the accuracy rate in the test dataset.

  Parameters
  ----------
  model : the training model
  test_loader : the test data loader
  DEVICE : cpu or cuda; if None, will be judged automatically

  Returns
  -------
  The accuracy rate in the test dataset. (Round to 6 decimal places)

  """
  if DEVICE is None: DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  model.eval()  # evaluation mode

  with torch.no_grad():
    test_result = list()  # for calculating the average accuracy rate.
    for (test_x, test_y) in test_loader:
      test_x, test_y = test_x.to(DEVICE), test_y.to(DEVICE)
      test_output: torch.Tensor = model(test_x)
      test_prediction: torch.Tensor = test_output.argmax(1)  # get classification result set
      cur_accuracy: torch.Tensor = sum(test_prediction.eq(test_y)) / test_y.size(0)
      test_result.append(cur_accuracy.item())  # tensor -> scaler
    accuracy: float = sum(test_result) / len(test_result)  # get average accuracy

  model.train()  # recover
  return round(accuracy, 6)



def calc_evaluation_indicators(model, test_loader, DEVICE=None):
  """
  Calculate the evaluation indicators in the test dataset.

  Added in version 1.4.2

  Parameters
  ----------
  model : the training model
  test_loader : the test data loader
  DEVICE : cpu or cuda; if None, will be judged automatically

  Returns
  -------
  The (accuracy, precision, recall, f1) evaluation indicators
  in the test dataset. (Round to 6 decimal places)

  """
  if DEVICE is None: DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  model.eval()  # evaluation mode

  with torch.no_grad():
    accuracy_list, precision_list = list(), list()
    recall_list, f1_list = list(), list()
    for (test_features, test_labels) in test_loader:
      test_features, test_labels = test_features.to(DEVICE), test_labels.to(DEVICE)
      test_output: torch.Tensor = model(test_features)
      test_prediction: torch.Tensor = test_output.argmax(1)  # get classification result set
      test_labels, test_prediction = test_labels.cpu(), test_prediction.cpu()

      cur_accuracy  = accuracy_score(test_labels, test_prediction)
      cur_precision = precision_score(test_labels, test_prediction, average="macro", zero_division=0)
      cur_recall    = recall_score(test_labels, test_prediction, average="macro", zero_division=0)
      cur_f1        = f1_score(test_labels, test_prediction, average="macro", zero_division=0)

      accuracy_list.append(cur_accuracy)
      precision_list.append(cur_precision)
      recall_list.append(cur_recall)
      f1_list.append(cur_f1)

    accuracy: float  = sum(accuracy_list) / len(accuracy_list)
    precision: float = sum(precision_list) / len(precision_list)
    recall: float    = sum(recall_list) / len(recall_list)
    f1: float        = sum(f1_list) / len(f1_list)

  model.train()  # recover
  return round(accuracy, 6), round(precision, 6), round(recall, 6), round(f1, 6)




@utility.calc_time
def torch_train(model, train_loader, test_loader, optimizer, loss_function, EPOCH=100, early_stop=None, name=None, save_model=True, DEVICE=None):
  """
  Train the model.

  Parameters
  ----------
  model : the model needs to be trained
  train_loader : train dataset loader
  test_loader : test dataset loader
  optimizer : optimizer function
  loss_function : loss function
  EPOCH : training epoch value
  early_stop : Whether use early stop; Pass an integer as the threshold
  name : if the training process needs to be saved, please pass the file name without postfix.
  save_model : whether the trained model needs to be saved; if needed please ensure the name parameter is not None.
  DEVICE : cpu or cuda; if None, will be judged automatically

  Returns
  -------
  Some files may be generated:
    1. the trained model file named f"{name}.pth".
    2. the values variation during the training named f"{name}.csv".
    3. the hyperparameters and time spent file named f"{name}.txt".

  """
  # Acquire device information
  if DEVICE is None:
    DEVICE_NAME = None
    try:
      DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      DEVICE_NAME = "CPU" if DEVICE == "cpu" else torch.cuda.get_device_name(0)
    except AssertionError:
      cprint("No cuda detected.", "yellow")
      DEVICE, DEVICE_NAME = "cpu", "CPU"
    except Exception:
      cprint("Unknown error in torch_tools.", "red")
      DEVICE, DEVICE_NAME = "cpu", "CPU"
    finally:
      cprint(f"{'='*10} Training in {DEVICE_NAME} {'='*10}", "green")

  if train_loader.batch_size == 1:
    cprint("Warning: you shouldn't set the batch_size to 1. since if the NN uses BN, it will arise an error.", "red")

  # for saving training data
  model_manager = ModelManager()
  train_recorder = TrainRecorder()

  # begin training
  model = model.to(DEVICE); model.train()  # initialization

  # for showing training process
  time_manager = TimeManager()

  for epoch in range(EPOCH):
    for _, (train_x, train_y) in enumerate(train_loader):
      train_x, train_y = train_x.to(DEVICE), train_y.to(DEVICE)
      output: torch.Tensor = model(train_x)
      loss: torch.Tensor = loss_function(output, train_y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    # end of epoch
    loss_value, (accuracy, precision, recall, f1) = loss.item(), calc_evaluation_indicators(model, test_loader)
    time_manager.time_tick()  # tick current time
    __show_progress(epoch, EPOCH, loss_value, accuracy, time_manager)

    # update and record
    model_manager.update(model, loss_value, accuracy, epoch)
    train_recorder.update(loss_value, accuracy, precision, recall, f1)

    # Judge early stop
    if early_stop is not None and __stop_training(epoch, model_manager, early_stop):
      cprint(f"Early stop: The model has gone through {early_stop} epochs without being optimized.", "yellow")
      break


  if name is None: return  # no save

  # save model and process
  SAVE_NAME = f"{name}_{model_manager.get_postfix()}"
  if save_model: model_manager.save(SAVE_NAME)
  train_recorder.save(SAVE_NAME)

  # save the parameters
  parameters_filename = f"{SAVE_NAME}.txt"
  with open(parameters_filename, "w") as file:
    file.write(f"optimizer:\n{str(optimizer)}\n")
    file.write(f"{'-' * 20}\n")
    file.write(f"loss function:\n{str(loss_function)}\n")
    file.write(f"{'-' * 20}\n")
    file.write(f"batch size: {train_loader.batch_size}\n")
    file.write(f"epoch: {EPOCH}\n")
    try:  # for saving the number of train and test data
      train_data_num = len(train_loader.batch_sampler.sampler.data_source.labels)
      test_data_num = len(test_loader.batch_sampler.sampler.data_source.labels)
    except AttributeError: cprint("Saving the number of train and test data error.", "red")
    else:
      file.write(f"{'-' * 20}\n")
      file.write(f"train_data_number: {train_data_num}\n")
      file.write(f"test_data_number: {test_data_num}\n")

    # save best info
    file.write(f"{'-' * 20}\n")
    file.write(f"The best model in the {model_manager.epoch} epoch.\n")

    # save time
    file.write(f"{'-' * 20}\n")
    cost_time = time_manager.get_elapsed_time()
    cost_time = utility.time_to_human(cost_time)
    file.write(f"Training cost: {cost_time}\n")


  cprint(f"{torch_train.__name__}: save a parameter file named {parameters_filename} successfully!", "green")





# ------------------------------------------------------------
# --------------- Function --- plot
# ------------------------------------------------------------

def load_record(file_path):
  """
  Load the traning record.

  Parameters
  ----------
  file_name : the record file path

  Returns
  -------
  (loss_list, accuracy_list)

  Raises
  ------
  ValueError : File path is illegal.

  """
  if len(file_path) < 4: raise ValueError("The file name is too short! (Missing postfix)")

  if file_path[-4:] == '.csv':
    csv = pd.read_csv(file_path, skipinitialspace=True)
    loss_list = csv["loss"].tolist()
    accuracy_list = csv["accuracy"].tolist()
    return loss_list, accuracy_list
  else: raise ValueError("The file name postfix is illegal.")



def draw_plot(*args, legend_names=None, x_name=None, y_name=None, percent=False):
  """
  Show a comparison of multiple models on a single plot.

  For example, you can draw the accuracy of multiple models in a plot.
  Notes: Please manually use 'plt.show()'.

  Parameters
  ----------
  *args : the list of `values`; `values`: loss_values, accuracy rates ...
  legend_names : if the legend is required, please pass a list of names in order of the args.
  x_name : set the name for the x-axis.
  y_name : set the name for the y-axis.
  percent : display the values of the y-axis as a percentage.

  """
  if legend_names is not None and len(args) != len(legend_names):
    raise ValueError("the length of legend is not equal to the number of args.")

  plt.figure()
  # draw plot
  plt_list = list() if legend_names is not None else None
  for cur in args:
    cur_plt, = plt.plot(range(1, len(cur) + 1), cur)  # unpack
    if legend_names is not None: plt_list.append(cur_plt)

  # add effects
  if legend_names is not None: plt.legend(handles=plt_list, labels=legend_names)
  if x_name is not None: plt.xlabel(x_name)
  if y_name is not None: plt.ylabel(y_name)
  if percent:
    plt.ylim(0, 1)
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))

  # plt.show()  # Note: This will lead to show the figure one by one.




# ------------------------------------------------------------
# --------------- Function --- Loading model
# ------------------------------------------------------------

def load_model(model, file_path, DEVICE=None):
  """See ModelManager.load function in data_structure module."""
  ModelManager.load(model, file_path, DEVICE)
