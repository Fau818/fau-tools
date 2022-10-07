import time

import pandas as pd
import torch
import matplotlib.pyplot as plt

from Fau_tools import utility


# ------------------------------------------------------------
# --------------- Hyper parameters
# ------------------------------------------------------------

SAVE_NOTICE = True  # control the notice that file saved successfully.



# ------------------------------------------------------------
# --------------- Data Structure
# ------------------------------------------------------------

class ModelManager:
	"""For saving the optimal model."""

	def __init__(self):
		self.loss, self.accuracy = None, None
		self.model = None

	def update(self, model, loss, accuracy):
		"""
		Args:
			model (): current model
			loss (): current loss value
			accuracy (): current accuracy rate

		Returns: None
		"""
		if self.accuracy is None:
			self.loss, self.accuracy = loss, accuracy
			self.model = model
		elif accuracy > self.accuracy and accuracy - self.accuracy <= 5E-3:
			if loss - self.loss <= 1E-1:  # slightly increase acc and loss is ok
				self.loss, self.accuracy = loss, accuracy
				self.model = model
		elif self.accuracy < accuracy:
			self.loss, self.accuracy = loss, accuracy
			self.model = model

	def save(self, file_name, only_param=True):
		"""
		Args:
			file_name (): the name of the model to save.
			only_param (): whether only to save the parameters of network.

		Returns: None
		"""

		file_name = f"{file_name}.pth"
		if only_param: torch.save(self.model.state_dict(), rf"{file_name}")
		else: torch.save(self.model, rf"{file_name}")
		if SAVE_NOTICE: print(rf"{__class__.__name__}: save a model named {file_name} successfully!")

	@staticmethod
	def load(model, file_path, DEVICE=None):
		"""
		load the trained model that saved only parameters.

		A new feature add in version 1.0.0  [test]

		Args:
			model (): load to which one
			file_path (): the trained model path
			DEVICE (): cpu or cuda; if it's None, it will judge for itself.

		Returns: None
		"""

		if DEVICE is None: DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		model.load_state_dict(torch.load(file_path))
		model.to(DEVICE)

	def get_postfix(self): return f"{round(self.accuracy * 10000)}"  # 87.65%  ->  8765





class TrainRecorder:
	"""For saving the process of training."""
	def __init__(self):
		self.loss_list, self.accuracy_list = list(), list()

	def update(self, loss_value, accuracy):
		"""
		Args:
			loss_value (): the loss value in this epoch
			accuracy (): the accuracy rate in this epoch

		Returns: None
		"""
		self.loss_list.append(loss_value)
		self.accuracy_list.append(accuracy)

	def save(self, file_name):
		"""
		Args:
			file_name (): the name of the process file

		csv_file: there are two columns named loss and accuracy.

		Returns: None
		"""
		file_name = rf"{file_name}.csv"
		with open(rf"{file_name}", "w") as file:
			col_list = ", ".join(("loss", "accuracy")) + "\n"
			file.write(col_list)
			for loss, accuracy in zip(self.loss_list, self.accuracy_list):
				line = f"{loss:.4f}, {accuracy:.4f}\n"
				file.write(line)

		if SAVE_NOTICE: print(rf"{__class__.__name__}: save a record file named {file_name} successfully!")



class TimeManager:
	def __init__(self):
		self.time_list = [time.time()]
		self.elapsed_time = 0

	def time_tick(self):
		cur_time = time.time()
		self.elapsed_time += cur_time - self.time_list[-1]
		self.time_list.append(cur_time)

	def get_average_time(self): return self.elapsed_time / (len(self.time_list) - 1)  # interval: len - 1

	def get_elapsed_time(self): return self.elapsed_time















# ------------------------------------------------------------
# --------------- Function --- training
# ------------------------------------------------------------

def _show_progress(now, total, loss=None, accuracy=None, time_manager=None):
	"""
	A function that displays a progress bar.

	Args:
		now (): the current epoch (start from zero)
		total (): the total epoch (EPOCH)
		loss (): current loss value; if it's None, it will not be display.
		accuracy (): current accuracy; if it's None, it will not be display.
		time_manager (): for showing the training time process

	Returns: None
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

		show += f" [{elapsed_time}<{total_time}]"

	if loss: show += f"  loss: {loss:.4f}"
	if accuracy: show += f"  accuracy: {accuracy:.2%}"

	print(show)



def calc_accuracy(model, test_loader, DEVICE=None):
	"""
	A function for calculating the accuracy rate in the test dataset.

	Args:
		model (): the traning model
		test_loader (): test loader
		DEVICE (): cpu or cuda; if it's None, it will judge for itself.

	Returns: the accuracy rate in the test dataset. (Round to 6 decimal places)
	"""
	if DEVICE is None: DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model.eval()  # no dropout...

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



@utility.calc_time
def torch_train(model, train_loader, test_loader, optimizer, loss_function, EPOCH=100, name=None, save_model=True, DEVICE=None):
	"""
	A function for training the best model.

	Args:
		model (): the model to be trained
		train_loader (): train dateset loader
		test_loader (): test dateset loader
		optimizer (): optimizer function
		loss_function (): loss function
		EPOCH (): total EPOCH
		name (): if the training information need to be saved, please pass the file name without postfix.
		save_model (): whether the trained model need to be saved; if needed please ensure the name parameter is not None.
		DEVICE (): cpu or cuda; if it's None, it will judge by itself.

	Returns: None

	some files will be generated.
	the trained model file named f"{name}.pth".
	the data of training process named f"{name}.csv".
	the parameters of optimizer and loss function named f"{name}.txt".

	"""

	# Acquire device information
	if DEVICE is None:
		DEVICE_NAME = None
		try:
			DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
			DEVICE_NAME = "CPU" if DEVICE == "cpu" else torch.cuda.get_device_name(0)
		except AssertionError:
			print("No cuda detected.")
			DEVICE, DEVICE_NAME = "cpu", "CPU"
		except Exception:
			print("Unknown error.")
			DEVICE, DEVICE_NAME = "cpu", "CPU"
		finally:
			print(f"{'='*10} Training in {DEVICE_NAME} {'='*10}")

	if train_loader.batch_size == 1:
		print("Warning: you shouldn't set the batch_size to 1. since if the NN uses BN, it will arise an error.")

	# for saving training data
	model_manager = ModelManager()
	train_recorder = TrainRecorder()

	# begin training
	model = model.to(DEVICE); model.train()  # initialization

	# for showing training process
	time_manager = TimeManager()

	for epoch in range(EPOCH):
		for step, (train_x, train_y) in enumerate(train_loader):
			train_x, train_y = train_x.to(DEVICE), train_y.to(DEVICE)
			output: torch.Tensor = model(train_x)
			loss: torch.Tensor = loss_function(output, train_y)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		# end of epoch
		# noinspection PyUnboundLocalVariable
		loss_value, accuracy = loss.item(), calc_accuracy(model, test_loader, DEVICE)  # get loss and acc
		time_manager.time_tick()  # tick current time
		_show_progress(epoch, EPOCH, loss_value, accuracy, time_manager)

		# update and record
		model_manager.update(model, loss_value, accuracy)
		train_recorder.update(loss_value, accuracy)


	if name is None: return  # no save

	# save model and process
	SAVE_NAME = f"{name}_{model_manager.get_postfix()}"
	if save_model: model_manager.save(SAVE_NAME)
	train_recorder.save(SAVE_NAME)

	# save the parameters
	parameters_filename = f"{SAVE_NAME}.txt"
	with open(parameters_filename, "w") as file:
		file.write(f"optimizer: \n{str(optimizer)}\n")
		file.write(f"{'-' * 20}\n")
		file.write(f"loss function: \n{str(loss_function)}\n")
		file.write(f"{'-' * 20}\n")
		file.write(f"batch size: {train_loader.batch_size}\n")
		file.write(f"epoch: {EPOCH}\n")
	if SAVE_NOTICE: print(f"{torch_train.__name__}: save a parameter file named {parameters_filename} successfully!")








# ------------------------------------------------------------
# --------------- Function --- plot
# ------------------------------------------------------------

def load_record(file_name):
	if len(file_name) < 4: raise ValueError("The file name is too short! (Missing postfix)")

	if file_name[-4:] == '.csv':
		csv = pd.read_csv(file_name, skipinitialspace=True)
		loss_list = csv["loss"].tolist()
		accuracy_list = csv["accuracy"].tolist()
		return loss_list, accuracy_list
	else: raise ValueError("The file name postfix is illegal.")



def draw_plot(*args, legend_names=None, x_name=None, y_name=None, percent=False):
	"""
	This function could show a comparison of multiple models on a single plot.
	For example, you can draw the accuracy of multiple models in a plot.

	Notes: Please manually use 'plt.show()'.

	Args:
		*args (): the list of values. values: loss_values, accuracy rates ...
		legend_names (): if the legend is required, please pass a list of names in order of the args.
		x_name (): set the name for the x-axis.
		y_name (): set the name for the y-axis.
		percent (): display the values of the y-axis as a percentage.

	Returns: None
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
		y_ticks = plt.yticks()[0]  # get y_ticks
		y_ticks_percent = [f"{op:.2%}" for op in y_ticks]  # convert to percent
		plt.yticks(y_ticks, y_ticks_percent)

# plt.show()  # Note: This will lead to show the figure one by one.
