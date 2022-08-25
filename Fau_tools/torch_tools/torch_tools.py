import pandas as pd
import torch
import matplotlib.pyplot as plt



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
			if loss - self.loss <= 5E-3:  # slightly increase acc and loss is ok
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
		postfix = f"{round(self.accuracy * 10000)}"  # 87.65%  ->  8765
		file_name = rf"{file_name}_{postfix}"
		if only_param: torch.save(self.model.state_dict(), rf"{file_name}.pth")
		else: torch.save(self.model, rf"{file_name}.pth")
	# save and load model
	# torch.save(model.state_dict(), "cnn_model.pth")
	# model = model.load_state_dict(torch.load("cnn_model.pth"))



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
		with open(rf"{file_name}.csv", "w") as file:
			col_list = ", ".join(("loss", "accuracy")) + "\n"
			file.write(col_list)
			for loss, accuracy in zip(self.loss_list, self.accuracy_list):
				line = f"{loss:.4f}, {accuracy:.4f}\n"
				file.write(line)








# ------------------------------------------------------------
# --------------- Function --- tools
# ------------------------------------------------------------

def calc_time(function):
	"""
	A decorator, used to display the function begin, end and the cost of time.
	"""
	def wrapper(*args, **kwargs):
		print('-' * 15, "BEGIN", function.__name__, '-' * 15)
		import time
		BEGIN = time.time()
		res = function(*args, **kwargs)
		END = time.time()
		print(f"{function.__name__} cost: {END - BEGIN:.6f}s")
		print('-' * 15, " END ", function.__name__, '-' * 15)
		return res

	return wrapper



def show_progress(now, total, loss=None, accuracy=None):
	"""
	A function that displays a progress bar.

	Args:
		now (): the current epoch (start from zero)
		total (): the total epoch (EPOCH)
		loss (): current loss value; if it's None, it will not be display.
		accuracy (): currenct accuracy; if it's None, it will not be display.

	Returns: None
	"""
	now += 1  # remap 0 -> 1
	FINISH, UNFINISH = '█', ' '
	N = 30  # the length
	PERCENT = now / total

	# main operation
	finish = int(PERCENT * N) * FINISH
	unfinish = (N - len(finish)) * UNFINISH
	show = f"|{finish}{unfinish}| {PERCENT:.2%}"
	if loss is not None: show += f"  loss: {loss:.4f}"
	if accuracy is not None: show += f"  accuracy: {accuracy:.2%}"

	print(show)





# ------------------------------------------------------------
# --------------- Function --- training
# ------------------------------------------------------------

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

	test_result = list()  # for calculating the average accuracy rate.
	for (test_x, test_y) in test_loader:
		test_x, test_y = test_x.to(DEVICE), test_y.to(DEVICE)
		test_output: torch.Tensor = model(test_x)
		test_prediction: torch.Tensor = test_output.argmax(1)  # get classification result set
		cur_accuracy: torch.Tensor = sum(test_prediction.eq(test_y)) / test_y.size(0)
		test_result.append(cur_accuracy.item())  # tensor -> scaler

	model.train()  # recover
	accuracy: float = sum(test_result) / len(test_result)  # get average
	return round(accuracy, 6)



@calc_time
def torch_train(model, train_loader, test_loader, optimizer, loss_function, EPOCH=100, DEVICE=None, SAVE_NAME=None):
	"""
	A function for training the best model.

	Args:
		model (): the model to be trained
		train_loader (): train dateset loader
		test_loader (): test dateset loader
		optimizer (): optimizer function
		loss_function (): loss function
		EPOCH (): total EPOCH
		DEVICE (): cpu or cuda; if it's None, it will judge for itself.
		SAVE_NAME (): if the model need to be saved, please pass the model name without postfix.

	Returns: None

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


	# for saving training data
	model_manager = ModelManager()
	train_recorder = TrainRecorder()

	# begin training
	model = model.to(DEVICE); model.train()  # initialization

	for epoch in range(EPOCH):
		for step, (train_x, train_y) in enumerate(train_loader):
			train_x, train_y = train_x.to(DEVICE), train_y.to(DEVICE)
			output: torch.Tensor = model(train_x)
			loss: torch.Tensor = loss_function(output, train_y)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		# end of epoch
		loss_value, accuracy = loss.item(), calc_accuracy(model, test_loader)  # get loss and acc
		show_progress(epoch, EPOCH, loss_value, accuracy)
		# update and record
		model_manager.update(model, loss_value, accuracy)
		train_recorder.update(loss_value, accuracy)


	if SAVE_NAME is None: return  # no save
	# save model and process
	model_manager.save(SAVE_NAME)
	train_recorder.save(f"{SAVE_NAME}_{round(model_manager.accuracy * 10000)}")



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

	plt.show()
