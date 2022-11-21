import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib import ticker

from Fau_tools import utility
from Fau_tools.data_structure import ModelManager, TimeManager, TrainRecorder
from Fau_tools.utility import cprint



# ------------------------------------------------------------
# --------------- Function --- training
# ------------------------------------------------------------

def __show_progress(now, total, loss=None, accuracy=None, time_manager=None):
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

		show += cprint(f"  [{elapsed_time}<{total_time}]", "cyan", ret=True)

	if loss: show += cprint(f"  loss: {loss:.6f}", "red", ret=True)
	if accuracy: show += cprint(f"  accuracy: {accuracy:.2%}", "green", ret=True)

	print(show)



def calc_accuracy(model, test_loader, DEVICE=None):
	"""
	A function for calculating the accuracy rate in the test dataset.

	Args:
		model (): the training model
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
		__show_progress(epoch, EPOCH, loss_value, accuracy, time_manager)

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
		try:  # for saving the number of train and test data
			train_data_num = len(train_loader.batch_sampler.sampler.data_source.labels)
			test_data_num = len(test_loader.batch_sampler.sampler.data_source.labels)
		except AttributeError: cprint("Saving the number of train and test data error.", "red")
		else:
			file.write(f"{'-' * 20}\n")
			file.write(f"train_data_number: {train_data_num}\n")
			file.write(f"test_data_number: {test_data_num}\n")


	cprint(f"{torch_train.__name__}: save a parameter file named {parameters_filename} successfully!", "green")





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
		plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))

	# plt.show()  # Note: This will lead to show the figure one by one.




# ------------------------------------------------------------
# --------------- Function --- Loading model
# ------------------------------------------------------------

def load_model(model, file_path, DEVICE=None):
	""" See ModelManager.load function in data_structure module. """
	ModelManager.load(model, file_path, DEVICE)
