import torch
import time

from Fau_tools.utility import cprint


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
		cprint(rf"{__class__.__name__}: save a model named {file_name} successfully!", "green")

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
				line = f"{loss:.6f}, {accuracy:.4f}\n"
				file.write(line)

		cprint(rf"{__class__.__name__}: save a record file named {file_name} successfully!", "green")





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
