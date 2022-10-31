import numpy as np


def time_to_human(time):
	"""
	the function is to convert time in seconds to the human-friendly time display.

	Args:
		time (): the time

	Returns:
		a string in the format HH:mm:ss
		but if the time is more than one day, will return "MTOD"

	"""
	time = int(time)
	second = time % 60; time //= 60
	minute = time % 60; time //= 60
	hour = time

	if hour >= 24: return "MTOD"  # more than one day

	if hour > 0: return f"{hour:02d}:{minute:02d}:{second:02d}"
	if minute > 0: return f"{minute:02d}:{second:02d}"
	if second > 0: return f"{second:02d}s"

	return "minor"



def calc_time(function):
	def wrapper(*args, **kwargs):
		print('-' * 15, "BEGIN", function.__name__, '-' * 15)
		import time
		BEGIN = time.time()
		res = function(*args, **kwargs)
		END = time.time()
		COST_TIME = time_to_human(END - BEGIN)
		print(f"{function.__name__} cost: {COST_TIME}")
		print('-' * 15, " END ", function.__name__, '-' * 15)
		return res

	return wrapper






class ActivationFunction:
	@staticmethod
	def sigmoid(x): return 1 / (1 + np.exp(-x))

	@staticmethod
	def tanh(x): return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

	@staticmethod
	def relu(x): return np.maximum(x, 0)














# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# def calc_grandient(f, x):
# 	""" others method
# 	def function(x):  # 定义函数
# 		return -x[0] ** 2 - x[1] ** 2 + 2 * x[1] + 1  # f(z) = -x^2 - y^2 + 2y + 1
# 	:param f:
# 	:param x:
# 	:return:
# 	"""
# 	h = 1e-4  # 定义一个微小量，不能太小，太小计算机没法正确表示
# 	grad = np.zeros_like(x)  # 生成和x形状相同的数组
# 	for idx in range(x.size):  # 计算所有偏导
# 		tmp_val = x[idx]
# 		x[idx] = tmp_val + h  # 要计算的那个自变量加h，其余不变
# 		fxh1 = f(x)  # 计算f(x+h)

# 		x[idx] = tmp_val - h  # 计算f(x-h)
# 		fxh2 = f(x)

# 		grad[idx] = (fxh1 - fxh2) / (2 * h)  # 计算偏导
# 		x[idx] = tmp_val
# 	return grad


# def calc_mse(list_y, list_yh):
# 	total = 0
# 	for x, y in zip(list_y, list_yh):
# 		total += (x - y) ** 2
# 	return total / len(list_y)


# def calc_rmse(list_y, list_yh):
# 	return math.sqrt(calc_mse(list_y, list_yh))


# def calc_nrmse(list_y, list_yh):
# 	return calc_rmse(list_y, list_yh) / (sum(list_y) / len(list_y))


# def calc_mae(list_y, list_yh):
# 	total = 0
# 	for x, y in zip(list_y, list_yh):
# 		total += abs(x - y)
# 	return total / len(list_y)


# def calc_lsm(list_x, list_y):
# 	n = len(list_x)
# 	X, Y = np.mat(list_x).reshape(n, 1), np.mat(list_y).reshape(n, 1)

# 	# (X^T * X)^-1 * X^T * y
# 	a = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)[0, 0]
# 	b = (sum(list_y) - a * sum(list_x)) / n
# 	return a, b
