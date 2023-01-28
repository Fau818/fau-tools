import torch.nn


def calc_time(function): ...

def time_to_human(time: float|int) -> str: ...

# def show_progress(now: int, total: int, time_manager: TimeManager=None, length: int=30, icons: str='█ ') -> None: ...

def cprint(text: str, color: str='red', sep: str='\n', ret: bool=False) -> None|str: ...


def __get_value_in_height_and_width(value: tuple[int,int]|int, value_name: str) -> tuple[int,int]: ...

def _calc_value_after_layer(x: int, k_size: int, stride: int, padding: int) -> int: ...

def calc_feature_size(channel: int, height: int, width: int, sequential: torch.nn.Sequential) -> int: ...


class ActivationFunction:
	@staticmethod
	def sigmoid(x: float) -> float: ...

	@staticmethod
	def tanh(x: float) -> float: ...

	@staticmethod
	def relu(x: float) -> float: ...
