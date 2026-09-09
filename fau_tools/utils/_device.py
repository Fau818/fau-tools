import typing

import torch

from ._color_print import cprint, notify_exception

__all__ = ["determine_device", "parse_device"]


def _get_device_name(device: torch.device) -> str:
  if device.type == "cuda" and torch.cuda.is_available(): return torch.cuda.get_device_name(device.index)
  return device.type


@typing.overload
def determine_device(return_name: typing.Literal[False]=False) -> torch.device: ...
@typing.overload
def determine_device(return_name: typing.Literal[True]) -> tuple[torch.device, str]: ...

def determine_device(return_name: bool=False) -> torch.device|tuple[torch.device, str]:
  """
  Determine the device used in pytorch automatically.

  Parameters
  ----------
  return_name : whether return the name of device

  Returns
  -------
  torch.device or (torch.device, device_name)

  """
  CUDA_DEVICE, MPS_DEVICE, CPU_DEVICE = "cuda", "mps", "cpu"
  device = None

  # cuda
  try:
    if torch.cuda.is_available(): device = torch.device(CUDA_DEVICE)
  except AssertionError: cprint("No cuda detected.", color="yellow")

  # mps or cpu
  if device is None:
    if torch.backends.mps.is_available(): device = torch.device(MPS_DEVICE)
    else: device = torch.device(CPU_DEVICE)

  device_name = _get_device_name(device)

  return (device, device_name) if return_name else device


@typing.overload
def parse_device(device: str|torch.device|None, return_name: typing.Literal[False]=False) -> torch.device: ...
@typing.overload
def parse_device(device: str|torch.device|None, return_name: typing.Literal[True]) -> tuple[torch.device, str]: ...

def parse_device(device: str|torch.device|None, return_name: bool=False) -> torch.device|tuple[torch.device, str]:
  """
  Parse the `device` to ensure is a torch.device.

  Returns
  -------
  Return the torch.device; if `return_name == True`, will return (torch.device, device_name)

  """
  if device is None: device = determine_device()
  elif not isinstance(device, torch.device):
    try: device = torch.device(device)
    except (RuntimeError, TypeError) as error: notify_exception(error)

  assert isinstance(device, torch.device)  # `notify_exception` has already exited on anything else
  device_name = _get_device_name(device)
  return (device, device_name) if return_name else device
