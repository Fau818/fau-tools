import torch

import fau_tools.utils._utils as utils


def _get_device_name(device: torch.device) -> str:
  return torch.cuda.get_device_name(device.index) if device.type == "cuda" else device.type


def determine_device(return_name: bool=False) -> str|tuple[str, str]:
  """
  Determine the device used in pytorch automatically.

  Parameters
  ----------
  return_name : whether return the name of device

  Returns
  -------
  torch.device or (torch.device, device_name)

  """
  CUDA_DEVICE, MPS_DEVICE, CPU_DEVICE = "cuda:0", "mps", "cpu"
  device = None

  # cuda
  try:
    if torch.cuda.is_available(): device = torch.device(CUDA_DEVICE)
  except AssertionError: utils.cprint("No cuda detected.", color="yellow")

  # mps or cpu
  if device is None:
    if torch.backends.mps.is_available(): device = torch.device(MPS_DEVICE)
    else: device = torch.device(CPU_DEVICE)

  device_name = _get_device_name(device)

  return (device, device_name) if return_name else device


def parse_device(device: str|torch.device, return_name: bool=False) -> str|tuple[str, str]:
  """
  Parse the `device` to ensure is a torch.device.

  Returns
  -------
  Return the torch.device; if `return_name == True`, will return (torch.device, device_name)

  """
  if device is None: return determine_device(return_name)

  if isinstance(device, torch.device): device_name = _get_device_name(device)
  elif isinstance(device, str):
    try: device = torch.device(device)
    except RuntimeError as runtime_error: utils.notify_exception(runtime_error)
    device_name = _get_device_name(device)
  else: utils.notify_exception(TypeError("`device` is not the `torch.device` or `str` type."))

  return (device, device_name) if return_name else device
