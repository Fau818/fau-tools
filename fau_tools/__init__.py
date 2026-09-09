"""
A small tool that uses the PyTorch framework, providing assistance in completing classification task using CNN.

Features: train model, print training process, save training files, plot figures, etc.
"""

from importlib.metadata import PackageNotFoundError, version

from . import torch_tools
from .data_structure import TaskRunner
from .utils import calc_time, cprint, custom_notify, notify

try:
  __version__ = version("fau-tools")
except PackageNotFoundError:
  __version__ = "unknown"  # imported from a source tree, not from an install

__all__ = ["TaskRunner", "calc_time", "cprint", "custom_notify", "notify", "torch_tools"]
