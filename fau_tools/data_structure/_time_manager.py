import time

import fau_tools.utils as utils


class _Timer:
  def __init__(self):
    self._time_list: list[float] = []


  def _is_calculable_timer(self):
    if len(self._time_list) < 2:
      utils.cprint("Warning: Timer does not have enough length.", color="yellow")
      return False
    return True


  def tick_time(self): self._time_list.append(time.time())


  # Fewer than two ticks means nothing was measured yet, so these report 0
  # instead of leaking `None` into every caller.
  def get_average_time(self) -> float:
    if not self._is_calculable_timer(): return 0.0
    return self.get_elapsed_time() / (len(self._time_list) - 1)


  def get_last_time_gap(self) -> float:
    if not self._is_calculable_timer(): return 0.0
    return self._time_list[-1] - self._time_list[-2]


  def get_elapsed_time(self) -> float:
    if not self._is_calculable_timer(): return 0.0
    return self._time_list[-1] - self._time_list[0]


class TimeManager:
  """Count the time consuming."""

  def __init__(self):
    self._time_dict: dict[str|None, _Timer] = {}


  def get_timer(self, name: str|None=None) -> _Timer:
    if name not in self._time_dict: self._time_dict[name] = _Timer()
    return self._time_dict[name]


  def tick_time(self, name: str|None=None) -> None:
    timer = self.get_timer(name)
    timer.tick_time()


  def get_average_time(self, name: str|None=None) -> float:
    timer = self.get_timer(name)
    return timer.get_average_time()


  def get_last_time_gap(self, name: str|None=None) -> float:
    timer = self.get_timer(name)
    return timer.get_last_time_gap()


  def get_elapsed_time(self, name: str|None=None) -> float:
    timer = self.get_timer(name)
    return timer.get_elapsed_time()
