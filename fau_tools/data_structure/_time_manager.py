import time

from fau_tools import utils


class _Timer:
  def __init__(self):
    self._time_list: list[float] = list()


  def _is_calculable_timer(self):
    if len(self._time_list) < 2:
      utils.cprint("Warning: Timer does not have enough length.", color="yellow")
      return False
    return True


  def tick_time(self): self._time_list.append(time.time())


  def get_average_time(self) -> float:
    if not self._is_calculable_timer(): return None
    return self.get_elapsed_time() / (len(self._time_list) - 1)


  def get_last_time_gap(self) -> float:
    if not self._is_calculable_timer(): return None
    return self._time_list[-2] - self._time_list[-1]


  def get_elapsed_time(self) -> float:
    if not self._is_calculable_timer(): return None
    return self._time_list[-1] - self._time_list[0]


class TimeManager:
  """Count the time consuming."""

  def __init__(self):
    self._time_dict: dict[str, _Timer] = dict()


  def get_timer(self, name: str=None) -> _Timer:
    if name not in self._time_dict: self._time_dict[name] = _Timer()
    return self._time_dict[name]


  def tick_time(self, name: str=None) -> None:
    timer = self.get_timer(name)
    timer.tick_time()


  def get_average_time(self, name: str=None) -> float:
    timer = self.get_timer(name)
    return timer.get_average_time()


  def get_last_time_gap(self, name: str=None) -> float:
    timer = self.get_timer(name)
    return timer.get_last_time_gap()


  def get_elapsed_time(self, name: str=None) -> float:
    timer = self.get_timer(name)
    return timer.get_elapsed_time()
