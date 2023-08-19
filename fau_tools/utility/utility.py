import time

import numpy as np


# ------------------------------------------------------------
# --------------- a decorator can show function running time
# ------------------------------------------------------------
def calc_time(function):
  def wrapper(*args, **kwargs):
    print('-' * 15, "BEGIN", function.__name__, '-' * 15)
    BEGIN = time.time()
    res = function(*args, **kwargs)
    END = time.time()
    COST_TIME = time_to_human(END - BEGIN)
    print(f"{function.__name__} cost: {COST_TIME}")
    print('-' * 15, " END ", function.__name__, '-' * 15)
    return res

  wrapper.__name__ = function.__name__  # to keep the function origin name
  return wrapper



# ----------------------------------------------------------------------
# --------------- a function can convert time to human-friendly display
# ----------------------------------------------------------------------
def time_to_human(time):
  """
  Convert time in seconds to the human-friendly time display.

  Parameters
  ----------
  time : time in seconds

  Returns
  -------
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



# ------------------------------------------------------------
# --------------- color print function
# ------------------------------------------------------------
class Color:
  BASIC_COLORS = {
    "black"  : "#000000", "B": "#000000",
    "red"    : "#C91B00", "r": "#C91B00",
    "green"  : "#00DC00", "g": "#00DC00",
    "yellow" : "#EEEE55", "y": "#EEEE55",
    "blue"   : "#007DFF", "b": "#007DFF",
    "magenta": "#AD83E9", "m": "#AD83E9", "p": "#AD83E9",
    "cyan"   : "#30E1FD", "c": "#30E1FD",
    "white"  : "#FFFFFF", "w": "#FFFFFF",
  }


  class Component:
    """A lua style dict."""
    start   = "\033["
    end     = "\033[0m"
    fg_lead = "38;2;"
    bg_lead = "48;2;"
    bold    = "1;"
    italic  = "3;"


  @staticmethod
  def hex2dec(color_hex: str):
    """
    Convert `#RRGGBB` to `(R, G, B)`.

    Parameters
    ----------
    color_hex : The hexadecimal representation of a color.

    Returns
    -------
    A tuple (R, G, B) represents the color in decimal.

    """
    color_hex = color_hex.lstrip("#")
    assert len(color_hex) == 6, "Error in color format, excepted `#RRGGBB`."
    r_hex, g_hex, b_hex = [color_hex[index:index+2] for index in range(0, 6, 2)]
    r_dec, g_dec, b_dec = [int(v_hex, 16) for v_hex in (r_hex, g_hex, b_hex)]
    return (r_dec, g_dec, b_dec)


  @staticmethod
  def dec2hex(color_dec: tuple[int,int,int]):
    """
    Convert `(R, G, B)` to `#RRGGBB`.

    Parameters
    ----------
    color_dec : The decimal representation of a color.

    Returns
    -------
    A string `#RRGGBB` represents the color in hexadecimal.

    """
    color_hex = "".join(hex(v_dec)[2:].zfill(2).upper() for v_dec in color_dec)
    return color_hex


  @classmethod
  def get_fg_by_bg(cls, bg_dec):
    """
    Determine the fg color by bg color.

    Parameters
    ----------
    bg_dec : The decimal representation of a color.

    Returns
    -------
    A string represents the fg color in hexadecimal.

    """
    r_dec, g_dec, b_dec = bg_dec
    luminance = (0.299 * r_dec + 0.587 * g_dec + 0.114 * b_dec) / 255
    is_bright = luminance > 0.5
    return cls.BASIC_COLORS["black"] if is_bright else cls.BASIC_COLORS["white"]


  @classmethod
  def parse_color(cls, color_hex, solid=False):
    if not solid:
      fg = cls.hex2dec(color_hex)
      return f"{cls.Component.fg_lead}{fg[0]};{fg[1]};{fg[2]}m"
    else:
      bg = cls.hex2dec(color_hex)
      fg = cls.hex2dec(cls.get_fg_by_bg(bg))
      return f"{cls.Component.fg_lead}{fg[0]};{fg[1]};{fg[2]};{cls.Component.bg_lead}{bg[0]};{bg[1]};{bg[2]}m"


  @classmethod
  def get_color_pattern(cls, color, bold=False, italic=False, solid=False):
    if color not in cls.BASIC_COLORS: raise ValueError(f"color should be defined in `BASIC_COLORS`, got `{color}`.")

    color_hex = cls.BASIC_COLORS[color]
    style = (cls.Component.bold if bold else "") + (cls.Component.italic if italic else "")
    return f"{style}{cls.parse_color(color_hex, solid)}"


  @classmethod
  def cprint(cls, *values, color='red', bold=False, italic=False, solid=False, show=True, sep=' ', end='\n', **kwargs):
    """
    Colorful printer.

    Parameters
    ----------
    values : the contents need to be printed
    color  : a string representing color; all the valid values shown in `__COLOR_DICT`
    bold   : whether to use bold text
    italic : whether to use italic text
    solid  : whether to use the color as background color.
    show   : if True, the colorful string will be printed; otherwise, will be returned.
    sep    : a kwarg in `print()` function
    end    : a kwarg in `print()` function

    Returns
    -------
    return the colorful string.

    """
    color_pattern = cls.get_color_pattern(color, bold, italic, solid)
    color_string = f"{cls.Component.start}{color_pattern}{sep.join(str(value) for value in values)}{cls.Component.end}"
    if show: print(color_string, sep=sep, end=end, **kwargs)
    return color_string

  @classmethod
  def custom_notify(cls, title, content, color, show=True):
    """
    Customize the notify message.

    Parameters
    ----------
    title   :  the title of notify
    content :  the content of notify
    color   :  a string or a tuple indicating the color of title and content
    show    :  whether to print

    Returns
    -------
    the colorful string.

    """
    if isinstance(color, str): title_color = content_color = color
    elif isinstance(color, (tuple, list)): title_color, content_color = color
    else: raise TypeError(f"color should be a string or a tuple, but got {type(color)}.")

    ctitle   = cls.cprint(f" {title} ", color=title_color, bold=True, solid=True, show=False)
    cconcent = cls.cprint(content, color=content_color, show=False)
    ctext = " ".join((ctitle, cconcent))

    if show: print(ctext)
    return ctext


  @classmethod
  def notify(cls, title, content, style="info", show=True):
    """
    Notify message.

    Parameters
    ----------
    title   : the title of notify
    content : the content of notify
    style   : the style of notify
    show    : whether to print

    Returns
    -------
    the colorful string.

    """
    STYLE_COLORS = {
      "info"   : "blue",
      "warn"   : "yellow", "warning": "yellow",
      "error"  : "red",
      "success": "green",
    }

    level = level.lower()
    if level not in STYLE_COLORS.keys(): raise ValueError(f"`level` should be defined in `STYLE_COLORS`.")

    ctext = cls.custom_notify(title, content, STYLE_COLORS[style], show=False)

    if show: print(ctext)
    return ctext


cprint        = Color.cprint
custom_notify = Color.custom_notify
notify        = Color.notify


if __name__ == "__main__":
  cprint("hello", color="g", bold=True, solid=True)
  ...



# ------------------------------------------------------------
# --------------- auto calculate feature size
# ------------------------------------------------------------
def __get_value_in_height_and_width(value, value_name):
  if isinstance(value, tuple): return value
  if isinstance(value, int): return value, value

  raise TypeError(f"The type of {value_name} requires int|tuple, but got {type(value)}.")


def _calc_value_after_layer(x, k_size, stride, padding): return (x - k_size + 2 * padding) // stride + 1


def calc_feature_size(channel, height, width, sequential):
  """
  Calculate the number of neurons of the convolutional layer to fully connected layer.

  Parameters
  ----------
  channel : the channel of input image
  height : the height of input image
  width : the width of input image
  sequential : the convolutional layers sequential function

  Returns
  -------
  An integer, indicating the number of neurons.

  """
  CONV, POOL = "torch.nn.modules.conv", "torch.nn.modules.pooling"
  for op in sequential:
    if op.__module__ == CONV:
      # get basic parameters
      in_channel, out_channel = op.in_channels, op.out_channels
      k_size, stride, padding = op.kernel_size, op.stride, op.padding

      # illegal channel
      if in_channel != channel: raise ValueError(f"Got {channel=}, but {in_channel=} in Conv2d.")

      # get values in height and width
      k_size_h, k_size_w = __get_value_in_height_and_width(k_size, "kernel_size")
      stride_h, stride_w = __get_value_in_height_and_width(stride, "stride")
      padding_h, padding_w = __get_value_in_height_and_width(padding, "padding")

      # calculate
      channel = out_channel
      height = _calc_value_after_layer(height, k_size_h, stride_h, padding_h)
      width = _calc_value_after_layer(width, k_size_w, stride_w, padding_w)
    elif op.__module__ == POOL:
      k_size, stride, padding = op.kernel_size, op.stride, op.padding

      # get values in height and width
      k_size_h, k_size_w = __get_value_in_height_and_width(k_size, "kernel_size")
      stride_h, stride_w = __get_value_in_height_and_width(stride, "stride")
      padding_h, padding_w = __get_value_in_height_and_width(padding, "padding")

      height = _calc_value_after_layer(height, k_size_h, stride_h, padding_h)
      width = _calc_value_after_layer(width, k_size_w, stride_w, padding_w)

  return channel * height * width



# ------------------------------------------------------------
# --------------- activation function definition
# ------------------------------------------------------------
class ActivationFunction:
  @staticmethod
  def sigmoid(x): return 1 / (1 + np.exp(-x))

  @staticmethod
  def tanh(x): return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

  @staticmethod
  def relu(x): return np.maximum(x, 0)



# # ------------------------------------------------------------
# # --------------- a function can show process bar (deprecated)
# # ------------------------------------------------------------
# def show_progress(now, total, time_manager=None, length=30, icons='█ '):
#   """
#   A function that displays a progress bar.
#
#   Args:
#     now (): current process
#     total (): total process
#     time_manager (): for showing the running time and predicting the end time;
#       it should be an instance of `TimeManager` class.
#     length (): the length of process bar
#     icons (): the process icons; a string contained only two char is necessary;
#       the first char is the finished part icon, the second is unfinished.
#
#
#
#   Returns: None
#   """
#
#   if len(icons) != 2: raise ValueError(f"the length of icons arg must be 2, but {len(icons)} is received.")
#
#   finish_icon, unfinish_icon = icons
#   percent = now / total
#
#   # for showing process bar
#   finish_bar = int(percent * length) * finish_icon
#   unfinish_bar = (length - len(finish_bar)) * unfinish_icon
#   show = f"|{finish_bar}{unfinish_bar}|"
#
#   if time_manager:  # for showing time process:
#     average_time, elapsed_time = time_manager.get_average_time(), time_manager.get_elapsed_time()
#     total_time = total * average_time
#
#     elapsed_time = time_to_human(elapsed_time)
#     total_time = time_to_human(total_time)
#     show += f" [{now}/{total}, {elapsed_time}<{total_time}]"
#
#   print(f"\r{show}", end="")


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# def calc_grandient(f, x):
#   """ others method
#   def function(x):  # 定义函数
#     return -x[0] ** 2 - x[1] ** 2 + 2 * x[1] + 1  # f(z) = -x^2 - y^2 + 2y + 1
#   :param f:
#   :param x:
#   :return:
#   """
#   h = 1e-4  # 定义一个微小量，不能太小，太小计算机没法正确表示
#   grad = np.zeros_like(x)  # 生成和x形状相同的数组
#   for idx in range(x.size):  # 计算所有偏导
#     tmp_val = x[idx]
#     x[idx] = tmp_val + h  # 要计算的那个自变量加h，其余不变
#     fxh1 = f(x)  # 计算f(x+h)

#     x[idx] = tmp_val - h  # 计算f(x-h)
#     fxh2 = f(x)

#     grad[idx] = (fxh1 - fxh2) / (2 * h)  # 计算偏导
#     x[idx] = tmp_val
#   return grad


# def calc_mse(list_y, list_yh):
#   total = 0
#   for x, y in zip(list_y, list_yh):
#     total += (x - y) ** 2
#   return total / len(list_y)


# def calc_rmse(list_y, list_yh):
#   return math.sqrt(calc_mse(list_y, list_yh))


# def calc_nrmse(list_y, list_yh):
#   return calc_rmse(list_y, list_yh) / (sum(list_y) / len(list_y))


# def calc_mae(list_y, list_yh):
#   total = 0
#   for x, y in zip(list_y, list_yh):
#     total += abs(x - y)
#   return total / len(list_y)


# def calc_lsm(list_x, list_y):
#   n = len(list_x)
#   X, Y = np.mat(list_x).reshape(n, 1), np.mat(list_y).reshape(n, 1)

#   # (X^T * X)^-1 * X^T * y
#   a = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)[0, 0]
#   b = (sum(list_y) - a * sum(list_x)) / n
#   return a, b
