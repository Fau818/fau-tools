import time


def get_device(return_name=False):
  """
  Determine the optimal device used in pytorch.

  Parameters
  ----------
  return_name : whether return the name of device

  Returns
  -------
  device or (device, device_name)

  """
  import torch

  CUDA_DEVICE, MPS_DEVICE, CPU_DEVICE = "cuda:0", "mps", "cpu"
  device, device_name = None, None

  # cuda
  try:
    if torch.cuda.is_available(): device, device_name = torch.device(CUDA_DEVICE), torch.cuda.get_device_name(0)
  except AssertionError: ...  # cprint("No cuda detected.", color="yellow")
  except Exception: notify("fau_tools", f"Unknown error in {get_device.__module__}.{get_device.__qualname__} function.", "error")

  # mps or cpu
  if device is None:
    if torch.backends.mps.is_available(): device, device_name = torch.device(MPS_DEVICE), MPS_DEVICE
    else: device, device_name = torch.device(CPU_DEVICE), CPU_DEVICE

  return device, device_name if return_name else device


def calc_time(function):
  """A decorator to display the running time of a function."""
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


def time_to_human(time):
  """
  Convert time in seconds to the human-friendly time display.

  Parameters
  ----------
  time : time in seconds

  Returns
  -------
  A string with format HH:mm:ss;
  but if the time is more than one day, will be "MTOD";
  but if the time is less than one second, will be "minor".

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


# -----------------------------------
# -------- Colorful print
# -----------------------------------
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
    """A lua style dict to store components of color print."""
    start   = "\033["
    end     = "\033[0m"
    fg_lead = "38;2;"
    bg_lead = "48;2;"
    bold    = "1;"
    italic  = "3;"


  @staticmethod
  def _hex2dec(color_hex):
    """
    Convert color format from `#RRGGBB` to `(R, G, B)`.

    Parameters
    ----------
    color_hex : the hexadecimal representation of a color

    Returns
    -------
    A tuple `(R, G, B)` represents the color in decimal.

    """
    color_hex = color_hex.lstrip("#")
    assert len(color_hex) == 6, "Error in color format, excepted `#RRGGBB`."
    r_hex, g_hex, b_hex = [color_hex[index:index+2] for index in range(0, 6, 2)]
    r_dec, g_dec, b_dec = [int(v_hex, 16) for v_hex in (r_hex, g_hex, b_hex)]
    return (r_dec, g_dec, b_dec)


  @staticmethod
  def _dec2hex(color_dec):
    """
    Convert color format from `(R, G, B)` to `#RRGGBB`.

    Parameters
    ----------
    color_dec : The decimal representation of a color.

    Returns
    -------
    A string with format `#RRGGBB` represents the color in hexadecimal.

    """
    color_hex = "".join(hex(v_dec)[2:].zfill(2).upper() for v_dec in color_dec)
    return color_hex


  @classmethod
  def _get_fg_by_bg(cls, bg_dec):
    """
    Determine the fg color by bg color.

    Parameters
    ----------
    bg_dec : the decimal representation of a color

    Returns
    -------
    A string represents the fg color in hexadecimal.

    """
    r_dec, g_dec, b_dec = bg_dec
    luminance = (0.299 * r_dec + 0.587 * g_dec + 0.114 * b_dec) / 255
    is_bright = luminance > 0.5
    return cls.BASIC_COLORS["black"] if is_bright else cls.BASIC_COLORS["white"]


  @classmethod
  def _get_color_pattern(cls, color, bold=False, italic=False, solid=False):
    """Return the colorful string pattern except the BEGIN and END parts."""
    def parse_color(color_hex, solid=False):
      if not solid:
        fg = cls._hex2dec(color_hex)
        return f"{cls.Component.fg_lead}{fg[0]};{fg[1]};{fg[2]}m"
      else:
        bg = cls._hex2dec(color_hex)
        fg = cls._hex2dec(cls._get_fg_by_bg(bg))
        return f"{cls.Component.fg_lead}{fg[0]};{fg[1]};{fg[2]};{cls.Component.bg_lead}{bg[0]};{bg[1]};{bg[2]}m"

    if color not in cls.BASIC_COLORS: raise ValueError(f"color should be defined in `{cls.BASIC_COLORS.__name__}`, but got `{color}`.")

    color_hex = cls.BASIC_COLORS[color]
    style = (cls.Component.bold if bold else "") + (cls.Component.italic if italic else "")
    return f"{style}{parse_color(color_hex, solid)}"


  @classmethod
  def cprint(cls, *values, color='red', bold=False, italic=False, solid=False, show=True, sep=' ', end='\n', **kwargs):
    """
    Colorful printer.

    Parameters
    ----------
    values : the content to be printed
    color  : the color of the string
    bold   : whether to use bold text
    italic : whether to use italic text
    solid  : whether to use the color as background color.
    show   : if is True, the colorful string will be printed; otherwise, will only be returned.
    sep    : the kwarg in `print()` function
    end    : the kwarg in `print()` function

    Returns
    -------
    The colorful string.

    """
    color_pattern = cls._get_color_pattern(color, bold, italic, solid)
    color_string = f"{cls.Component.start}{color_pattern}{sep.join(str(value) for value in values)}{cls.Component.end}"
    if show: print(color_string, sep=sep, end=end, **kwargs)
    return color_string


  @classmethod
  def custom_notify(cls, title, content, color, show=True):
    """
    Customize the notice.

    Parameters
    ----------
    title   : the title of notice
    content : the content of notice
    color   : a string or tuple indicates the color of title and content
    show    : whether to print

    Returns
    -------
    The colorful string.

    """
    if isinstance(color, str): title_color = content_color = color
    elif isinstance(color, (tuple, list)):
      if len(color) != 2: raise ValueError(cls.cprint("The length of `color` should be 2.", color="red", show=False))
      title_color, content_color = color
    else: raise TypeError(cls.cprint(f"color should be a string or a tuple, but got {type(color)}.", color="red", show=False))

    ctitle   = cls.cprint(f" {title} ", color=title_color, bold=True, solid=True, show=False)
    cconcent = cls.cprint(content, color=content_color, show=False)
    ctext = " ".join((ctitle, cconcent))

    if show: print(ctext)
    return ctext


  @classmethod
  def notify(cls, title, content, notify_type="info", show=True):
    """
    Notify a message.

    Parameters
    ----------
    title       : the title of notice
    content     : the content of notice
    notify_type : the type of notice
    show        : whether to print

    Returns
    -------
    The colorful string.

    """
    TYPE_COLORS = {
      "info"   : "blue",
      "warn"   : "yellow", "warning": "yellow",
      "error"  : "red",
      "success": "green",
    }

    notify_type = notify_type.lower()
    if notify_type not in TYPE_COLORS.keys(): raise ValueError(cls.cprint("`notify_type` should be defined in `TYPE_COLORS`.", color="red", show=False))

    ctext = cls.custom_notify(title, content, TYPE_COLORS[notify_type], show=False)

    if show: print(ctext)
    return ctext


cprint        = Color.cprint
custom_notify = Color.custom_notify
notify        = Color.notify



# ------------------------------------------------------------
# --------------- Auto calculate feature size
# ------------------------------------------------------------
# TODO: integrate to a class and test.
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
