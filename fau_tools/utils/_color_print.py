class Color:
  BASIC_COLORS = {
    "black"  : "#000000", "B": "#000000",
    "red"    : "#C91B00", "r": "#C91B00",
    "green"  : "#00BF00", "g": "#00BF00",
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
  def _hex2dec(color_hex: str) -> tuple[int, int, int]:
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
  def _dec2hex(color_dec: tuple[int, int, int]) -> str:
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
  def _get_fg_by_bg(cls, bg_dec: tuple[int, int, int]) -> str:
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
  def _get_color_pattern(cls, color: str, bold: bool=False, italic: bool=False, solid: bool=False) -> str:
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
  def cprint(
    cls,
    *values,
    color: str="red",
    bold: bool=False, italic: bool=False, solid: bool=False,
    show: bool=True,
    sep: str=' ',
    end: str='\n',
    **kwargs
  ) -> str:
    """
    Colorful printer.

    Parameters
    ----------
    values : the content to be printed
    color  : the color of the string
    bold   : whether to use bold text
    italic : whether to use italic text
    solid  : whether to use the color as background color
    show   : if is True, the colorful string will be printed; otherwise, will only be returned
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
  def custom_notify(cls, title: str, content: str, color: str|tuple[str, str], show: bool=True) -> str:
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
  def notify(cls, title: str, content: str, notify_type: str="info", show: bool=True) -> str:
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


  @classmethod
  def notify_exception(cls, error: Exception, exit: bool=True):
    cls.notify(error.__class__.__name__, error, notify_type="error")
    if exit: raise SystemExit(1)


cprint           = Color.cprint
custom_notify    = Color.custom_notify
notify           = Color.notify
notify_exception = Color.notify_exception
