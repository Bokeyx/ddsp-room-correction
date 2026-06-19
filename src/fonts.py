"""Register the bundled Korean font so matplotlib renders Korean labels.

matplotlib's default fonts have no Korean glyphs, so Korean plot text shows as
tofu boxes on a server (and anywhere without a CJK font). We ship NanumGothic
(SIL Open Font License) and register it here. Pure-ish: it mutates matplotlib
rcParams as a side effect and returns the resolved family name, so a test can
confirm the font is bundled and loadable.
"""
import os

from matplotlib import font_manager, rcParams

FONT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets", "fonts", "NanumGothic-Regular.ttf",
)


def register_korean_font(path=FONT_PATH):
    """Register the bundled Korean font with matplotlib and make it the default.

    Returns the resolved font family name. Raises FileNotFoundError if the font
    file is missing (so a broken bundle fails loudly rather than silently
    falling back to tofu boxes).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    font_manager.fontManager.addfont(path)
    name = font_manager.FontProperties(fname=path).get_name()
    rcParams["font.family"] = name
    rcParams["axes.unicode_minus"] = False  # keep the minus sign rendering
    return name
