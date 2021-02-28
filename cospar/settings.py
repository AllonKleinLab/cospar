"""Settings
"""

verbosity = 3
"""Verbosity level (0=errors, 1=warnings, 2=info, 3=hints)
"""

plot_prefix = "scvelo_"
"""Global prefix that is appended to figure filenames.
"""

data_path = "data_cospar"
"""Directory where adata is stored (default 'data_cospar').
"""

figure_path = "figure_cospar"
"""Directory where plots are saved (default 'figure_cospar').
"""

file_format_figs = "pdf"
"""File format for saving figures.
For example 'png', 'pdf' or 'svg'. Many other formats work as well (see
`matplotlib.pyplot.savefig`).
"""

fig_width=4
fig_height=3.5
fig_point_size=2


logfile = ""
"""Name of logfile. By default is set to '' and writes to standard output."""

# --------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------

from matplotlib import rcParams, cm, colors, cbook
from cycler import cycler
import warnings

warnings.filterwarnings("ignore", category=cbook.mplDeprecation)


def set_rcParams_cospar(fontsize=12, color_map=None, frameon=None):
    """Set matplotlib.rcParams to scvelo defaults."""

    # dpi options (mpl default: 100, 100)
    rcParams["figure.dpi"] = 100
    rcParams["savefig.dpi"] = 150

    # figure (mpl default: 0.125, 0.96, 0.15, 0.91)
    rcParams["figure.figsize"] = (6, 4)
    # rcParams["figure.subplot.left"] = 0.18
    # rcParams["figure.subplot.right"] = 0.96
    # rcParams["figure.subplot.bottom"] = 0.15
    # rcParams["figure.subplot.top"] = 0.91

    # lines (defaults:  1.5, 6, 1)
    rcParams["lines.linewidth"] = 1.5  # the line width of the frame
    rcParams["lines.markersize"] = 6
    rcParams["lines.markeredgewidth"] = 1

    # font
    rcParams["font.sans-serif"] = [
        "Arial",
        "Helvetica",
        "DejaVu Sans",
        "Bitstream Vera Sans",
        "sans-serif",
    ]

    fontsize = fontsize
    labelsize = 0.92 * fontsize

    # fonsizes (mpl default: 10, medium, large, medium)
    rcParams["font.size"] = fontsize
    rcParams["legend.fontsize"] = labelsize
    rcParams["axes.titlesize"] = fontsize
    rcParams["axes.labelsize"] = labelsize

    # legend (mpl default: 1, 1, 2, 0.8)
    rcParams["legend.numpoints"] = 1
    rcParams["legend.scatterpoints"] = 1
    rcParams["legend.handlelength"] = 0.5
    rcParams["legend.handletextpad"] = 0.4
    rcParams["pdf.fonttype"] = 42

    # color cycle
    #rcParams["axes.prop_cycle"] = cycler(color=vega_10)

    # axes
    rcParams["axes.linewidth"] = 0.8
    rcParams["axes.edgecolor"] = "black"
    rcParams["axes.facecolor"] = "white"

    # ticks (mpl default: k, k, medium, medium)
    rcParams["xtick.color"] = "k"
    rcParams["ytick.color"] = "k"
    rcParams["xtick.labelsize"] = labelsize
    rcParams["ytick.labelsize"] = labelsize

    # axes grid (mpl default: False, #b0b0b0)
    rcParams["axes.grid"] = False
    rcParams["grid.color"] = ".8"

    # color map
    rcParams["image.cmap"] = "Reds" if color_map is None else color_map

    # frame (mpl default: True)
    frameon = False if frameon is None else frameon
    global _frameon
    _frameon = frameon

# def set_up_plotting(fontsize):
#     """
#     Change matplotlib setting for beautiful plots.
#     """

#     plt.rc('font', family='sans-serif')
#     plt.rcParams['font.sans-serif']=['Helvetica']
#     plt.rc('xtick',labelsize=12) #14
#     plt.rc('ytick', labelsize=12) #14
#     #plt.rc('font', weight='bold')
#     plt.rc('font', weight='regular')
#     plt.rcParams.update({'font.size': fontsize}) #16
#     #plt.rcParams['axes.labelweight'] = 'bold'
#     plt.rcParams['axes.labelweight'] = 'regular'
#     #plt.rcParams['pdf.fonttype'] = 42 #make the figure editable, this comes with a heavy cost of file size


def set_figure_params(
    style="cospar",
    dpi=100,
    dpi_save=300,
    frameon=None,
    vector_friendly=True,
    transparent=True,
    fontsize=14,
    figsize=None,
    pointsize=2,
    color_map=None,
    facecolor=None,
    format="pdf",
    ipython_format="png2x",
):
    """Set resolution/size, styling and format of figures.

    Arguments
    ---------
    style : `str` (default: `None`)
        Init default values for ``matplotlib.rcParams`` suited for `scvelo` or `scanpy`.
        Use `None` for the default matplotlib values.

    dpi : `int` (default: `None`)
        Resolution of rendered figures - affects the size of figures in notebooks.
    dpi_save : `int` (default: `None`)
        Resolution of saved figures. This should typically be higher to achieve
        publication quality.
    frameon : `bool` (default: `None`)
        Add frames and axes labels to scatter plots.
    vector_friendly : `bool` (default: `True`)
        Plot scatter plots using `png` backend even when exporting as `pdf` or `svg`.
    transparent : `bool` (default: `True`)
        Save figures with transparent back ground. Sets
        `rcParams['savefig.transparent']`.
    fontsize : `int` (default: 14)
        Set the fontsize for several `rcParams` entries.
    figsize: `[float, float]` (default: `None`)
        Width and height for default figure size.
    color_map : `str` (default: `None`)
        Convenience method for setting the default color map.
    facecolor : `str` (default: `None`)
        Sets backgrounds `rcParams['figure.facecolor']`
        and `rcParams['axes.facecolor']` to `facecolor`.
    format : {'png', 'pdf', 'svg', etc.} (default: 'pdf')
        This sets the default format for saving figures: `file_format_figs`.
    ipython_format : list of `str` (default: 'png2x')
        Only concerns the notebook/IPython environment; see
        `IPython.core.display.set_matplotlib_formats` for more details.
    """
    try:
        import IPython

        if isinstance(ipython_format, str):
            ipython_format = [ipython_format]
        IPython.display.set_matplotlib_formats(*ipython_format)
    except:
        pass

    global _rcParams_style
    _rcParams_style = style
    global _vector_friendly
    _vector_friendly = vector_friendly
    global file_format_figs
    file_format_figs = format
    if transparent is not None:
        rcParams["savefig.transparent"] = transparent
    if facecolor is not None:
        rcParams["figure.facecolor"] = facecolor
        rcParams["axes.facecolor"] = facecolor
    if style == "cospar":
        set_rcParams_cospar(fontsize=fontsize, color_map=color_map, frameon=frameon)
    # Overwrite style options if given
    if figsize is not None:
        rcParams["figure.figsize"] = figsize
        global fig_width
        global fig_height
        fig_width=figsize[0]
        fig_height=figsize[1]
    if dpi is not None:
        rcParams["figure.dpi"] = dpi
    if dpi_save is not None:
        rcParams["savefig.dpi"] = dpi_save

    global fig_point_size
    fig_point_size=pointsize

def set_rcParams_defaults():
    """Reset `matplotlib.rcParams` to defaults."""
    from matplotlib import rcParamsDefault

    rcParams.update(rcParamsDefault)





def _set_start_time():
    from time import time

    return time()


_start = _set_start_time()
"""Time when the settings module is first imported."""

_previous_time = _start
"""Variable for timing program parts."""
