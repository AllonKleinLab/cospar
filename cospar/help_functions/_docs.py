from textwrap import dedent


def _doc_params(**kwds):
    """\
    Docstrings should start with "\" in the first line for proper formatting.
    """

    def dec(obj):
        obj.__orig_doc__ = obj.__doc__
        obj.__doc__ = dedent(obj.__doc__).format_map(kwds)
        return obj

    return dec


selected_fates = """\
selected_fates: `list`
    List of cluster ids consistent with adata.obs['state_info'].
    It allows a nested structure. If so, we merge clusters within
    each sub-list into a mega-fate cluster.\
"""

map_source = """\
source: `str`
    The transition map to be used for plotting: {'transition_map',
    'intraclone_transition_map',...}. The actual available
    map depends on adata itself, which can be accessed at adata.uns['available_map']\
"""

map_backward = """\
map_backward: `bool`, optional (default: True)
    If `map_backward=True`, show fate properties of initial cell states :math:`i`;
    otherwise, show progenitor properties of later cell states :math:`j`.
    This is used for building the fate map :math:`P_i(\mathcal{C})`. See :func:`.fate_map`.\
"""

fate_method = """\
method: `str`, optional (default: 'norm-sum')
    Method to obtain the fate probability map :math:`P_i(\mathcal{C})` towards a set
    of states annotated with fate :math:`\mathcal{C}`. Available options:
    {'sum', 'norm-sum'}. See :func:`.fate_map`.\
"""

sum_fate_prob_thresh = """\
sum_fate_prob_thresh: `float`, optional (default: 0.05)
    The fate bias of a state is plotted only when it has a cumulative fate
    probability to the combined cluster (A+B) larger than this threshold,
    i.e., P(i->A)+P(i+>B) >  sum_fate_prob_thresh.\
"""

selected_times = """\
selected_times: `list`, optional (default: all)
    A list of time points to further restrict the cell states to plot.
    The default choice is not to constrain the cell states to show.\
"""

all_source = """\
source: `str`
    Choices: {'X_clone', 'transition_map',
    'intraclone_transition_map',...}. If set to be 'clone', use only the clonal
    information. If set to be any of the precomputed transition map, use the
    transition map to compute the fate coupling. The actual available
    map depends on adata itself, which can be accessed at adata.uns['available_map']\
"""


rename_fates = """\
rename_fates: `list`, optional (default: None)
    Provide new names in substitution of names in selected_fates.
    For this to be effective, the new name list needs to have names
    in exact correspondence to those in the old list.\
"""


background = """\
background: `bool`, optional (default: True)
    If true, plot all cell states (t1+t2) in grey as the background.\
"""

show_histogram = """\
show_histogram: `bool`, optional (default: False)
    If true, show the distribution of inferred fate probability.\
"""

plot_target_state = """\
plot_target_state: `bool`, optional (default: True)
    If true, highlight the target clusters as defined in selected_fates.\
"""

color_bar = """\
color_bar: `bool`, optional (default: True)
    plot the color bar if True.\
"""

auto_color_scale = """\
auto_color_scale:
    True: automatically rescale the color range to match the value range.\
"""

target_transparency = """\
target_transparency: `float`, optional (default: 0.2)
    It controls the transparency of the plotted target cell states,
    for visual effect. Range: [0,1].\
"""

figure_index = """\
figure_index: `str`, optional (default: '')
    String index for annotate filename for saved figures. Used to distinuigh plots from different conditions.\
"""

mask = """\
mask: `np.array`, optional (default: None)
    A boolean array for available cell states. It should has the length as adata.shape[0].
    Especially useful to constrain the states to show fate bias.\
"""

color_map = """\
color_map:
    The color map (a matplotlib.pyplot.cm object) to visualize the result.\
"""
