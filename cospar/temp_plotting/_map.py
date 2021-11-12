import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as ssp
import scipy.stats as stats
import statsmodels.sandbox.stats.multicomp
from ete3 import Tree
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy

# from plotnine import *
from sklearn.manifold import SpectralEmbedding

from cospar import tool as tl

from .. import help_functions as hf
from .. import logging as logg
from .. import settings
from ._utils import (
    customized_embedding,
    fate_map_embedding,
    heatmap,
    plot_neighbor_joining,
    print_hierarchy,
    start_subplot_figure,
)


def fate_coupling(
    adata,
    source="transition_map",
    color_bar=True,
    rename_fates=None,
    color_map=plt.cm.Reds,
    figure_index="",
):
    """
    Plot fate coupling determined by the transition map.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    source: `str`, optional (default: 'transition_map')
        Choices: {'clone', 'transition_map',
        'intraclone_transition_map',...}. If set to be 'clone', use only the clonal
        information. If set to be any of the precomputed transition map, use the
        transition map to compute the fate coupling. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    color_bar: `bool`, optional (default: True)
        Plot the color bar.
    rename_fates: `list`, optional (default: [])
        Provide new names in substitution of names in selected_fates.
        For this to be effective, the new name list needs to have names
        in exact correspondence to those in the old list.

    Returns
    -------
    ax:
        The axis object of this plot.
    """

    fig_width = settings.fig_width
    fig_height = settings.fig_height
    figure_path = settings.figure_path

    data_des = adata.uns["data_des"][-1]

    if color_bar:
        fig_width = fig_width + 0.5

    key_word = f"fate_coupling_{source}"
    available_choices = hf.parse_output_choices(adata, key_word, where="uns")
    X_coupling = adata.uns[key_word]["X_coupling"]
    fate_names = adata.uns[key_word]["fate_names"]
    rename_fates = hf.rename_list(fate_names, rename_fates)

    ax = heatmap(
        X_coupling,
        x_ticks=rename_fates,
        y_ticks=rename_fates,
        order_map=True,
        color_bar_label=f"Fate coupling",
        color_bar=color_bar,
        fig_width=fig_width,
        fig_height=fig_height,
        color_map=color_map,
    )

    plt.tight_layout()
    ax.set_title(f"source: {source}")
    if figure_index != "":
        figure_index == f"_{figure_index}"
    plt.savefig(
        os.path.join(
            figure_path,
            f"{data_des}_{key_word}_{source}{figure_index}.{settings.file_format_figs}",
        )
    )
    return ax


def fate_hierarchy(
    adata,
    source="transition_map",
    rename_fates=None,
    plot_history=False,
):
    """
    Plot fate coupling determined by the transition map.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    source: `str`, optional (default: 'transition_map')
        Choices: {'clone', 'transition_map',
        'intraclone_transition_map',...}. If set to be 'clone', use only the clonal
        information. If set to be any of the precomputed transition map, use the
        transition map to compute the fate coupling. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    rename_fates: `list`, optional (default: None)
        Provide new names in substitution of names in selected_fates.
        For this to be effective, the new name list needs to have names
        in exact correspondence to those in the old list.

    Returns
    -------
    ax:
        The axis object of this plot.
    """

    key_word = f"fate_hierarchy_{source}"

    available_choices = hf.parse_output_choices(adata, key_word, where="uns")

    parent_map = adata.uns[key_word]["parent_map"]
    node_mapping = adata.uns[key_word]["node_mapping"]
    history = adata.uns[key_word]["history"]
    fate_names = adata.uns[key_word]["fate_names"]

    rename_fates = hf.rename_list(fate_names, rename_fates)

    print_hierarchy(parent_map, rename_fates)
    if plot_history:
        plot_neighbor_joining(
            settings.figure_path,
            node_mapping,
            history[0],
            history[1],
            history[2],
        )


def fate_map(
    adata,
    selected_fates=None,
    source="transition_map",
    selected_times=None,
    background=True,
    show_histogram=False,
    plot_target_state=True,
    auto_color_scale=True,
    color_bar=True,
    target_transparency=0.2,
    figure_index="",
    horizontal=False,
    mask=None,
    color_map=plt.cm.Reds,
):
    """
    Plot transition probability to given fate/ancestor clusters.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_fates: `list`, optional (default: all fates)
        List of cluster ids consistent with adata.obs['state_info'].
        It allows a nested list, where we merge clusters within
        each sub-list into a mega-fate cluster. If not set, plot all pre-computed
        fate maps.
    source: `str`, optional (default: 'transition_map')
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map',...}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    selected_times: `list`, optional (default: all)
        A list of time points to further restrict the cell states to plot.
        The default choice is not to constrain the cell states to show.
    background: `bool`, optional (default: True)
        If true, plot all cell states (t1+t2) in grey as the background.
    show_histogram: `bool`, optional (default: False)
        If true, show the distribution of inferred fate probability.
    plot_target_state: `bool`, optional (default: True)
        If true, highlight the target clusters as defined in selected_fates.
    color_bar: `bool`, optional (default: True)
        plot the color bar if True.
    target_transparency: `float`, optional (default: 0.2)
        It controls the transparency of the plotted target cell states,
        for visual effect. Range: [0,1].
    figure_index: `str`, optional (default: '')
        String index for annotate filename for saved figures. Used to distinuigh plots from different conditions.
    horizontal: `bool`, optional (default: False)
        If true, plot the figure panels horizontally; else, vertically.
    mask: `np.array`, optional (default: None)
        A boolean array for available cell states. It should has the length as adata.shape[0].
        Especially useful to constrain the states to show fate bias.

    Returns
    -------
    Fate map for each targeted fate cluster is updated at adata.obs[f'fate_map_{fate_name}']
    """

    key_word = f"fate_map_{source}"
    available_choices = hf.parse_output_choices(adata, key_word, where="obs")
    map_backward = adata.uns[key_word]["map_backward"]
    method = adata.uns[key_word]["method"]
    if map_backward:
        cell_id_t1 = adata.uns["Tmap_cell_id_t1"]
    else:
        cell_id_t1 = adata.uns["Tmap_cell_id_t2"]

    if method == "norm-sum":
        color_bar_label = "Progenitor prob."
    else:
        color_bar_label = "Fate prob."

    time_info = np.array(adata.obs["time_info"])
    sp_idx = hf.selecting_cells_by_time_points(time_info[cell_id_t1], selected_times)

    if selected_fates is None:
        mega_cluster_list = available_choices
    else:
        state_info = adata.obs["state_info"]
        (
            mega_cluster_list,
            valid_fate_list,
            __,
            sel_index_list,
        ) = hf.analyze_selected_fates(state_info, selected_fates)

    for j, fate_key in enumerate(mega_cluster_list):
        if fate_key not in available_choices:
            logg.error(
                f"The fate_key map for {fate_key} have not been computed yet. Skipped!"
            )
        else:
            fate_vector = np.array(adata.obs[f"fate_map_{source}_{fate_key}"])
            if plot_target_state:
                target_list = valid_fate_list[j]
            else:
                target_list = None

            fate_map_embedding(
                adata,
                fate_vector,
                cell_id_t1,
                sp_idx,
                mask=mask,
                target_list=target_list,
                color_bar_label=color_bar_label,
                color_bar_title="",
                figure_title=fate_key,
                background=background,
                show_histogram=show_histogram,
                auto_color_scale=auto_color_scale,
                color_bar=color_bar,
                horizontal=horizontal,
                target_transparency=target_transparency,
                color_map=color_map,
            )

            plt.tight_layout()
            data_des = adata.uns["data_des"][-1]
            if figure_index != "":
                figure_index == f"_{figure_index}"
            plt.savefig(
                os.path.join(
                    f"{settings.figure_path}",
                    f"{data_des}_{key_word}_{fate_key}{figure_index}.{settings.file_format_figs}",
                )
            )


def fate_potency(
    adata,
    source="transition_map",
    selected_times=None,
    background=True,
    show_histogram=False,
    auto_color_scale=True,
    color_bar=True,
    figure_index="",
    mask=None,
    color_map=plt.cm.Reds,
):
    """
    Plot fate potency.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    source: `str`, optional (default: 'transition_map')
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map',...}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    selected_times: `list`, optional (default: all)
        A list of time points to further restrict the cell states to plot.
        The default choice is not to constrain the cell states to show.
    background: `bool`, optional (default: True)
        If true, plot all cell states (t1+t2) in grey as the background.
    show_histogram: `bool`, optional (default: False)
        If true, show the distribution of inferred fate probability.
    color_bar: `bool`, optional (default: True)
        plot the color bar if True.
    figure_index: `str`, optional (default: '')
        String index for annotate filename for saved figures. Used to distinuigh plots from different conditions.
    """

    key_word = f"fate_potency_{source}"
    available_choices = hf.parse_output_choices(adata, key_word, where="obs")
    map_backward = adata.uns[key_word]["map_backward"]
    method = adata.uns[key_word]["method"]
    if map_backward:
        cell_id_t1 = adata.uns["Tmap_cell_id_t1"]
    else:
        cell_id_t1 = adata.uns["Tmap_cell_id_t2"]

    time_info = np.array(adata.obs["time_info"])
    sp_idx = hf.selecting_cells_by_time_points(time_info[cell_id_t1], selected_times)

    fate_vector = np.array(adata.obs[key_word])

    fate_map_embedding(
        adata,
        fate_vector,
        cell_id_t1,
        sp_idx,
        mask=mask,
        target_list=None,
        color_bar_label="Fate potency",
        color_bar_title="",
        figure_title="",
        background=background,
        show_histogram=show_histogram,
        auto_color_scale=auto_color_scale,
        color_bar=color_bar,
        color_map=color_map,
    )

    plt.tight_layout()
    data_des = adata.uns["data_des"][-1]
    if figure_index != "":
        figure_index == f"_{figure_index}"

    plt.savefig(
        os.path.join(
            f"{settings.figure_path}",
            f"{data_des}_{key_word}{figure_index}.{settings.file_format_figs}",
        )
    )


def fate_bias(
    adata,
    selected_fates=None,
    source="transition_map",
    selected_times=None,
    background=True,
    show_histogram=False,
    plot_target_state=True,
    auto_color_scale=True,
    color_bar=True,
    target_transparency=0.2,
    figure_index="",
    horizontal=False,
    mask=None,
    color_bar_title=None,
    color_map=plt.cm.bwr,
):
    """
    Plot fate bias.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    source: `str`, optional (default: 'transition_map')
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map',...}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    selected_times: `list`, optional (default: all)
        A list of time points to further restrict the cell states to plot.
        The default choice is not to constrain the cell states to show.
    background: `bool`, optional (default: True)
        If true, plot all cell states (t1+t2) in grey as the background.
    show_histogram: `bool`, optional (default: False)
        If true, show the distribution of inferred fate probability.
    color_bar: `bool`, optional (default: True)
        plot the color bar if True.
    figure_index: `str`, optional (default: '')
        String index for annotate filename for saved figures. Used to distinuigh plots from different conditions.
    """

    key_word = f"fate_bias_{source}"
    available_choices = hf.parse_output_choices(adata, key_word, where="obs")
    map_backward = adata.uns[key_word]["map_backward"]
    method = adata.uns[key_word]["method"]
    if map_backward:
        cell_id_t1 = adata.uns["Tmap_cell_id_t1"]
    else:
        cell_id_t1 = adata.uns["Tmap_cell_id_t2"]

    time_info = np.array(adata.obs["time_info"])
    sp_idx = hf.selecting_cells_by_time_points(time_info[cell_id_t1], selected_times)

    if selected_fates is None:
        fate_key = available_choices[0]
        valid_fate_list = [xx.split("_") for xx in fate_key.split("*")]
    else:
        state_info = adata.obs["state_info"]
        (
            mega_cluster_list,
            valid_fate_list,
            __,
            sel_index_list,
        ) = hf.analyze_selected_fates(state_info, selected_fates)

        if len(mega_cluster_list) != 2:
            raise ValueError("selected_fates must have only two valid fates")

        fate_key = f"{mega_cluster_list[0]}_{mega_cluster_list[1]}"

    fate_vector = np.array(adata.obs[f"fate_bias_{source}_{fate_key}"])

    if plot_target_state:
        target_list = valid_fate_list[0] + valid_fate_list[1]
    else:
        target_list = None

    if color_bar_title is None:
        color_bar_title = fate_key.split("*")[0]

    method = adata.uns[f"fate_bias_{source}"]["method"]
    if method == "norm-sum":
        color_bar_label = "Progenitor bias"
    else:
        color_bar_label = "Fate bias"

    fate_map_embedding(
        adata,
        fate_vector,
        cell_id_t1,
        sp_idx,
        mask=mask,
        color_map=color_map,
        target_list=target_list,
        color_bar_label=color_bar_label,
        color_bar_title=color_bar_title,
        figure_title="",
        background=background,
        show_histogram=show_histogram,
        auto_color_scale=auto_color_scale,
        color_bar=color_bar,
        horizontal=horizontal,
        target_transparency=target_transparency,
        histogram_scales=[0, 1],
    )

    plt.tight_layout()
    data_des = adata.uns["data_des"][-1]
    if figure_index != "":
        figure_index == f"_{figure_index}"

    plt.savefig(
        os.path.join(
            f"{settings.figure_path}",
            f"{data_des}_{key_word}_{fate_key}{figure_index}.{settings.file_format_figs}",
        )
    )
