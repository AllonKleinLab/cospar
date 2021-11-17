import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as ssp
import scipy.stats as stats
import statsmodels.sandbox.stats.multicomp
from ete3 import Tree
from matplotlib import pyplot as plt
from numpy.lib.twodim_base import tril_indices
from scipy.cluster import hierarchy

# from plotnine import *
from sklearn.manifold import SpectralEmbedding

from cospar import tool as tl
from cospar.plotting import _utils as pl_util

from .. import help_functions as hf
from .. import logging as logg
from .. import settings


def fate_coupling(
    adata,
    source="transition_map",
    color_bar=True,
    rename_fates=None,
    color_map=plt.cm.Reds,
    figure_index="",
    **kwargs,
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

    ax = pl_util.heatmap(
        X_coupling,
        x_ticks=rename_fates,
        y_ticks=rename_fates,
        order_map=True,
        color_bar_label=f"Fate coupling",
        color_bar=color_bar,
        fig_width=fig_width,
        fig_height=fig_height,
        color_map=color_map,
        **kwargs,
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

    pl_util.print_hierarchy(parent_map, rename_fates)
    if plot_history:
        pl_util.plot_neighbor_joining(
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
    time_info = np.array(adata.obs["time_info"])
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
                f"The fate map for {fate_key} have not been computed yet. Skipped!"
            )
        else:
            fate_vector = np.array(adata.obs[f"fate_map_{source}_{fate_key}"])
            params = adata.uns["fate_map_params"][f"{source}_{fate_key}"]
            map_backward = params["map_backward"]
            method = params["method"]

            if map_backward:
                cell_id_t1 = adata.uns["Tmap_cell_id_t1"]
            else:
                cell_id_t1 = adata.uns["Tmap_cell_id_t2"]
            sp_idx = hf.selecting_cells_by_time_points(
                time_info[cell_id_t1], selected_times
            )

            if method == "norm-sum":
                color_bar_label = "Progenitor prob."
            else:
                color_bar_label = "Fate prob."

            if plot_target_state:
                target_list = valid_fate_list[j]
            else:
                target_list = None

            pl_util.fate_map_embedding(
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

    key_word_0 = "fate_potency"
    key_word = f"{key_word_0}_{source}"
    available_choices = hf.parse_output_choices(adata, key_word, where="obs")
    map_backward = adata.uns[f"{key_word_0}_params"][f"{source}"]["map_backward"]
    if map_backward:
        cell_id_t1 = adata.uns["Tmap_cell_id_t1"]
    else:
        cell_id_t1 = adata.uns["Tmap_cell_id_t2"]

    time_info = np.array(adata.obs["time_info"])
    sp_idx = hf.selecting_cells_by_time_points(time_info[cell_id_t1], selected_times)

    fate_vector = np.array(adata.obs[key_word])

    pl_util.fate_map_embedding(
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
    auto_color_scale=False,
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

    key_word_0 = "fate_bias"
    key_word = f"{key_word_0}_{source}"
    available_choices = hf.parse_output_choices(adata, key_word, where="obs")

    if selected_fates is None:
        logg.info(
            "selected_fates not specified. Using the first available pre-computed fate_bias"
        )
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

        fate_key = f"{mega_cluster_list[0]}*{mega_cluster_list[1]}"

    params = adata.uns[f"{key_word_0}_params"][f"{source}_{fate_key}"]
    map_backward = params["map_backward"]
    method = params["method"]

    if map_backward:
        cell_id_t1 = adata.uns["Tmap_cell_id_t1"]
    else:
        cell_id_t1 = adata.uns["Tmap_cell_id_t2"]
    time_info = np.array(adata.obs["time_info"])
    sp_idx = hf.selecting_cells_by_time_points(time_info[cell_id_t1], selected_times)

    fate_vector = np.array(adata.obs[f"fate_bias_{source}_{fate_key}"])

    if plot_target_state:
        target_list = valid_fate_list[0] + valid_fate_list[1]
    else:
        target_list = None

    if color_bar_title is None:
        color_bar_title = fate_key.split("*")[0]

    if method == "norm-sum":
        color_bar_label = "Progenitor bias"
    else:
        color_bar_label = "Fate bias"

    pl_util.fate_map_embedding(
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


def progenitor(
    adata,
    selected_fates=None,
    source="transition_map",
    selected_times=None,
    background=True,
    plot_target_state=True,
    auto_color_scale=False,
    target_transparency=0.2,
    figure_index="",
):

    key_word_0 = "progenitor"
    key_word = f"{key_word_0}_{source}"
    available_choices = hf.parse_output_choices(adata, key_word, where="obs")

    if selected_fates is None:
        mega_cluster_list = available_choices
        valid_fate_list = [xx.split("_") for xx in mega_cluster_list]
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
            raise ValueError(f"The {key_word_0} has not been computed for {fate_key}.")

        params = adata.uns[f"{key_word_0}_params"][f"{source}_{fate_key}"]
        map_backward = params["map_backward"]

        if map_backward:
            cell_id_t1 = adata.uns["Tmap_cell_id_t1"]
        else:
            cell_id_t1 = adata.uns["Tmap_cell_id_t2"]
        time_info = np.array(adata.obs["time_info"])
        sp_idx = hf.selecting_cells_by_time_points(
            time_info[cell_id_t1], selected_times
        )

        fate_vector = np.array(adata.obs[f"{key_word_0}_{source}_{fate_key}"])

        if plot_target_state:
            target_list = valid_fate_list[j]
        else:
            target_list = None

        pl_util.fate_map_embedding(
            adata,
            fate_vector,
            cell_id_t1,
            sp_idx,
            target_list=target_list,
            color_bar_label="",
            color_bar_title="",
            figure_title="",
            background=background,
            show_histogram=False,
            auto_color_scale=auto_color_scale,
            color_bar=False,
            target_transparency=target_transparency,
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


def iterative_differentiation(
    adata,
    selected_fates=None,
    source="transition_map",
    figure_index="",
):

    key_word_0 = "iterative_diff"
    key_word = f"{key_word_0}_{source}"
    available_choices = hf.parse_output_choices(adata, key_word, where="uns")

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

    x_emb = adata.obsm["X_emb"][:, 0]
    y_emb = adata.obsm["X_emb"][:, 1]
    point_size = settings.fig_point_size
    fig_width = settings.fig_width
    fig_height = settings.fig_height
    for j, fate_key in enumerate(mega_cluster_list):
        if fate_key not in available_choices:
            raise ValueError(f"The diff. traj. has not been computed for {fate_key}.")

        prob_array = np.array(
            adata.uns[f"{key_word_0}_{source}_{fate_key}"]["diff_prob_list"]
        )
        sorted_time_info = np.array(
            adata.uns[f"{key_word_0}_{source}_{fate_key}"]["sorted_time_info"]
        )

        row = len(prob_array)
        col = 1
        fig = plt.figure(figsize=(fig_width * col, fig_height * row))
        for k, t_0 in enumerate(sorted_time_info):
            ax = plt.subplot(row, col, 1 + k)
            pl_util.customized_embedding(
                x_emb,
                y_emb,
                prob_array[k],
                ax=ax,
                point_size=point_size,
                title=f"Iter. {k};  t={t_0}",
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


def single_cell_transition(
    adata,
    selected_state_id_list,
    source="transition_map",
    map_backward=True,
    savefig=False,
    initial_point_size=3,
    color_bar=True,
):
    """
    Plot transition probability from given initial cell states.

    If `map_backward=True`, plot the probability :math:`T_{ij}` over initial states :math:`i`
    at a given later state :math:`j`. Otherwise, plot the probability :math:`T_{ij}`
    over later states :math:`j` at a fixed initial state :math:`i`.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_state_id_list: `list`
        List of cell id's. Like [0,1,2].
    source: `str`
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map',...}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    map_backward: `bool`, optional (default: True)
        If `map_backward=True`, plot the probability of source states where the current cell state comes from;
        otherwise, plot future state probability starting from given initial state.
    initial_point_size: `int`, optional (default: 3)
        Relative size of the data point for the selected cells.
    save_fig: `bool`, optional (default: False)
        If true, save figure to defined directory at settings.figure_path
    color_bar: `bool`, optional (default: True)
        Plot the color bar.
    """

    hf.check_available_map(adata)
    fig_width = settings.fig_width
    fig_height = settings.fig_height
    point_size = settings.fig_point_size
    if color_bar:
        fig_width = fig_width + 0.5

    if source not in adata.uns["available_map"]:
        logg.error(f"{source} should be among {adata.uns['available_map']}")

    else:
        state_annote = adata.obs["state_info"]
        x_emb = adata.obsm["X_emb"][:, 0]
        y_emb = adata.obsm["X_emb"][:, 1]
        data_des = adata.uns["data_des"][-1]
        figure_path = settings.figure_path

        if not map_backward:
            cell_id_t1 = adata.uns["Tmap_cell_id_t1"]
            cell_id_t2 = adata.uns["Tmap_cell_id_t2"]
            Tmap = adata.uns[source]

        else:
            cell_id_t2 = adata.uns["Tmap_cell_id_t1"]
            cell_id_t1 = adata.uns["Tmap_cell_id_t2"]
            Tmap = adata.uns[source].T

        selected_state_id_list = np.array(selected_state_id_list)
        full_id_list = np.arange(len(cell_id_t1))
        valid_idx = np.in1d(full_id_list, selected_state_id_list)
        if np.sum(valid_idx) < len(selected_state_id_list):
            logg.error(f"Valid id is a integer, ranged in (0,{len(cell_id_t1)-1}).")
            selected_state_id_list = full_id_list[valid_idx]

        if len(selected_state_id_list) == 0:
            logg.error(f"Valid id is a integer, ranged in (0,{len(cell_id_t1)-1}).")
        else:
            if ssp.issparse(Tmap):
                Tmap = Tmap.A

            row = len(selected_state_id_list)
            col = 1
            fig = plt.figure(figsize=(fig_width * col, fig_height * row))

            for j, target_cell_ID in enumerate(selected_state_id_list):
                ax0 = plt.subplot(row, col, col * j + 1)

                if target_cell_ID < Tmap.shape[0]:
                    prob_vec = np.zeros(len(x_emb))
                    prob_vec[cell_id_t2] = Tmap[target_cell_ID, :]
                    prob_vec = prob_vec / np.max(prob_vec)
                    pl_util.customized_embedding(
                        x_emb,
                        y_emb,
                        prob_vec,
                        point_size=point_size,
                        ax=ax0,
                        color_bar=color_bar,
                        color_bar_label="Probability",
                    )

                    ax0.plot(
                        x_emb[cell_id_t1][target_cell_ID],
                        y_emb[cell_id_t1][target_cell_ID],
                        "*b",
                        markersize=initial_point_size * point_size,
                    )

                    if map_backward:
                        ax0.set_title(f"ID (t2): {target_cell_ID}")
                    else:
                        ax0.set_title(f"ID (t1): {target_cell_ID}")

            # if color_bar:
            #     Clb=fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Reds), ax=ax0,label='Probability')

            plt.tight_layout()
            if savefig:
                fig.savefig(
                    f"{figure_path}/{data_des}_single_cell_transition_{source}_{map_backward}.{settings.file_format_figs}"
                )
