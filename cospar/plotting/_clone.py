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


def barcode_heatmap(
    adata,
    selected_times=None,
    selected_fates=None,
    color_bar=True,
    rename_fates=None,
    log_transform=False,
    fig_width=4,
    fig_height=6,
    figure_index="",
    **kwargs,
):
    """
    Plot barcode heatmap among different fate clusters.

    We clonal measurement at selected time points and show the
    corresponding heatmap among selected fate clusters.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    selected_times: `list`, optional (default: None)
        Time points to select the cell states.
    selected_fates: `list`, optional (default: all)
        List of fate clusters to use. If set to be [], use all.
    color_bar: `bool`, optional (default: True)
        Plot color bar.
    rename_fates: `list`, optional (default: None)
        Provide new names in substitution of names in selected_fates.
        For this to be effective, the new name list needs to have names
        in exact correspondence to those in the old list.
    log_transform: `bool`, optional (default: False)
        If true, perform a log transform. This is needed when the data
        matrix has entries varying by several order of magnitude.
    fig_width: `float`, optional (default: 4)
        Figure width.
    fig_height: `float`, optional (default: 6)
        Figure height.
    """

    time_info = np.array(adata.obs["time_info"])
    if selected_times is not None:
        if type(selected_times) is not list:
            selected_times = [selected_times]
    sp_idx = hf.selecting_cells_by_time_points(time_info, selected_times)
    clone_annot = adata[sp_idx].obsm["X_clone"]
    state_annote = adata[sp_idx].obs["state_info"]

    if np.sum(sp_idx) == 0:
        logg.error("No cells selected. Computation aborted!")
    else:
        (
            mega_cluster_list,
            __,
            __,
            sel_index_list,
        ) = hf.analyze_selected_fates(state_annote, selected_fates)
        if len(mega_cluster_list) == 0:
            logg.error("No cells selected. Computation aborted!")
        else:
            data_des = adata.uns["data_des"][-1]
            data_des = f"{data_des}_clonal"
            figure_path = settings.figure_path

            coarse_clone_annot = np.zeros(
                (len(mega_cluster_list), clone_annot.shape[1])
            )
            for j, idx in enumerate(sel_index_list):
                coarse_clone_annot[j, :] = clone_annot[idx].sum(0)

            if rename_fates is None:
                rename_fates = mega_cluster_list

            if len(rename_fates) != len(mega_cluster_list):
                logg.warn(
                    "rename_fates does not have the same length as selected_fates, thus not used."
                )
                rename_fates = mega_cluster_list

            ax = pl_util.heatmap(
                coarse_clone_annot.T,
                x_ticks=rename_fates,
                color_bar_label="Barcode count",
                log_transform=log_transform,
                fig_width=fig_width,
                fig_height=fig_height,
                color_bar=color_bar,
                **kwargs,
            )

            plt.tight_layout()
            if figure_index != "":
                figure_index == f"_{figure_index}"
            plt.savefig(
                os.path.join(
                    figure_path,
                    f"{data_des}_barcode_heatmap{figure_index}.{settings.file_format_figs}",
                )
            )
            return ax


def clones_on_manifold(
    adata,
    selected_clone_list=[0],
    clone_point_size=12,
    color_list=["red", "blue", "purple", "green", "cyan", "black"],
    selected_times=None,
    title=True,
):
    """
    Plot clones on top of state embedding.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    selected_clone_list: `list`
        List of selected clone ID's.
    clone_point_size: `int`, optional (default: 12)
        Relative size of the data point belonging to a clone,
        as compared to other background points.
    color_list: `list`, optional (default: ['red','blue','purple','green','cyan','black'])
        The list of color that defines color at respective time points.
    selected_times: `list`, optional (default: all)
        Select time points to show corresponding states. If set to be [], use all states.
    title: `bool`, optional (default: True)
        If ture, show the clone id as panel title.
    """

    fig_width = settings.fig_width
    fig_height = settings.fig_height
    point_size = settings.fig_point_size
    x_emb = adata.obsm["X_emb"][:, 0]
    y_emb = adata.obsm["X_emb"][:, 1]
    data_des = adata.uns["data_des"][-1]
    # data_path=settings.data_path
    figure_path = settings.figure_path
    clone_annot = adata.obsm["X_clone"]
    time_info = np.array(adata.obs["time_info"])

    # use only valid time points
    sp_idx = hf.selecting_cells_by_time_points(time_info, selected_times)
    selected_times = np.sort(list(set(time_info[sp_idx])))

    selected_clone_list = np.array(selected_clone_list)
    full_id_list = np.arange(clone_annot.shape[1])
    valid_idx = np.in1d(full_id_list, selected_clone_list)
    if np.sum(valid_idx) < len(selected_clone_list):
        logg.error(
            f"Valid id range is (0,{clone_annot.shape[1]-1}). Please use a smaller ID!"
        )
        selected_clone_list = full_id_list[valid_idx]

    if len(selected_clone_list) == 0:
        logg.error("No valid states selected.")
    else:
        # using all data
        for my_id in selected_clone_list:
            fig = plt.figure(figsize=(fig_width, fig_height))
            ax = plt.subplot(1, 1, 1)
            idx_t = np.zeros(len(time_info), dtype=bool)
            for j, xx in enumerate(selected_times):
                idx_t0 = time_info == selected_times[j]
                idx_t = idx_t0 | idx_t

            pl_util.customized_embedding(
                x_emb[idx_t],
                y_emb[idx_t],
                np.zeros(len(y_emb[idx_t])),
                ax=ax,
                point_size=point_size,
            )
            for j, xx in enumerate(selected_times):
                idx_t = time_info == selected_times[j]
                idx_clone = clone_annot[:, my_id].A.flatten() > 0
                idx = idx_t & idx_clone
                ax.plot(
                    x_emb[idx],
                    y_emb[idx],
                    ".",
                    color=color_list[j % len(color_list)],
                    markersize=clone_point_size * point_size,
                    markeredgecolor="white",
                    markeredgewidth=point_size,
                )

                if title:
                    ax.set_title(f"ID: {my_id}")

            fig.savefig(
                f"{figure_path}/{data_des}_different_clones_{my_id}.{settings.file_format_figs}"
            )


def clonal_fate_bias(adata, show_histogram=True, FDR=0.05):
    """
    Plot clonal fate bias towards a cluster.

    The clonal fate bias is -log(Q-value). We calculated a P-value that
    that a clone is enriched (or depleted) in a fate, using Fisher-Exact
    test (accounting for clone size). The P-value is then corrected to
    give a Q-value by Benjamini-Hochberg procedure. The alternative
    hypothesis options are: {'two-sided', 'greater', 'less'}.
    The default is 'two-sided'.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    selected_fate: `str`
        The targeted fate cluster, from adata.obs['state_info'].
    show_histogram: `bool`, optional (default: True)
        If true, show the distribution of inferred fate probability.
    FDR: `float`, optional (default: 0.05)
        False-discovery rate after the Benjamini-Hochberg correction.
    alternative: `str`, optional (default: 'two-sided')
        Defines the alternative hypothesis. The following options are
        available (default is ‘two-sided’): ‘two-sided’;
        ‘less’: one-sided; ‘greater’: one-sided

    Returns
    -------
    result: `pd.DataFrame`
    """

    df = adata.uns["clonal_fate_bias"]
    clone_size_array = df["Clone_ID"]
    Q_value = df["Q_value"]
    fate_bias = df["Fate_bias"]
    target_fraction_array = df["clonal_fraction_in_target_fate"]

    fig_width = settings.fig_width
    fig_height = settings.fig_height
    point_size = settings.fig_point_size
    state_info = adata.obs["state_info"]
    data_des = adata.uns["data_des"][-1]
    data_path = settings.data_path
    figure_path = settings.figure_path
    state_list = list(set(state_info))
    FDR_threshold = -np.log10(FDR)

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = plt.subplot(1, 1, 1)
    ax.plot(np.arange(len(fate_bias)), fate_bias, ".", color="blue", markersize=5)
    ax.plot(
        np.arange(len(fate_bias)),
        np.zeros(len(fate_bias)) + FDR_threshold,
        "-.",
        color="grey",
        markersize=5,
        label=f"FDR={FDR}",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.set_ylabel('Fate bias ($-\\log_{10}P_{value}$)')
    ax.set_ylabel("Clonal fate bias")
    ax.set_xlabel("Clonal index")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        f"{figure_path}/{data_des}_clonal_fate_bias.{settings.file_format_figs}"
    )

    if show_histogram:
        fig = plt.figure(figsize=(fig_width, fig_height))
        ax = plt.subplot(1, 1, 1)
        ax.hist(target_fraction_array, color="#2ca02c", density=True)
        ax.set_xlim([0, 1])
        ax.set_xlabel("Clonal fraction in selected fates")
        ax.set_ylabel("Density")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(f"Average: {np.mean(target_fraction_array):.2f}")
        fig.tight_layout()
        fig.savefig(
            f"{figure_path}/{data_des}_observed_clonal_fraction.{settings.file_format_figs}"
        )
