import os

import numpy as np
import pandas as pd
import scipy.sparse as ssp
import scipy.stats as stats
import seaborn as sns
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
    normalize=False,
    binarize=False,
    log_transform=False,
    fig_width=4,
    fig_height=6,
    figure_index="",
    plot=True,
    pseudocount=10 ** (-10),
    order_map_x=False,
    order_map_y=False,
    fate_normalize_source="X_clone",
    select_clones_with_fates: list = None,
    select_clones_without_fates: list = None,
    select_clones_mode: str = "or",
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
    normalize:
        To perform cluster-wise then clone-wise normalization per time point
    binarize: `bool`
        Binarize the coarse-grained barcode count matrix, just for the purpose of plotting.
    log_transform: `bool`, optional (default: False)
        If true, perform a log transform. This is needed when the data
        matrix has entries varying by several order of magnitude.
    fig_width: `float`, optional (default: 4)
        Figure width.
    fig_height: `float`, optional (default: 6)
        Figure height.
    plot: `bool`
        True: plot the result. False, suppress the plot.
    pseudocount: `float`
        Pseudocount for the heatmap (needed for ordering the map)
    order_map_x: `bool`
        Whether to re-order the x coordinate of the matrix or not
    order_map_y: `bool`
        Whether to re-order the y coordinate of the matrix or not
    fate_normalize_source:
        Source for cluster-wise normalization: {'X_clone','state_info'}. 'X_clone': directly row-normalize coarse_X_clone; 'state_info': compute each cluster size directly, and then normalize coarse_X_clone. The latter method is useful if we have single-cell resolution for each fate.
    select_clones_with_fates: list = None,
        Select clones that labels fates from this list.
    select_clones_without_fates: list = None,
        Exclude clones that labels fates from this list.
    select_clones_mode: str = {'or','and'}
        Logic rule for selection.

    Returns:
    --------
    The coarse-grained X_clone matrix and the selected clusters are returned at
    adata.uns['barcode_heatmap']. The coarse-grained X_clone keeps all clones and maintains their ordering.
    """

    data_des = adata.uns["data_des"][-1]
    data_des = f"{data_des}_clonal"
    figure_path = settings.figure_path

    coarse_X_clone, mega_cluster_list = tl.coarse_grain_clone_over_cell_clusters(
        adata,
        selected_times=selected_times,
        selected_fates=selected_fates,
        normalize=normalize,
        fate_normalize_source=fate_normalize_source,
        select_clones_with_fates=select_clones_with_fates,
        select_clones_without_fates=select_clones_without_fates,
        select_clones_mode=select_clones_mode,
        **kwargs,
    )

    if rename_fates is None:
        rename_fates = mega_cluster_list

    if len(rename_fates) != len(mega_cluster_list):
        logg.warn(
            "rename_fates does not have the same length as selected_fates, thus not used."
        )
        rename_fates = mega_cluster_list

    if "x_ticks" not in kwargs.keys():
        kwargs["x_ticks"] = rename_fates

    coarse_X_clone_new = pl_util.custom_hierachical_ordering(
        np.arange(coarse_X_clone.shape[0]), coarse_X_clone
    )
    adata.uns["barcode_heatmap"] = {
        "coarse_X_clone": coarse_X_clone,
        "fate_names": rename_fates,
    }
    logg.info("Data saved at adata.uns['barcode_heatmap']")
    if plot:
        if binarize:
            final_matrix = coarse_X_clone_new > 0
            color_bar_label = "Binarized barcode count"
        else:
            final_matrix = coarse_X_clone_new
            color_bar_label = "Barcode count"

        if normalize:
            color_bar_label += " (normalized)"

        clone_idx = final_matrix.sum(0) > 0
        ax = pl_util.heatmap(
            final_matrix[:, clone_idx].T + pseudocount,
            order_map_x=order_map_x,
            order_map_y=order_map_y,
            color_bar_label=color_bar_label,
            log_transform=log_transform,
            fig_width=fig_width,
            fig_height=fig_height,
            color_bar=color_bar,
            **kwargs,
        )
        plt.title(f"{np.sum(clone_idx)} clones")

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


def clonal_fates_across_time(adata, selected_times, **kwargs):
    """
    Scatter plot for clonal fate number across time point

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    selected_times: `list`, optional (default: None)
        Time points to select the cell states.

    Returns
    -------
    Results updated at adata.uns["clonal_fates_across_time"]
    """
    if len(selected_times) != 2:
        raise ValueError("selected_times must be a list with two values")
    barcode_heatmap(
        adata,
        selected_times=selected_times[0],
        color_bar=True,
        log_transform=False,
        plot=False,
    )
    clonal_fates_t1 = (adata.uns["barcode_heatmap"]["coarse_X_clone"] > 0).sum(0)
    barcode_heatmap(
        adata,
        selected_times=selected_times[1],
        color_bar=True,
        log_transform=False,
        plot=False,
    )
    clonal_fates_t2 = (adata.uns["barcode_heatmap"]["coarse_X_clone"] > 0).sum(0)

    pl_util.jitter(clonal_fates_t1, clonal_fates_t2, **kwargs)
    plt.xlabel(f"Number of fates per clone (t={selected_times[0]})")
    plt.ylabel(f"Number of fates per clone (t={selected_times[1]})")
    data_des = adata.uns["data_des"][0]
    plt.savefig(
        os.path.join(
            settings.figure_path,
            f"{data_des}_barcode_coupling_across_time.{settings.file_format_figs}",
        )
    )
    adata.uns["clonal_fates_across_time"] = {
        "clonal_fates_t1": clonal_fates_t1,
        "clonal_fates_t2": clonal_fates_t2,
    }
    logg.info("Data saved at adata.uns['clonal_fates_across_time']")


def clones_on_manifold(
    adata,
    selected_clone_list=[0],
    color_list=["red", "blue", "purple", "green", "cyan", "black"],
    selected_times=None,
    title=True,
    clone_markersize=12,
    clone_markeredgewidth=1,
    markeredgecolor="black",
    **kwargs,
):
    """
    Plot clones on top of state embedding.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    selected_clone_list: `list`
        List of selected clone ID's.
    color_list: `list`, optional (default: ['red','blue','purple','green','cyan','black'])
        The list of color that defines color at respective time points.
    selected_times: `list`, optional (default: all)
        Select time points to show corresponding states. If set to be [], use all states.
    title: `bool`, optional (default: True)
        If ture, show the clone id as panel title.
    clone_markersize: `int`, optional (default: 12)
        Clone marker size
    clone_markeredgewidth: `int`, optional (default: 1)
        Edige size for clone marker
    """

    fig_width = settings.fig_width
    fig_height = settings.fig_height
    point_size = settings.fig_point_size
    x_emb = adata.obsm["X_emb"][:, 0]
    y_emb = adata.obsm["X_emb"][:, 1]
    data_des = adata.uns["data_des"][-1]
    # data_path=settings.data_path
    figure_path = settings.figure_path
    X_clone = adata.obsm["X_clone"]
    time_info = np.array(adata.obs["time_info"])

    # use only valid time points
    sp_idx = hf.selecting_cells_by_time_points(time_info, selected_times)
    selected_times = np.sort(list(set(time_info[sp_idx])))

    selected_clone_list = np.array(selected_clone_list)
    full_id_list = np.arange(X_clone.shape[1])
    valid_idx = np.in1d(full_id_list, selected_clone_list)
    if np.sum(valid_idx) < len(selected_clone_list):
        logg.error(
            f"Valid id range is (0,{X_clone.shape[1]-1}). Please use a smaller ID!"
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
                idx_clone = X_clone[:, my_id].A.flatten() > 0
                idx = idx_t & idx_clone
                ax.plot(
                    x_emb[idx],
                    y_emb[idx],
                    ".",
                    color=color_list[j % len(color_list)],
                    markersize=clone_markersize,
                    markeredgecolor=markeredgecolor,
                    markeredgewidth=clone_markeredgewidth,
                    **kwargs,
                )

                if title:
                    ax.set_title(f"ID: {my_id}")

            fig.savefig(
                os.path.join(
                    figure_path,
                    f"{data_des}_different_clones_{my_id}.{settings.file_format_figs}",
                )
            )


def clonal_fate_bias(adata, show_histogram=True, FDR=0.05):
    """
    Plot clonal fate bias towards a cluster.

    The results should be pre-computed from :func:`cospar.tl.clonal_fate_bias`

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    show_histogram: `bool`, optional (default: True)
        If true, show the distribution of inferred fate probability.
    FDR: `float`, optional (default: 0.05)
        False-discovery rate after the Benjamini-Hochberg correction.
    """

    if "clonal_fate_bias" not in adata.uns.keys():
        raise ValueError(
            "clonal_fate_bias has not been computed. Run cs.tl.clonal_fate_bias first"
        )
    else:
        df = adata.uns["clonal_fate_bias"]
        fate_bias = df["Fate_bias"]
        target_fraction_array = df["clonal_fraction_in_target_fate"]

    fig_width = settings.fig_width
    fig_height = settings.fig_height
    data_des = adata.uns["data_des"][-1]
    figure_path = settings.figure_path
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
        os.path.join(
            figure_path, f"{data_des}_clonal_fate_bias.{settings.file_format_figs}"
        )
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
            os.path.join(
                figure_path,
                f"{data_des}_observed_clonal_fraction.{settings.file_format_figs}",
            )
        )


def clonal_reports(adata, selected_times=None, **kwargs):
    """
    Report the statistics of the clonal data.

    It includes the statistics for clone size , and the barcode number per cell.
    """

    time_info = np.array(adata.obs["time_info"])
    sp_idx = hf.selecting_cells_by_time_points(time_info, selected_times)
    adata_1 = adata[sp_idx]
    persistent_clone_ids = tl.identify_persistent_clones(adata_1)
    X_clone = adata_1.obsm["X_clone"]
    total_clone_N = X_clone.shape[1]
    print(
        f"  Clones observed across selected times: {len(persistent_clone_ids)} (out of {total_clone_N} clones)"
    )

    for x in set(adata_1.obs["time_info"]):
        print(f"---------t={x}---------")
        adata_sp = adata_1[adata_1.obs["time_info"] == x]
        X_clone = adata_sp.obsm["X_clone"]
        clone_size = X_clone.sum(0).A.flatten()
        clonal_bc_number = X_clone.sum(1).A.flatten()
        clonal_cells_N = np.sum(clonal_bc_number > 0)
        total_N = X_clone.shape[0]
        total_clone_N = X_clone.shape[1]
        useful_clone_N = np.sum(clone_size > 0)
        print(f"    Cells with barcode: {clonal_cells_N} (out of {total_N} cells)")
        print(
            f"    Barcodes with cells: {useful_clone_N} (out of {total_clone_N} clones)"
        )

        fig, axs = plt.subplots(1, 2, figsize=(8, 3.5))
        ax = sns.histplot(clone_size[clone_size > 0], ax=axs[0], **kwargs)
        ax.set_xlabel("Clone size")
        ax.set_ylabel("Count")

        ax = sns.histplot(clonal_bc_number[clonal_bc_number > 0], ax=axs[1], **kwargs)
        ax.set_xlabel("Clonal barcode number per cell")
        ax.set_ylabel("Count")
        fig.suptitle(f"Time={x}")
