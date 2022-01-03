import os

import numpy as np
import pandas as pd
import scipy.sparse as ssp
import scipy.stats as stats
import statsmodels.sandbox.stats.multicomp
from ete3 import Tree
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy
from sklearn.manifold import SpectralEmbedding

from .. import help_functions as hf
from .. import logging as logg
from .. import settings
from .. import tool as tl


def darken_cmap(cmap, scale_factor):
    """
    Generate a gradient color map for plotting.
    """

    cdat = np.zeros((cmap.N, 4))
    for ii in range(cdat.shape[0]):
        curcol = cmap(ii)
        cdat[ii, 0] = curcol[0] * scale_factor
        cdat[ii, 1] = curcol[1] * scale_factor
        cdat[ii, 2] = curcol[2] * scale_factor
        cdat[ii, 3] = 1
    cmap = cmap.from_list(cmap.N, cdat)
    return cmap


def start_subplot_figure(n_subplots, n_columns=5, fig_width=14, row_height=3):
    """
    Generate a figure object with given subplot specification.
    """

    n_rows = int(np.ceil(n_subplots / float(n_columns)))
    fig = plt.figure(figsize=(fig_width, n_rows * row_height))
    return fig, n_rows, n_columns


def embedding_genes(adata, basis="X_emb", color=None, color_bar=True, **kwargs):
    """
    A test embedding method. Works better with subplots
    """
    if basis not in adata.obsm.keys():
        raise ValueError(f"basis={basis} is not among {adata.obsm.keys()}")
    if color in adata.var_names:
        vector = adata[:, color].X.A.flatten()

    elif color in adata.obs.keys():
        vector = np.array(adata.obs[color])
    else:
        raise ValueError(f"color value is not right")

    x_emb = adata.obsm[basis][:, 0]
    y_emb = adata.obsm[basis][:, 1]
    customized_embedding(
        x_emb,
        y_emb,
        vector,
        title=color,
        col_range=(0, 99.8),
        color_bar=color_bar,
        **kwargs,
    )


def embedding(
    adata, basis="X_emb", color=None, cmap=darken_cmap(plt.cm.Reds, 0.9), **kwargs
):
    """
    Scatter plot for user-specified embedding basis.
    We imported :func:`~scanpy.pl.embedding` for this purpose.
    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    basis: `str`, optional (default: 'X_emb')
        The embedding to use for the plot.
    color: `str, list of str, or None` (default: None)
        Keys for annotations of observations/cells or variables/genes,
        e.g., 'state_info', 'time_info',['Gata1','Gata2']
    """

    from scanpy.plotting._tools import embedding as sc_embedding

    sc_embedding(adata, basis=basis, color=color, cmap=cmap, **kwargs)


def customized_embedding(
    x,
    y,
    vector,
    title=None,
    ax=None,
    order_points=True,
    set_ticks=False,
    col_range=None,
    buffer_pct=0.03,
    point_size=1,
    color_map=None,
    smooth_operator=None,
    set_lim=True,
    vmax=None,
    vmin=None,
    color_bar=False,
    color_bar_label="",
    color_bar_title="",
):
    """
    Plot a vector on an embedding.
    Parameters
    ----------
    x: `np.array`
        x coordinate of the embedding
    y: `np.array`
        y coordinate of the embedding
    vector: `np.array`
        A vector to be plotted.
    color_map: {plt.cm.Reds,plt.cm.Blues,...}, (default: None)
    ax: `axis`, optional (default: None)
        An external ax object can be passed here.
    order_points: `bool`, optional (default: True)
        Order points to plot by the gene expression
    col_range: `tuple`, optional (default: None)
        The default setting is to plot the actual value of the vector.
        If col_range is set within [0,100], it will plot the percentile of the values,
        and the color_bar will show range [0,1]. This re-scaling is useful for
        visualizing gene expression.
    buffer_pct: `float`, optional (default: 0.03)
        Extra space for the plot box frame
    point_size: `int`, optional (default: 1)
        Size of the data point
    smooth_operator: `np.array`, optional (default: None)
        A smooth matrix to be applied to the subsect of gene expression matrix.
    set_lim: `bool`, optional (default: True)
        Set the plot range (x_limit, and y_limit) automatically.
    vmax: `float`, optional (default: np.nan)
        Maximum color range (saturation).
        All values above this will be set as vmax.
    vmin: `float`, optional (default: np.nan)
        The minimum color range, all values below this will be set to be vmin.
    color_bar: `bool`, optional (default, False)
        If True, plot the color bar.
    set_ticks: `bool`, optional (default, False)
        If False, remove figure ticks.

    Returns
    -------
    ax:
        The figure axis
    """

    from matplotlib.colors import Normalize as mpl_Normalize

    if color_map is None:
        color_map = darken_cmap(plt.cm.Reds, 0.9)
    if ax is None:
        fig, ax = plt.subplots()

    coldat = vector.astype(float)

    if smooth_operator is None:
        coldat = coldat.squeeze()
    else:
        coldat = np.dot(smooth_operator, coldat).squeeze()

    if order_points:
        o = np.argsort(coldat)
    else:
        o = np.arange(len(coldat))

    if vmin is None:
        if col_range is None:
            vmin = np.min(coldat)
        else:
            vmin = np.percentile(coldat, col_range[0])

    if vmax is None:
        if col_range is None:
            vmax = np.max(coldat)
        else:
            vmax = np.percentile(coldat, col_range[1])

    if vmax == vmin:
        vmax = coldat.max()

    ax.scatter(
        x[o], y[o], c=coldat[o], s=point_size, cmap=color_map, vmin=vmin, vmax=vmax
    )

    if not set_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    if set_lim == True:
        ax.set_xlim(x.min() - x.ptp() * buffer_pct, x.max() + x.ptp() * buffer_pct)
        ax.set_ylim(y.min() - y.ptp() * buffer_pct, y.max() + y.ptp() * buffer_pct)

    if title is not None:
        ax.set_title(title)

    if color_bar:

        norm = mpl_Normalize(vmin=vmin, vmax=vmax)
        Clb = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=color_map),
            ax=ax,
            label=color_bar_label,
        )
        Clb.ax.set_title(color_bar_title)
    return ax


def plot_neighbor_joining(
    output_directory,
    node_groups,
    X_history,
    merged_pairs_history,
    node_names_history,
):
    fig, axs = plt.subplots(1, len(X_history))
    for i, X in enumerate(X_history):
        vmax = 1.2 * np.max(np.triu(X, k=1))
        axs[i].imshow(X, vmax=vmax)
        ii, jj = merged_pairs_history[i]
        axs[i].scatter([jj], [ii], s=100, marker="*", c="white")

        column_groups = [node_groups[n] for n in node_names_history[i]]
        column_labels = [" + ".join([n for n in grp]) for grp in column_groups]
        axs[i].set_xticks(np.arange(X.shape[1]) + 0.2)
        axs[i].set_xticklabels(column_labels, rotation=90, ha="right")
        axs[i].set_xlim([-0.5, X.shape[1] - 0.5])
        axs[i].set_ylim([X.shape[1] - 0.5, -0.5])
        axs[i].set_yticks(np.arange(X.shape[1]) + 0.2)
        axs[i].set_yticklabels(["" for grp in column_groups], rotation=90, ha="right")
    fig.set_size_inches((16, 4))
    plt.savefig(os.path.join(output_directory, "neighbor_joint_heatmaps.pdf"))


def heatmap(
    data_matrix,
    order_map_x=True,
    order_map_y=True,
    x_ticks=None,
    y_ticks=None,
    col_range=[0, 99],
    color_bar_label="",
    log_transform=False,
    color_map=plt.cm.Reds,
    vmin=None,
    vmax=None,
    fig_width=4,
    fig_height=6,
    color_bar=True,
    x_label=None,
    y_label=None,
    pseudo_count=10 ** (-10),
):
    """
    Plot ordered heat map of non-square data_matrix matrix
    Parameters
    ----------
    data_matrix: `np.array`
        The data matrix to be plotted
    order_map_x: `bool`
        Whether to re-order the x coordinate of the matrix or not
    order_map_y: `bool`
        Whether to re-order the y coordinate of the matrix or not
    x_ticks, y_ticks: `list`
        List of variable names for x and y ticks
    color_bar_label: `str`, optional (default: 'cov')
        Color bar label
    data_des: `str`, optional (default: '')
        String to distinguish different saved objects.
    log_transform: `bool`, optional (default: False)
        If true, perform a log transform. This is needed when the data
        matrix has entries varying by several order of magnitude.
    col_range: `tuple`, optional (default: None)
        The default setting is to plot the actual value of the vector.
        If col_range is set within [0,100], it will plot the percentile of the values,
        and the color_bar will show range [0,1]. This re-scaling is useful for
        visualizing gene expression.
    """

    from matplotlib.colors import Normalize as mpl_Normalize

    x_array = np.arange(data_matrix.shape[1])
    y_array = np.arange(data_matrix.shape[0])
    if order_map_x and (data_matrix.shape[1] > 2):
        if data_matrix.shape[0] != data_matrix.shape[1]:
            X = tl.get_normalized_covariance(data_matrix + pseudo_count, method="SW")
            Z = hierarchy.ward(X)
            order_x = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(Z, X))
        else:
            Z = hierarchy.ward(data_matrix + pseudo_count)
            order_x = hierarchy.leaves_list(
                hierarchy.optimal_leaf_ordering(Z, data_matrix + pseudo_count)
            )
    else:
        order_x = x_array

    if order_map_y and (data_matrix.shape[0] > 2):
        if data_matrix.shape[0] != data_matrix.shape[1]:
            order_y = hf.get_hierch_order(data_matrix + pseudo_count)
        else:
            Z = hierarchy.ward(data_matrix + pseudo_count)
            order_y = hierarchy.leaves_list(
                hierarchy.optimal_leaf_ordering(Z, data_matrix + pseudo_count)
            )
    else:
        order_y = y_array

    if log_transform:
        new_data = np.log(data_matrix[order_y][:, order_x] + 1) / np.log(10)
        label_ = " (log10)"
    else:
        new_data = data_matrix[order_y][:, order_x]
        label_ = ""

    col_data = new_data.flatten()
    if vmin is None:
        if col_range is None:
            vmin = np.min(col_data)
        else:
            vmin = np.percentile(col_data, col_range[0])

    if vmax is None:
        if col_range is None:
            vmax = np.max(col_data)
        else:
            vmax = np.percentile(col_data, col_range[1])
            if (vmax == 0) & (np.max(col_data) >= 1):
                vmax = 1
            if (vmax == 0) & (np.max(col_data) <= 1):
                vmax = np.max(col_data)

    fig, ax = plt.subplots()
    ax.imshow(new_data, aspect="auto", cmap=color_map, vmin=vmin, vmax=vmax)

    if x_ticks is None:
        plt.xticks([])
    else:
        plt.xticks(
            x_array + 0.4,
            np.array(x_ticks)[order_x],
            rotation=70,
            ha="right",
        )

    if y_ticks is None:
        plt.yticks([])
    else:
        plt.yticks(
            y_array + 0.4,
            np.array(y_ticks)[order_y],
            ha="right",
        )

    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    if color_bar:
        norm = mpl_Normalize(vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=color_map))
        cbar.set_label(f"{color_bar_label}{label_}", rotation=270, labelpad=20)
    plt.gcf().set_size_inches((fig_width, fig_height))
    return ax


def fate_map_embedding(
    adata,
    fate_vector,
    cell_id_t1,
    sp_idx,
    mask=None,
    target_list=None,
    color_bar_label="",
    color_bar_title="",
    figure_title="",
    background=True,
    show_histogram=False,
    auto_color_scale=True,
    color_bar=True,
    horizontal=False,
    target_transparency=0.2,
    histogram_scales=None,
    color_map=None,
):
    """
    Note: sp_idx is a bool array, of the length len(cell_id_t1)
    mask: bool array  of length adata.shape[0]
    """

    fig_width = settings.fig_width
    fig_height = settings.fig_height
    point_size = settings.fig_point_size

    x_emb = adata.obsm["X_emb"][:, 0]
    y_emb = adata.obsm["X_emb"][:, 1]
    state_info = np.array(adata.obs["state_info"])

    if mask is not None:
        if len(mask) == adata.shape[0]:
            mask = mask.astype(bool)
            sp_idx = sp_idx & (mask[cell_id_t1])
        else:
            logg.error("mask length does not match adata.shape[0]. Ignored mask.")

    if np.sum(sp_idx) == 0:
        raise ValueError("No cells selected")

    if color_bar:
        fig_width = fig_width + 1

    if show_histogram:
        tot_N = 2
    else:
        tot_N = 1

    if horizontal:
        row = 1
        col = tot_N
    else:
        row = tot_N
        col = 1

    fig = plt.figure(figsize=(fig_width * col, fig_height * row))

    fate_map_temp = fate_vector[cell_id_t1][sp_idx]
    ax0 = plt.subplot(row, col, 1)
    if background:
        customized_embedding(
            x_emb,
            y_emb,
            np.zeros(len(y_emb)),
            point_size=point_size,
            ax=ax0,
            title=figure_title,
        )
    else:
        customized_embedding(
            x_emb[cell_id_t1][sp_idx],
            y_emb[cell_id_t1][sp_idx],
            np.zeros(len(y_emb[cell_id_t1][sp_idx])),
            point_size=point_size,
            ax=ax0,
            title=figure_title,
        )

    if target_list is not None:
        for zz in target_list:
            idx_2 = state_info == zz
            ax0.plot(
                x_emb[idx_2],
                y_emb[idx_2],
                ".",
                color="cyan",
                markersize=point_size * 1,
                alpha=target_transparency,
            )

    if auto_color_scale:
        vmax = None
        vmin = None
    else:
        vmax = 1
        vmin = 0

    customized_embedding(
        x_emb[cell_id_t1][sp_idx],
        y_emb[cell_id_t1][sp_idx],
        fate_map_temp,
        point_size=point_size,
        ax=ax0,
        title=figure_title,
        set_lim=False,
        vmax=vmax,
        vmin=vmin,
        color_bar=color_bar,
        color_bar_label=color_bar_label,
        color_bar_title=color_bar_title,
        color_map=color_map,
    )

    if show_histogram:
        ax = plt.subplot(row, col, 2)
        ax.hist(fate_map_temp, 50, color="#2ca02c", density=True)
        if histogram_scales is not None:
            ax.set_xlim(histogram_scales)
        ax.set_xlabel(color_bar_label)
        ax.set_ylabel("Density")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(f"Ave.: {np.mean(fate_map_temp):.2f}")


def rand_jitter(arr, std):
    stdev = std * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def jitter(
    x,
    y,
    s=10,
    c="b",
    marker="o",
    cmap=None,
    norm=None,
    vmin=None,
    vmax=None,
    alpha=None,
    linewidths=None,
    std=0.01,
    **kwargs,
):
    return plt.scatter(
        rand_jitter(x, std),
        rand_jitter(y, std),
        s=s,
        c=c,
        marker=marker,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        linewidths=linewidths,
        **kwargs,
    )
