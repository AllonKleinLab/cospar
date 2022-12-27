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


def gene_expression_dynamics(
    adata,
    selected_fate,
    gene_name_list,
    traj_threshold=0.1,
    source="transition_map",
    invert_PseudoTime=False,
    mask=None,
    compute_new=True,
    gene_exp_percentile=99,
    n_neighbors=8,
    plot_raw_data=False,
    stat_smooth_method="loess",
    ggplot_font_size=11,
):
    """
    Plot gene trend along the inferred dynamic trajectory.

    The results should be pre-computed from :func:`cospar.tl.progenitor` or
    :func:`cospar.tl.iterative_differentiation`

    Using the states that belong to the trajectory, it computes the pseudo time
    for these states and shows expression dynamics of selected genes along
    this pseudo time.

    Specifically, we first construct KNN graph, compute spectral embedding,
    and take the first component as the pseudo time. To create dynamics for a
    selected gene, we re-weight the expression of this gene at each cell by its
    probability belonging to the trajectory, and rescale the expression at selected
    percentile value. Finally, we fit a curve to the data points.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_fate: `str`, or `list`
        targeted cluster of the trajectory, as consistent with adata.obs['state_info']
        When it is a list, the listed clusters are combined into a single fate cluster.
    gene_name_list: `list`
        List of genes to plot on the dynamic trajectory.
    traj_threshold: `float`, optional (default: 0.1), range: (0,1)
        Relative threshold, used to thresholding the inferred dynamic trajecotry to select states.
    invert_PseudoTime: `bool`, optional (default: False)
        If true, invert the pseudotime: 1-pseuotime. This is useful when the direction
        of pseudo time does not agree with intuition.
    mask: `np.array`, optional (default: None)
        A boolean array for further selecting cell states.
    compute_new: `bool`, optional (default: True)
        If true, compute everyting from stratch (as we save computed pseudotime)
    gene_exp_percentile: `int`, optional (default: 99)
        Plot gene expression below this percentile.
    n_neighbors: `int`, optional (default: 8)
        Number of nearest neighbors for constructing KNN graph.
    plot_raw_data: `bool`, optional (default: False)
        Plot the raw gene expression values of each cell along the pseudotime.
    stat_smooth_method: `str`, optional (default: 'loess')
        Smooth method used in the ggplot. Current available choices are:
        'auto' (Use loess if (n<1000), glm otherwise),
        'lm' (Linear Model),
        'wls' (Linear Model),
        'rlm' (Robust Linear Model),
        'glm' (Generalized linear Model),
        'gls' (Generalized Least Squares),
        'lowess' (Locally Weighted Regression (simple)),
        'loess' (Locally Weighted Regression),
        'mavg' (Moving Average),
        'gpr' (Gaussian Process Regressor)}.
    """

    if mask == None:
        final_mask = np.ones(adata.shape[0]).astype(bool)
    else:
        if mask.shape[0] == adata.shape[0]:
            final_mask = mask
        else:
            logg.error(
                "mask must be a boolean array with the same size as adata.shape[0]."
            )
            return None

    hf.check_available_map(adata)
    fig_width = settings.fig_width
    fig_height = settings.fig_height
    point_size = settings.fig_point_size

    if len(adata.uns["available_map"]) == 0:
        logg.error(f"There is no transition map available yet")

    else:

        if type(selected_fate) == str:
            selected_fate = [selected_fate]

        (
            mega_cluster_list,
            valid_fate_list,
            fate_array_flat,
            sel_index_list,
        ) = hf.analyze_selected_fates(adata.obs["state_info"], selected_fate)
        if len(mega_cluster_list) == 0:
            logg.error("No cells selected. Computation aborted!")
            return adata
        else:
            fate_name = mega_cluster_list[0]
            target_idx = sel_index_list[0]

            x_emb = adata.obsm["X_emb"][:, 0]
            y_emb = adata.obsm["X_emb"][:, 1]
            data_des = adata.uns["data_des"][-1]
            data_path = settings.data_path
            figure_path = settings.figure_path
            file_name = os.path.join(
                data_path, f"{data_des}_fate_trajectory_pseudoTime_{fate_name}.npy"
            )

            traj_name = f"diff_trajectory_{source}_{fate_name}"
            if traj_name not in adata.obs.keys():
                logg.error(
                    f"The target fate trajectory for {fate_name} with {source} have not been inferred yet.\n"
                    "Please infer the trajectory with first with cs.tl.progenitor, \n"
                    "or cs.tl.iterative_differentiation."
                )

            else:
                prob_0 = np.array(adata.obs[traj_name])

                sel_cell_idx = (prob_0 > traj_threshold * np.max(prob_0)) & final_mask
                if np.sum(sel_cell_idx) == 0:
                    raise ValueError("No cells selected!")

                sel_cell_id = np.nonzero(sel_cell_idx)[0]

                if os.path.exists(file_name) and (not compute_new):
                    logg.info("Load pre-computed pseudotime")
                    PseudoTime = np.load(file_name)
                else:

                    from sklearn import manifold

                    data_matrix = adata.obsm["X_pca"][sel_cell_idx]
                    method = SpectralEmbedding(n_components=1, n_neighbors=n_neighbors)
                    PseudoTime = method.fit_transform(data_matrix)
                    np.save(file_name, PseudoTime)
                    # logg.info("Run time:",time.time()-t)

                PseudoTime = PseudoTime - np.min(PseudoTime)
                PseudoTime = (PseudoTime / np.max(PseudoTime)).flatten()

                ## re-order the pseudoTime such that the target fate has the pseudo time 1.
                if invert_PseudoTime:
                    # target_fate_id=np.nonzero(target_idx)[0]
                    # convert_fate_id=hf.converting_id_from_fullSpace_to_subSpace(target_fate_id,sel_cell_id)[0]
                    # if np.mean(PseudoTime[convert_fate_id])<0.5: PseudoTime=1-PseudoTime
                    PseudoTime = 1 - PseudoTime

                # pdb.set_trace()
                if (
                    np.sum((PseudoTime > 0.25) & (PseudoTime < 0.75)) == 0
                ):  # the cell states do not form a contiuum. Plot raw data instead
                    logg.error(
                        "The selected cell states do not form a connected graph. Cannot form a continuum of pseudoTime. Only plot the raw data"
                    )
                    plot_raw_data = True

                ## plot the pseudotime ordering
                fig = plt.figure(figsize=(fig_width * 2, fig_height))
                ax = plt.subplot(1, 2, 1)
                pl_util.customized_embedding(
                    x_emb,
                    y_emb,
                    sel_cell_idx,
                    ax=ax,
                    title="Selected cells",
                    point_size=point_size,
                )
                ax1 = plt.subplot(1, 2, 2)
                pl_util.customized_embedding(
                    x_emb[sel_cell_idx],
                    y_emb[sel_cell_idx],
                    PseudoTime,
                    ax=ax1,
                    title="Pseudotime",
                    point_size=point_size,
                )
                # customized_embedding(x_emb[final_id],y_emb[final_id],PseudoTime,ax=ax1,title='Pseudo time')
                Clb = fig.colorbar(
                    plt.cm.ScalarMappable(cmap=plt.cm.Reds), ax=ax1, label="Pseudotime"
                )
                plt.tight_layout()
                fig.savefig(
                    os.path.join(
                        figure_path,
                        f"{data_des}_fate_trajectory_pseudoTime_{fate_name}.{settings.file_format_figs}",
                    )
                )

                temp_dict = {"PseudoTime": PseudoTime}
                for gene_name in gene_name_list:
                    yy_max = np.percentile(
                        adata.obs_vector(gene_name), gene_exp_percentile
                    )  # global blackground
                    yy = np.array(adata.obs_vector(gene_name)[sel_cell_idx])
                    rescaled_yy = (
                        yy * prob_0[sel_cell_idx] / yy_max
                    )  # rescaled by global background
                    temp_dict[gene_name] = rescaled_yy

                from plotnine import (
                    aes,
                    geom_point,
                    ggplot,
                    labs,
                    stat_smooth,
                    theme_classic,
                )

                data2 = pd.DataFrame(temp_dict)
                data2_melt = pd.melt(
                    data2, id_vars=["PseudoTime"], value_vars=gene_name_list
                )
                gplot = (
                    ggplot(
                        data=data2_melt,
                        mapping=aes(x="PseudoTime", y="value", color="variable"),
                    )
                    + (
                        geom_point()
                        if plot_raw_data
                        else stat_smooth(method=stat_smooth_method)
                    )
                    + theme_classic(base_size=ggplot_font_size)
                    + labs(
                        x="Pseudotime",
                        y="Normalized gene expression",
                        color="Gene name",
                    )
                )

                gplot.save(
                    os.path.join(
                        figure_path,
                        f"{data_des}_fate_trajectory_pseutoTime_gene_expression_{fate_name}.{settings.file_format_figs}",
                    ),
                    width=fig_width,
                    height=fig_height,
                    verbose=False,
                )
                gplot.draw()


def gene_expression_heatmap(
    adata,
    selected_genes=None,
    selected_fates=None,
    rename_fates=None,
    color_bar=True,
    method="relative",
    fig_width=6,
    fig_height=3,
    horizontal="True",
    vmin=None,
    vmax=None,
    figure_index="",
    order_map_x=True,
    order_map_y=True,
    **kwargs,
):
    """
    Plot heatmap of gene expression within given clusters.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    selected_genes: `list`, optional (default: None)
        A list of selected genes.
    selected_fates: `list`, optional (default: all)
        List of cluster ids consistent with adata.obs['state_info'].
        It allows a nested structure. If so, we merge clusters within
        each sub-list into a mega-fate cluster.
    method: `str`, optional (default: 'relative')
        Method to normalize gene expression. Options: {'relative','zscore',f'relative_{j}'}.
        'relative': given coarse-grained gene expression
        in given clusters, normalize the expression across clusters to be 1;
        'zscore': given coarse-grained gene expression in given clusters, compute its zscore.
        f'relative_{j}': use j-th row for normalization
    rename_fates: `list`, optional (default: None)
        Provide new names in substitution of names in selected_fates.
        For this to be effective, the new name list needs to have names
        in exact correspondence to those in the old list.
    color_bar: `bool`, optional (default: True)
        If true, show the color bar.
    fig_width: `int`, optional (default: 6)
        Figure width.
    fig_height: `int`, optional (default: 3)
        Figure height.
    horizontal: `bool`, optional (default: True)
        Figure orientation.
    vmin: `float`, optional (default: None)
        Minimum value to show.
    vmax: `float`, optional (default: None)
        Maximum value to show.
    figure_index:
        A string for index the figure names.
    order_map_x: `bool`
        Whether to re-order the x coordinate of the matrix or not
    order_map_y: `bool`
        Whether to re-order the y coordinate of the matrix or not

    Returns
    -------
    gene_expression_matrix: `np.array`
    """

    if (method not in ["relative", "zscore"]) and (not method.startswith("relative_")):
        logg.warn("method not in ['relative','zscore']; set it to be 'relative'")
        method = "relative"

    if method.startswith("relative_"):
        # assuming method=f'relative_{j}'
        rela_idx = int(method.split("relative_")[1])
        logg.info(f"Use {rela_idx} row for normalization")

    gene_list = selected_genes
    state_info = np.array(adata.obs["state_info"])
    (
        mega_cluster_list,
        valid_fate_list,
        fate_array_flat,
        sel_index_list,
    ) = hf.analyze_selected_fates(state_info, selected_fates)
    gene_full = np.array(adata.var_names)
    gene_list = np.array(gene_list)
    sel_idx = np.in1d(gene_full, gene_list)
    valid_sel_idx = np.in1d(gene_list, gene_full)

    if np.sum(valid_sel_idx) > 0:
        cleaned_gene_list = gene_list[valid_sel_idx]
        if np.sum(valid_sel_idx) < len(gene_list):
            invalid_gene_list = gene_list[~valid_sel_idx]
            logg.info(f"These are invalid gene names: {invalid_gene_list}")
    else:
        raise ValueError("No valid genes selected.")
    gene_expression_matrix = np.zeros((len(mega_cluster_list), len(cleaned_gene_list)))

    X = adata.X
    resol = 10 ** (-10)

    if method == "zscore":
        logg.hint("Using zscore (range: [-2,2], or [-1,1]")
        color_bar_label = "Z-score"
    else:
        logg.hint("Using relative gene expression. Range [0,1]")
        color_bar_label = "Relative expression"

    for k, temp in enumerate(cleaned_gene_list):
        temp_id = np.nonzero(gene_full == temp)[0][0]
        temp_vector = np.zeros(len(sel_index_list))
        for j, temp_idx in enumerate(sel_index_list):
            temp_vector[j] = np.mean(X[temp_idx, temp_id])

        if method == "zscore":
            z_score = stats.zscore(temp_vector)
            gene_expression_matrix[:, k] = z_score
        elif method == "relative":
            temp_vector = (temp_vector + resol) / (resol + np.sum(temp_vector))
            gene_expression_matrix[:, k] = temp_vector
        else:
            temp_vector = (temp_vector + resol) / (resol + temp_vector[rela_idx])
            gene_expression_matrix[:, k] = temp_vector

    if (rename_fates is None) or (len(rename_fates) != len(mega_cluster_list)):
        rename_fates = mega_cluster_list

    if horizontal:
        ax = pl_util.heatmap(
            gene_expression_matrix,
            x_ticks=cleaned_gene_list,
            y_ticks=rename_fates,
            log_transform=False,
            color_map=plt.cm.coolwarm,
            fig_width=fig_width,
            fig_height=fig_height,
            color_bar=color_bar,
            vmin=vmin,
            vmax=vmax,
            color_bar_label=color_bar_label,
            order_map_x=order_map_x,
            x_tick_style="italic",
            order_map_y=order_map_y,
            **kwargs,
        )
    else:
        ax = pl_util.heatmap(
            gene_expression_matrix.T,
            x_ticks=rename_fates,
            y_ticks=cleaned_gene_list,
            log_transform=False,
            color_map=plt.cm.coolwarm,
            fig_width=fig_height,
            fig_height=fig_width,
            color_bar=color_bar,
            vmin=vmin,
            vmax=vmax,
            color_bar_label=color_bar_label,
            order_map_x=order_map_x,
            y_tick_style="italic",
            order_map_y=order_map_y,
            **kwargs,
        )

    plt.tight_layout()
    if figure_index != "":
        figure_index == f"_{figure_index}"

    data_des = adata.uns["data_des"][-1]
    plt.savefig(
        os.path.join(
            settings.figure_path,
            f"{data_des}_gene_expression_matrix{figure_index}.{settings.file_format_figs}",
        )
    )
    return gene_expression_matrix


def gene_expression_on_manifold(
    adata, selected_genes, savefig=False, selected_times=None, color_bar=False
):
    """
    Plot gene expression on the state manifold.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_genes: `list` or 'str'
        List of genes to plot.
    savefig: `bool`, optional (default: False)
        Save the figure.
    selected_times: `list`, optional (default: all)
        A list of time points to further restrict the cell states to plot.
        If map_backward=True, plot initial states that are among these time points;
        otherwise, show later states that are among these time points.
    color_bar: `bool`, (default: False)
        If True, plot the color bar.
    """

    selected_genes = list(selected_genes)
    fig_width = settings.fig_width
    fig_height = settings.fig_height
    point_size = settings.fig_point_size
    if color_bar:
        fig_width = fig_width + 0.5

    if type(selected_genes) == str:
        selected_genes = [selected_genes]

    x_emb = adata.obsm["X_emb"][:, 0]
    y_emb = adata.obsm["X_emb"][:, 1]
    figure_path = settings.figure_path

    time_info = np.array(adata.obs["time_info"])
    if selected_times is not None:
        sp_idx = np.zeros(adata.shape[0], dtype=bool)
        for xx in selected_times:
            sp_id_temp = np.nonzero(time_info == xx)[0]
            sp_idx[sp_id_temp] = True
    else:
        sp_idx = np.ones(adata.shape[0], dtype=bool)

    for j, g in enumerate(selected_genes):
        vector = adata[:, g].X.A.flatten()
        fig = plt.figure(figsize=(fig_width, fig_height))
        ax = plt.subplot(1, 1, 1)
        pl_util.customized_embedding(
            x_emb[sp_idx],
            y_emb[sp_idx],
            vector[sp_idx],
            title=g,
            ax=ax,
            col_range=(0, 99.8),
            color_bar=color_bar,
            point_size=point_size,
            color_bar_label="Normalized expression",
        )
        plt.title(g, style="italic")

        plt.tight_layout()
        if savefig:
            plt.savefig(
                os.path.join(
                    figure_path,
                    f"gene_expression_{selected_genes[j]}.{settings.file_format_figs}",
                )
            )
