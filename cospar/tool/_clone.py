import os
import time
from logging import raiseExceptions

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
from tqdm import tqdm

from cospar.tool import _utils as tl_util

from .. import help_functions as hf
from .. import logging as logg
from .. import plotting as pl


def clonal_fate_bias(adata, selected_fate="", alternative="two-sided"):
    """
    Compute clonal fate bias towards a cluster.

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
    alternative: `str`, optional (default: 'two-sided')
        Defines the alternative hypothesis. The following options are
        available (default is ‘two-sided’): ‘two-sided’;
        ‘less’: one-sided; ‘greater’: one-sided

    Returns
    -------
    result: `pd.DataFrame`
    """

    if alternative not in ["two-sided", "less", "greater"]:
        logg.warn(
            "alternative not in ['two-sided','less','greater']. Use 'two-sided' instead."
        )
        alternative = "two-sided"

    state_info = adata.obs["state_info"]
    X_clone = adata.obsm["X_clone"]
    clone_N = X_clone.shape[1]
    cell_N = X_clone.shape[0]

    if type(selected_fate) == list:
        selected_fate = [selected_fate]

    if type(selected_fate) == str:
        selected_fate = [selected_fate]

    (
        mega_cluster_list,
        valid_fate_list,
        fate_array_flat,
        sel_index_list,
    ) = hf.analyze_selected_fates(state_info, selected_fate)
    if len(mega_cluster_list) == 0:
        logg.error("No cells selected. Computation aborted!")
        return None, None
    else:
        fate_name = mega_cluster_list[0]
        target_idx = sel_index_list[0]

        ## target clone
        target_ratio_array = np.zeros(clone_N)
        P_value = np.zeros(clone_N)

        for m in tqdm(range(clone_N)):
            target_cell_idx = (X_clone[:, m].sum(1).A > 0).flatten()
            target_clone_size = np.sum(target_cell_idx)

            if target_clone_size > 0:
                target_ratio = np.sum(target_idx[target_cell_idx]) / target_clone_size
                target_ratio_array[m] = target_ratio
                cell_N_in_target = np.sum(target_idx[target_cell_idx])

                remain_cell_idx = ~target_cell_idx
                remain_cell_N_in_target = np.sum(target_idx[remain_cell_idx])
                oddsratio, pvalue = stats.fisher_exact(
                    [
                        [cell_N_in_target, target_clone_size - cell_N_in_target],
                        [remain_cell_N_in_target, cell_N - remain_cell_N_in_target],
                    ],
                    alternative=alternative,
                )
                P_value[m] = pvalue

        P_value = statsmodels.sandbox.stats.multicomp.multipletests(
            P_value, alpha=0.05, method="fdr_bh"
        )[1]

        clone_size_array = X_clone.sum(0).A.flatten()
        resol = 10 ** (-20)
        sort_idx = np.argsort(P_value)
        P_value = P_value[sort_idx] + resol
        fate_bias = -np.log10(P_value)

        result = pd.DataFrame(
            {
                "Clone_ID": sort_idx,
                "Clone_size": clone_size_array[sort_idx],
                "Q_value": P_value,
                "Fate_bias": fate_bias,
                "clonal_fraction_in_target_fate": target_ratio_array[sort_idx],
            }
        )

        adata.uns["clonal_fate_bias"] = result
        logg.info("Data saved at adata.uns['clonal_fate_bias']")


def identify_persistent_clones(adata):
    time_info = np.array(adata.obs["time_info"])
    X_clone = adata.obsm["X_clone"]
    unique_time_info = list(set(time_info))
    persistent_clone_idx = np.ones(X_clone.shape[1]).astype(bool)
    for t in unique_time_info:
        persistent_clone_idx = persistent_clone_idx & (
            X_clone[time_info == t].sum(0).A.flatten() > 0
        )
    persistent_clone_ids = np.nonzero(persistent_clone_idx)[0]
    return persistent_clone_ids


def fate_biased_clones(
    adata,
    selected_fate,
    fraction_threshold=0.1,
    FDR_threshold=0.05,
    persistent_clones=False,
):
    """
    Find clones that significantly biased towards a given fate

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    selected_fate: `str`
        The targeted fate cluster, from adata.obs['state_info'].
    fraction_threshold: float
        The fraction of cells in the target fate cluster for a clone to be included.
    FDR_threshold:
        False discovery rate for identifying the fate biased clone.
    persistent_clones: bool
        True: find clones that are shared across time points.

    Returns
    -------
    valid_clone_ids:
        List of valid clone ids.
    """
    clonal_fate_bias(adata, selected_fate=selected_fate)
    result = adata.uns["clonal_fate_bias"]
    valid_clone_ids = list(
        result[
            (result.Q_value < FDR_threshold)
            & (result.clonal_fraction_in_target_fate > fraction_threshold)
        ]["Clone_ID"]
    )

    if persistent_clones:
        persistent_ids = identify_persistent_clones(adata)
        valid_clone_ids = list(set(valid_clone_ids).intersection(set(persistent_ids)))

    return valid_clone_ids


def get_normalized_coarse_X_clone(adata, selected_fates):
    """
    We first normalize per cluster, then within each time point, normalize within the clone.
    In this case, the normalized coarse_X_clone matrix sums to 1 for each clone, thereby directly
    highlighting which cell type is more preferred by a clone. Note that the cluster-cluster correlation
    will be affected by both the cluster and clone normalization.
    """
    cell_type_N = []
    for x in selected_fates:
        temp = np.sum(np.array(adata.obs["state_info"]) == x)
        cell_type_N.append(temp)

    pl.barcode_heatmap(
        adata,
        selected_fates=selected_fates,
        color_bar=True,
        log_transform=False,
        fig_height=4,
        fig_width=8,
        binarize=False,
        plot=False,
    )

    coarse_X_clone = adata.uns["barcode_heatmap"]["coarse_X_clone"]

    # normalize cluster wise
    sum_X = np.array(cell_type_N)
    norm_X_cluster = coarse_X_clone / sum_X[:, np.newaxis]

    # normalize clone wise within each time point
    selected_fates = np.array(selected_fates)
    time_info = adata.obs["time_info"]
    X_list = []
    new_fate_list = []
    for t in sorted(list(set(time_info))):
        adata_t = adata[time_info == t]
        fates_t = list(set(adata_t.obs["state_info"]).intersection(selected_fates))
        sel_idx = np.in1d(selected_fates, fates_t)
        # print(f"time {t}; sel_idx {sel_idx}")
        sum_t = norm_X_cluster[sel_idx].sum(0)
        norm_X_cluster_clone_t = (
            (norm_X_cluster[sel_idx].transpose()) / (sum_t[:, np.newaxis] + 10**-10)
        ).transpose()
        X_list.append(norm_X_cluster_clone_t)
        new_fate_list += list(selected_fates[sel_idx])
    norm_X_cluster_clone = np.vstack(X_list)
    df_X_cluster = pd.DataFrame(
        norm_X_cluster_clone,
        columns=[f"clone {j}" for j in range(norm_X_cluster_clone.shape[1])],
    )
    df_X_cluster.index = new_fate_list

    coarse_X_clone = df_X_cluster.to_numpy()
    X_clone = adata.obsm["X_clone"].A
    fate_map = np.zeros((adata.shape[0], coarse_X_clone.shape[0]))
    cell_id_list = []
    fate_list = []
    for j in np.arange(coarse_X_clone.shape[1]):
        sel_ids = np.nonzero(X_clone[:, j] > 0)[0]
        cell_id_list += list(sel_ids)
        for i in sel_ids:
            fate_list.append(list(coarse_X_clone[:, j]))

    fate_map[cell_id_list, :] = np.array(fate_list)
    sel_fates = list(df_X_cluster.index)
    for j, x in enumerate(fate_map.transpose()):
        adata.obs[f"clonal_traj_{sel_fates[j]}"] = x
    return df_X_cluster


def get_normalized_coarse_X_clone_v1(adata, selected_fates):
    """
    We first normalize per cclone within a time point, then per cluster.
    In this case, the normalized coarse_X_clone matrix sums to 1 for each cluster, thereby directly
    highlighting which cell clone is more preferred for a specific cluster. Note that the cluster-cluster
    correlation is not sensitive to the cluster normalization. So, the only useful normalization is the
    initial clone normalization.
    """

    pl.barcode_heatmap(
        adata,
        selected_fates=selected_fates,
        color_bar=True,
        log_transform=False,
        fig_height=4,
        fig_width=8,
        binarize=False,
        plot=False,
    )

    coarse_X_clone = adata.uns["barcode_heatmap"]["coarse_X_clone"]

    # normalize clone wise within each time point
    selected_fates = np.array(selected_fates)
    time_info = adata.obs["time_info"]
    X_list = []
    new_fate_list = []
    for t in sorted(list(set(time_info))):
        adata_t = adata[time_info == t]
        fates_t = list(set(adata_t.obs["state_info"]).intersection(selected_fates))
        sel_idx = np.in1d(selected_fates, fates_t)
        # print(f"time {t}; sel_idx {sel_idx}")
        sum_t = coarse_X_clone[sel_idx].sum(0)
        norm_X_clone_t = (
            (coarse_X_clone[sel_idx].transpose()) / (sum_t[:, np.newaxis] + 10**-10)
        ).transpose()
        X_list.append(norm_X_clone_t)
        new_fate_list += list(selected_fates[sel_idx])
    norm_X_clone = np.vstack(X_list)

    # normalize cluster wise
    sum_X = norm_X_clone.sum(1)
    norm_X_cluster_clone = norm_X_clone / sum_X[:, np.newaxis]

    df_X_cluster = pd.DataFrame(
        norm_X_cluster_clone,
        columns=[f"clone {j}" for j in range(norm_X_cluster_clone.shape[1])],
    )
    df_X_cluster.index = new_fate_list

    coarse_X_clone = df_X_cluster.to_numpy()
    X_clone = adata.obsm["X_clone"].A
    fate_map = np.zeros((adata.shape[0], coarse_X_clone.shape[0]))
    cell_id_list = []
    fate_list = []
    for j in np.arange(coarse_X_clone.shape[1]):
        sel_ids = np.nonzero(X_clone[:, j] > 0)[0]
        cell_id_list += list(sel_ids)
        for i in sel_ids:
            fate_list.append(list(coarse_X_clone[:, j]))

    fate_map[cell_id_list, :] = np.array(fate_list)
    sel_fates = list(df_X_cluster.index)
    for j, x in enumerate(fate_map.transpose()):
        adata.obs[f"clonal_traj_{sel_fates[j]}"] = x
    return df_X_cluster
