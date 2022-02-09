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
