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

from cospar.tool import _utils as tl_util

from .. import help_functions as hf
from .. import logging as logg
from .. import settings


def clonal_fate_bias(adata, selected_fate="", alternative="two-sided"):
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

    if alternative not in ["two-sided", "less", "greater"]:
        logg.warn(
            "alternative not in ['two-sided','less','greater']. Use 'two-sided' instead."
        )
        alternative = "two-sided"

    state_info = adata.obs["state_info"]
    X_clone = adata.obsm["X_clone"]
    clone_N = X_clone.shape[1]
    cell_N = X_clone.shape[0]

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

        for m in range(clone_N):
            if m % 50 == 0:
                logg.info(f"Current clone id: {m}")
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
