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


def differential_genes(
    adata,
    cell_group_A=None,
    cell_group_B=None,
    FDR_cutoff=0.05,
    sort_by="ratio",
    min_frac_expr=0.05, 
    pseudocount=1
):
    """
    Perform differential gene expression analysis and plot top DGE genes.

    We use Wilcoxon rank-sum test to calculate P values, followed by
    Benjamini-Hochberg correction.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Need to contain gene expression matrix.
    cell_group_A: `np.array`, optional (default: None)
        A boolean array of the size adata.shape[0] for defining population A.
        If not specified, we set it to be adata.obs['cell_group_A'].
    cell_group_B: `np.array`, optional (default: None)
        A boolean array of the size adata.shape[0] for defining population B.
        If not specified, we set it to be adata.obs['cell_group_A'].
    FDR_cutoff: `float`, optional (default: 0.05)
        Cut off for the corrected Pvalue of each gene. Only genes below this
        cutoff will be shown.
    sort_by: `float`, optional (default: 'ratio')
        The key to sort the differentially expressed genes. The key can be: 'ratio' or 'Qvalue'.
    min_frac_expr: `float`, optional (default: 0.05)
        Minimum expression fraction among selected states for a
        gene to be considered for DGE analysis.
    pseudocount: `int`, optional (default: 1)
        pseudo count for taking the gene expression ratio between the two groups

    Returns
    -------
    diff_gene_A: `pd.DataFrame`
        Genes differentially expressed in cell state group A, ranked
        by the ratio of mean expressions between
        the two groups, with the top being more differentially expressed.
    diff_gene_B: `pd.DataFrame`
        Genes differentially expressed in cell state group B, ranked
        by the ratio of mean expressions between
        the two groups, with the top being more differentially expressed.
    """

    diff_gene_A = []
    diff_gene_B = []

    if sort_by not in ["ratio", "Qvalue"]:
        raise ValueError(f"sort_by must be among {['ratio','Qvalue']}")

    state_info = np.array(adata.obs["state_info"])
    inputs = [cell_group_A, cell_group_B]
    selections = []
    for cell_group_X in inputs:
        if type(cell_group_X) is str:
            if cell_group_X in list(set(state_info)):
                group_idx = state_info == cell_group_X
            else:
                raise ValueError(
                    "cell_group_A (or B) should be either a cluster name among adata.obs['state_info'] or a boolean array of size adata.shape[0]."
                )
        else:
            group_idx = np.array(cell_group_X).astype("bool")

        selections.append(group_idx)

    if (np.sum(selections[0]) == 0) or (np.sum(selections[1]) == 0):
        raise ValueError("Group A or B has zero selected cell states.")

    else:

        dge = hf.get_dge_SW(adata, selections[0], selections[1],min_frac_expr=min_frac_expr,pseudocount=pseudocount)

        dge = dge.sort_values(by=sort_by, ascending=False)
        diff_gene_A_0 = dge
        diff_gene_A = diff_gene_A_0[(dge["Qvalue"] < FDR_cutoff) & (dge["ratio"] > 0)]
        diff_gene_A = diff_gene_A.reset_index()

        dge = dge.sort_values(by=sort_by, ascending=True)
        diff_gene_B_0 = dge
        diff_gene_B = diff_gene_B_0[(dge["Qvalue"] < FDR_cutoff) & (dge["ratio"] < 0)]
        diff_gene_B = diff_gene_B.reset_index()

    return diff_gene_A, diff_gene_B
