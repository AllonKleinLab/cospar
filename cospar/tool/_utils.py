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

from .. import help_functions as hf
from .. import logging as logg
from .. import settings


def get_normalized_covariance(data, method="Weinreb"):
    """
    Compute the normalized correlation of the data matrix.

    For a given data matrix :math:`X_{il}`, where :math:`i` can be
    a state id or a barcode id, while :math:`l` is a id for fate cluster.
    We want to compute the coupling :math:`Y_{ll'}` between two fate clusters.

    * If method='SW': we first obtain :math:`Y_{ll'}=\sum_i X_{il}X_{il'}`.
      Then, we normalize the the coupling: :math:`Y_{ll'}\leftarrow Y_{ll'}/\sqrt{Y_{ll}Y_{l'l'}}`.

    * If method='Weinreb', we first compute the mean over variable :math:`i`, i.e., :math:`X^*_l`.
      Then, the covariance: :math:`Y_{ll'}=\sum_i (X_{il}-X^*_l)(X_{il'}-X^*_{l'})`.
      Finally, normalization by mean: :math:`Y_{ll'}\leftarrow Y_{ll'}/(X^*_lX^*_{l'})`.
      This method is developed to infer lineage coupling from clonal data
      (Weinreb & Klein, 2021, PNAS).

    Parameters
    ----------
    data: `np.array`, shape (n_obs, n_fates)
        An observation matrix for the fate distribution. The observable
        could be the number of barcodes in each fate, or the probability
        of a cell to enter a fate.
    method: `str`, optional (default: 'Weinreb')
        Method for computing the normalized covariance. Choice: {'Weinreb', 'SW'}

    Returns
    -------
    Normalized covariance matrix.
    """

    if method not in ["Weinreb", "SW", "Jaccard"]:
        logg.warn("method not among [Weinreb, SW]; set method=SW")
        method = "SW"

    if method == "Weinreb":
        cc = np.cov(data.T)
        mm = np.mean(data, axis=0) + 0.0001
        X, Y = np.meshgrid(mm, mm)
        cc = cc / X / Y
        return cc  # /np.max(cc)
    elif method == "SW":
        resol = 10 ** (-10)

        # No normalization performs better.  Not all cell states contribute equally to lineage coupling
        # Some cell states are in the progenitor regime, most ambiguous. They have a larger probability to remain in the progenitor regime, rather than differentiate.
        # Normalization would force these cells to make early choices, which could add noise to the result.
        # data=core.sparse_rowwise_multiply(data,1/(resol+np.sum(data,1)))

        X = data.T.dot(data)
        diag_temp = np.sqrt(np.diag(X))
        for j in range(len(diag_temp)):
            for k in range(len(diag_temp)):
                X[j, k] = X[j, k] / (diag_temp[j] * diag_temp[k])
        return X  # /np.max(X)

    elif method == "Jaccard":
        from scipy.spatial import distance

        data = np.array(data.T) > 0
        X = np.zeros((data.shape[0], data.shape[0]))
        for j, x in enumerate(data):
            for k, y in enumerate(data):
                X[j, k] = distance.jaccard(x, y)
        return 1 - X


def convert_to_tree(parent_map, celltype_names):
    child_map = {
        i: [] for i in set(list(parent_map.values()) + list(parent_map.keys()))
    }
    for i, j in parent_map.items():
        child_map[j].append(i)

    leaf_names = {i: n for i, n in enumerate(celltype_names)}

    def get_newick(n):
        if n in leaf_names:
            return leaf_names[n]
        else:
            return (
                "("
                + ",".join([get_newick(nn) for nn in sorted(child_map[n])[::-1]])
                + ")"
            )

    tree_string = get_newick(np.max(list(child_map.keys()))) + ";"

    t = Tree(tree_string)
    return t


def compute_fate_probability_map(
    adata,
    selected_fates=None,
    used_Tmap="transition_map",
    map_backward=True,
    method="norm-sum",
    fate_count=True,
):
    """
    Compute fate map and the relative bias compared to the expectation.

    `selected_fates` could contain a nested list of clusters. If so, we combine each sub-list
    into a mega-fate cluster and compute the fate map correspondingly.

    The relative bias is obtained by comparing the fate_prob with the
    expected_prob from the relative size of the targeted cluster. It ranges from [0,1],
    with 0.5 being the point that the fate_prob agrees with expected_prob.
    1 is extremely biased.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_fates: `list`, optional (default: all)
        List of targeted clusters, consistent with adata.obs['state_info'].
        If set to be None, use all fate clusters in adata.obs['state_info'].
    used_Tmap: `str`
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map','weinreb_transition_map','naive_transition_map',
        'OT_transition_map','HighVar_transition_map'}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    map_backward: `bool`, optional (default: True)
        If `map_backward=True`, compute for initial cell states (rows of Tmap, at t1);
        else, compute for later cell states (columns of Tmap, at t2)
    method: `str`, optional (default: 'norm-sum')
        Method to aggregate the transition probability within a cluster. Available options: {'sum','mean','max','norm-sum'},
        which computes the sum, mean, or max of transition probability within a cluster as the final fate probability.
    fate_count: `bool`, optional (default: True)
        Used to determine the method for computing the fate potential of a state.
        If ture, jus to count the number of possible fates; otherwise, use the Shannon entropy.

    Returns
    -------
    Store `fate_array`, `fate_map`, `fate_entropy` in adata.uns['fate_map'].

    fate_map: `np.array`, shape (n_cell, n_fate)
        n_fate is the number of mega cluster, equals len(selected_fates).
    mega_cluster_list: `list`, shape (n_fate)
        The list of names for the mega cluster. This is relevant when
        `selected_fates` has a nested structure.
    relative_bias: `np.array`, shape (n_cell, n_fate)
    expected_prob: `np.array`, shape (n_fate,)
    valid_fate_list: `list`, shape (n_fate)
        It is the same as selected_fates, could contain a nested list
        of fate clusters. It screens for valid fates, though.
    """

    hf.check_available_map(adata)
    if method not in ["max", "sum", "mean", "norm-sum"]:
        logg.warn(
            "method not in {'max','sum','mean','norm-sum'}; use the 'norm-sum' method"
        )
        method = "norm-sum"

    if map_backward:
        cell_id_t2 = adata.uns["Tmap_cell_id_t2"]
    else:
        cell_id_t2 = adata.uns["Tmap_cell_id_t1"]

    state_annote = adata.obs["state_info"]
    if selected_fates is None:
        selected_fates = list(set(state_annote))
    (
        mega_cluster_list,
        valid_fate_list,
        fate_array_flat,
        sel_index_list,
    ) = hf.analyze_selected_fates(state_annote, selected_fates)

    state_annote_0 = np.array(adata.obs["state_info"])
    if map_backward:
        cell_id_t1 = adata.uns["Tmap_cell_id_t1"]
        cell_id_t2 = adata.uns["Tmap_cell_id_t2"]

    else:
        cell_id_t2 = adata.uns["Tmap_cell_id_t1"]
        cell_id_t1 = adata.uns["Tmap_cell_id_t2"]

    x_emb = adata.obsm["X_emb"][:, 0]
    y_emb = adata.obsm["X_emb"][:, 1]
    data_des = adata.uns["data_des"][-1]

    state_annote_1 = state_annote_0.copy()
    for j1, new_cluster_id in enumerate(mega_cluster_list):
        idx = np.in1d(state_annote_0, valid_fate_list[j1])
        state_annote_1[idx] = new_cluster_id

    state_annote_BW = state_annote_1[cell_id_t2]

    if used_Tmap in adata.uns["available_map"]:
        used_map = adata.uns[used_Tmap]

        fate_map, fate_entropy = compute_state_potential(
            used_map,
            state_annote_BW,
            mega_cluster_list,
            fate_count=fate_count,
            map_backward=map_backward,
            method=method,
        )

    else:
        raise ValueError(f"used_Tmap should be among {adata.uns['available_map']}")

    # Note: we compute relative_bias (normalze against cluster size). This is no longer in active use
    N_macro = len(valid_fate_list)
    relative_bias = np.zeros((fate_map.shape[0], N_macro))
    expected_prob = np.zeros(N_macro)
    for jj in range(N_macro):
        for yy in valid_fate_list[jj]:
            expected_prob[jj] = expected_prob[jj] + np.sum(
                state_annote[cell_id_t2] == yy
            ) / len(cell_id_t2)

        # transformation, this is useful only when the method =='sum'
        temp_idx = fate_map[:, jj] < expected_prob[jj]
        temp_diff = fate_map[:, jj] - expected_prob[jj]
        relative_bias[temp_idx, jj] = temp_diff[temp_idx] / expected_prob[jj]
        relative_bias[~temp_idx, jj] = temp_diff[~temp_idx] / (1 - expected_prob[jj])

        relative_bias[:, jj] = (
            relative_bias[:, jj] + 1
        ) / 2  # rescale to the range [0,1]

    return (
        fate_map,
        mega_cluster_list,
        relative_bias,
        expected_prob,
        valid_fate_list,
        sel_index_list,
        fate_entropy,
    )


def mapout_trajectories(
    transition_map, state_prob_t2, threshold=0.1, cell_id_t1=[], cell_id_t2=[]
):
    """
    Map out the ancestor probability for a given later state distribution.

    We assume that transition_map is a normalized probabilistic map from
    t1-state to t2-states. Given a distribution of states at t2, we infer the initial state distribution.

    Although it is designed to map trajectories backward, one can simply
    transpose the Tmap, and swap everything related to t1 and t2, to map forward.

    Parameters
    ----------
    transition_map: `np.array` (also accept `sp.spsparse`), shape (n_t1, n_t2)
        A transition matrix that is properly normalized.
    state_prob_t2: `np.array`, shape (n_t2,)
        A continuous-valued vector that defines the probability of the final states.
    threshold: `float`, optional (default: 0.1), range ([0,1])
        We set to zero entries < threshold * max(state_prob_t1).
    cell_id_t1: `np.array` (also accept `list`)
        The id array for cell states at t1 in the full space
    cell_id_t2: `np.array` (also accept `list`)
        The id array for cell states at t2 in the full space

    Returns
    -------
    state_prob_t1_truc: `np.array`, shape (n_t1,)
        The fate probability of each t1-cell state to enter the soft
        t2-cluster as defined by state_prob_t2.
    """

    ########## We assume that the transition_map has been properly normalized.
    # if not ssp.issparse(transition_map): transition_map=ssp.csr_matrix(transition_map).copy()
    # resol=10**(-10)
    # transition_map=sparse_rowwise_multiply(transition_map,1/(resol+np.sum(transition_map,1).A.flatten()))

    if ssp.issparse(transition_map):
        transition_map = transition_map.A

    N1, N2 = transition_map.shape
    if (
        len(cell_id_t1) == 0 and N1 == N2
    ):  # cell_id_t1 and cell_id_t2 live in the same state space
        state_prob_t1 = transition_map.dot(state_prob_t2)
        state_prob_t1_idx = state_prob_t1 > threshold * np.max(state_prob_t1)
        state_prob_t1_id = np.nonzero(state_prob_t1_idx)[0]

        state_prob_t1_truc = np.zeros(len(state_prob_t1))
        state_prob_t1_truc[state_prob_t1_id] = state_prob_t1[state_prob_t1_id]
    else:
        # both cell_id_t1 and cell_id_t2 are id's in the full space
        # selected_cell_id is also in the full space
        cell_id_t1 = np.array(cell_id_t1)
        cell_id_t2 = np.array(cell_id_t2)
        state_prob_t2_subspace = state_prob_t2[cell_id_t2]

        state_prob_t1 = transition_map.dot(state_prob_t2_subspace)
        state_prob_t1_idx = state_prob_t1 > threshold * np.max(state_prob_t1)
        state_prob_t1_id = np.nonzero(state_prob_t1_idx)[0]  # id in t1 subspace
        # state_prob_t1_truc=state_prob_t1[state_prob_t1_id]
        state_prob_t1_truc = np.zeros(len(state_prob_t1))
        state_prob_t1_truc[state_prob_t1_id] = state_prob_t1[state_prob_t1_id]

    return state_prob_t1_truc


def compute_state_potential(
    transition_map,
    state_annote,
    fate_array,
    fate_count=False,
    map_backward=True,
    method="sum",
):
    """
    Compute state probability towards/from given clusters

    Before any calculation, we row-normalize the transition map.
    If map_backward=True, compute the fate map towards given
    clusters. Otherwise, compute the ancestor map, the probabilities
    of a state to originate from given clusters.

    Parameters
    ----------
    transition_map: `sp.spmatrix` (also accept `np.array`)
        Transition map of the shape: (n_t1_cells, n_t2_cells).
    state_annote: `np.array`
        Annotation for each cell state.
    fate_array: `np.array` or `list`
        List of targeted clusters, consistent with state_annote.
    fate_count: `bool`, optional (default: False)
        Relevant for compute the fate_entropy. If true, just count
        the number of possible (Prob>0) fate outcomes for each state;
        otherwise, compute the shannon entropy of fate outcome for each state
    map_backward: `bool`, optional (default: True)
        If `map_backward=True`, compute for initial cell states (rows of Tmap, at t1);
        else, for later cell states (columns of Tmap, at t2)
    method: `str`, optional (default: 'sum')
        Method to aggregate the transition probability within a cluster. Available options: {'sum','mean','max','norm-sum'},
        which computes the sum, mean, or max of transition probability within a cluster as the final fate probability.

    Returns
    -------
    fate_map: `np.array`, shape (n_cells, n_fates)
        A matrix of fate potential for each state
    fate_entropy: `np.array`, shape (n_fates,)
        A vector of fate entropy for each state
    """

    if not ssp.issparse(transition_map):
        transition_map = ssp.csr_matrix(transition_map).copy()
    resol = 10 ** (-10)
    transition_map = hf.sparse_rowwise_multiply(
        transition_map, 1 / (resol + np.sum(transition_map, 1).A.flatten())
    )
    fate_N = len(fate_array)
    N1, N2 = transition_map.shape

    # logg.info(f"Use the method={method} to compute differentiation bias")

    if map_backward:
        idx_array = np.zeros((N2, fate_N), dtype=bool)
        for k in range(fate_N):
            idx_array[:, k] = state_annote == fate_array[k]

        fate_map = np.zeros((N1, fate_N))
        fate_entropy = np.zeros(N1)

        for k in range(fate_N):
            if method == "max":
                fate_map[:, k] = np.max(
                    transition_map[:, idx_array[:, k]], 1
                ).A.flatten()
            elif method == "mean":
                fate_map[:, k] = np.mean(
                    transition_map[:, idx_array[:, k]], 1
                ).A.flatten()
            else:  # just perform summation
                fate_map[:, k] = np.sum(
                    transition_map[:, idx_array[:, k]], 1
                ).A.flatten()

        # rescale. After this, the fate map value spreads between [0,1]. Otherwise, they can be tiny.
        if (method != "sum") and (method != "norm-sum"):
            fate_map = fate_map / np.max(fate_map)
        elif method == "norm-sum":
            # perform normalization of the fate map. This works only if there are more than two fates
            if fate_N > 1:
                # logg.info('conditional method: perform column normalization')
                fate_map = hf.sparse_column_multiply(
                    fate_map, 1 / (resol + np.sum(fate_map, 0).flatten())
                ).A
                fate_map = fate_map / np.max(fate_map)

        for j in range(N1):
            ### compute the "fate-entropy" for each state
            if fate_count:
                p0 = fate_map[j, :]
                fate_entropy[j] = np.sum(p0 > 0)
            else:
                p0 = fate_map[j, :]
                p0 = p0 / (resol + np.sum(p0)) + resol
                for k in range(fate_N):
                    fate_entropy[j] = fate_entropy[j] - p0[k] * np.log(p0[k])

    ### forward map
    else:
        idx_array = np.zeros((N1, fate_N), dtype=bool)
        for k in range(fate_N):
            idx_array[:, k] = state_annote == fate_array[k]

        fate_map = np.zeros((N2, fate_N))
        fate_entropy = np.zeros(N2)

        for k in range(fate_N):
            if method == "max":
                fate_map[:, k] = np.max(
                    transition_map[idx_array[:, k], :], 0
                ).A.flatten()
            elif method == "mean":
                fate_map[:, k] = np.mean(
                    transition_map[idx_array[:, k], :], 0
                ).A.flatten()
            else:
                fate_map[:, k] = np.sum(
                    transition_map[idx_array[:, k], :], 0
                ).A.flatten()

        # rescale. After this, the fate map value spreads between [0,1]. Otherwise, they can be tiny.
        if (method != "sum") and (method != "norm-sum"):
            fate_map = fate_map / np.max(fate_map)
        elif method == "norm-sum":
            # perform normalization of the fate map. This works only if there are more than two fates
            if fate_N > 1:
                # logg.info('conditional method: perform column normalization')
                fate_map = hf.sparse_column_multiply(
                    fate_map, 1 / (resol + np.sum(fate_map, 0).flatten())
                ).A

        for j in range(N1):

            ### compute the "fate-entropy" for each state
            if fate_count:
                p0 = fate_map[j, :]
                fate_entropy[j] = np.sum(p0 > 0)
            else:
                p0 = fate_map[j, :]
                p0 = p0 / (resol + np.sum(p0)) + resol
                for k in range(fate_N):
                    fate_entropy[j] = fate_entropy[j] - p0[k] * np.log(p0[k])

    return fate_map, fate_entropy
