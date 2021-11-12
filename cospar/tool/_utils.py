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


def fate_hierarchy(
    adata,
    selected_fates: list = None,
    source: str = "X_clone",
    selected_times: list = None,
    method: str = "SW",
):
    """
    Build fate hierarchy or lineage trees

    Parameters
    ----------
    source:
        Which information to use for hierarchy construction: 'clone' or any of the pre-computed transition map like 'transition_map'.

    Returns
    -------
    parent_map:
        A dictionary that returns the parent node for each child node.
        As an example:  {1: 4, 3: 4, 0: 5, 4: 5, 5: 6, 2: 6}
        In this simple map, node 1 and 3 are child of both 4, and node 0 and 4 are child of both 5 etc. In neighbor joining algorithm, typically you get a binary branching tree, so each parent only has two child node. Note that the last node '6' is the founder node, and this founder node by default includes all leaf node, and are not included in the node_groups
    node_groups:
        For each node (internal or leaf node), give its composition of all leaf nodes. As an example: {0: [0], 1: [1], 2: [2], 3: [3], 4: [1, 3], 5: [0, 1, 3]}.  5 is an internal node, and it composes [0,1,3], which are all leaf nodes.
    history:
        The history of the iterative reconstruction
    """

    if not (type(selected_fates) == list and len(selected_fates) > 0):
        raise ValueError("selected_fates must be a list with more than one elements")

    fate_N = len(selected_fates)
    X_history = []
    merged_pairs_history = []
    node_names_history = []
    node_groups = {i: [i] for i in range(fate_N)}

    parent_map = {}
    selected_fates_tmp = []
    for xx in selected_fates:
        if type(xx) is not list:
            xx = [xx]
        selected_fates_tmp.append(xx)
    node_names = list(range(fate_N))
    next_node = fate_N

    counter = 0
    while len(node_names) > 2:
        counter += 1
        fate_N_tmp = len(selected_fates_tmp)
        node_names_history.append(node_names)
        fate_coupling(
            adata,
            selected_fates=selected_fates_tmp,
            source=source,
            selected_times=selected_times,
            method=method,
            silence=True,
        )
        X_coupling = adata.uns[f"fate_coupling_{source}"]["X_coupling"]
        if counter == 1:
            fate_names = adata.uns[f"fate_coupling_{source}"]["fate_names"]

        X_history.append(np.array(X_coupling))
        floor = X_coupling.min() - 100
        for i in range(X_coupling.shape[0]):
            for j in range(X_coupling.shape[1]):
                if i >= j:
                    X_coupling[i, j] = floor

        ii = np.argmax(X_coupling.max(1))
        jj = np.argmax(X_coupling.max(0))
        merged_pairs_history.append((ii, jj))
        node_groups[next_node] = (
            node_groups[node_names[ii]] + node_groups[node_names[jj]]
        )

        parent_map[node_names[ii]] = next_node
        parent_map[node_names[jj]] = next_node

        ix = np.min([ii, jj])
        node_names = [
            n for n in node_names if not n in np.array(node_names)[np.array([ii, jj])]
        ]
        new_ix = np.array([i for i in range(fate_N_tmp) if not i in [ii, jj]])

        if len(new_ix) == 0:
            break
        new_fate = selected_fates_tmp[ii] + selected_fates_tmp[jj]
        selected_fates_tmp_1 = [selected_fates_tmp[new_ix[xx]] for xx in range(ix)]
        selected_fates_tmp_1.append(new_fate)
        for xx in range(ix, fate_N_tmp - 2):
            selected_fates_tmp_1.append(selected_fates_tmp[new_ix[xx]])
        selected_fates_tmp = selected_fates_tmp_1
        node_names.insert(ix, next_node)
        next_node += 1

    for i in node_names:
        parent_map[i] = next_node

    node_mapping = {}
    for key, value in node_groups.items():
        node_mapping[key] = [fate_names[xx] for xx in value]

    history = (X_history, merged_pairs_history, node_names_history)

    adata.uns[f"fate_hierarchy_{source}"] = {
        "parent_map": parent_map,
        "node_mapping": node_mapping,
        "history": history,
        "fate_names": fate_names,
    }
    logg.info(f"Results saved as dictionary at adata.uns['fate_hierarchy_{source}']")


def fate_coupling(
    adata,
    selected_fates=None,
    source="transition_map",
    selected_times=None,
    fate_map_method="sum",
    method="SW",
    silence=False,
):
    """
    Plot fate coupling determined by the transition map.

    We use the fate map :math:`P_i(\mathcal{C}_l)` towards a set of
    fate clusters :math:`\{\mathcal{C}_l, l=0,1,2...\}` to compute the
    fate coupling :math:`Y_{ll'}`.

    * If method='SW': we first obtain :math:`Y_{ll'}=\sum_i P_i(\mathcal{C}_l)P_i(\mathcal{C}_{l'})`.
      Then, we normalize the the coupling: :math:`Y_{ll'}\leftarrow Y_{ll'}/\sqrt{Y_{ll}Y_{l'l'}}`.

    * If method='Weinreb', we calculate the normalized
      covariance as in :func:`~cospar.tl.get_normalized_covariance`

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_fates: `list`, optional (default: all fates)
        List of cluster ids consistent with adata.obs['state_info'].
        It allows a nested list, where we merge clusters within
        each sub-list into a mega-fate cluster.
    source: `str`, optional (default: 'transition_map')
        Choices: {'clone', 'transition_map',
        'intraclone_transition_map',...}. If set to be 'clone', use only the clonal
        information. If set to be any of the precomputed transition map, use the
        transition map to compute the fate coupling. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    selected_times: `list`, optional (default: all)
        A list of time points to further restrict the cell states to plot.
        The default choice is not to constrain the cell states to show.
    fate_map_method: `str`, optional (default: 'sum')
        Method to obtain the fate probability map :math:`P_i(\mathcal{C})` towards a set
        of states annotated with fate :math:`\mathcal{C}`. Available options:
        {'sum', 'norm-sum'}. See :func:`.fate_map`.
        Plot the color bar.
    method: `str`, optional (default: 'SW')
        Method to normalize the coupling matrix: {'SW', 'Weinreb'}.

    Returns
    -------
    X_coupling and the corresponding fate_names index are stored as a dictionary
    at adata.uns[f"fate_coupling_{source}"]={"X_coupling": X_coupling, 'fate_names': fate_names}
    """

    hf.check_available_map(adata)
    time_info = np.array(adata.obs["time_info"])
    choices = list(adata.uns["available_map"]) + ["X_clone"]
    if source not in choices:
        raise ValueError(f"source should be among {choices}")
    elif source == "X_clone":
        sp_idx = hf.selecting_cells_by_time_points(time_info, selected_times)
        if np.sum(sp_idx) == 0:
            raise ValueError("No cells selected. Please change selected_times")

        else:
            # aggregrate cell states
            clone_annot = adata[sp_idx].obsm["X_clone"]
            state_annote = adata[sp_idx].obs["state_info"]
            (
                mega_cluster_list,
                __,
                __,
                sel_index_list,
            ) = hf.analyze_selected_fates(state_annote, selected_fates)
            if len(mega_cluster_list) == 0:
                raise ValueError("No cells selected. Computation aborted!")

            else:
                # coarse-grain the clonal matrix
                coarse_clone_annot = np.zeros(
                    (len(mega_cluster_list), clone_annot.shape[1])
                )
                for j, idx in enumerate(sel_index_list):
                    coarse_clone_annot[j, :] = clone_annot[idx].sum(0)

                X_coupling = tl.get_normalized_covariance(
                    coarse_clone_annot.T, method=method
                )
    else:
        cell_id_t1 = adata.uns["Tmap_cell_id_t1"]
        state_annote = adata.obs["state_info"]
        sp_idx = hf.selecting_cells_by_time_points(
            time_info[cell_id_t1], selected_times
        )

        (
            fate_map,
            mega_cluster_list,
            __,
            __,
            __,
            __,
            __,
        ) = compute_fate_probability_map(
            adata,
            selected_fates=selected_fates,
            used_Tmap=source,
            map_backward=True,
            method=fate_map_method,
        )

        if (len(mega_cluster_list) == 0) or (np.sum(sp_idx) == 0):
            raise ValueError("No cells selected. Computation aborted!")

        else:
            X_coupling = tl.get_normalized_covariance(fate_map[sp_idx], method=method)

    adata.uns[f"fate_coupling_{source}"] = {
        "X_coupling": X_coupling,
        "fate_names": mega_cluster_list,
    }

    if not silence:
        logg.info(f"Results saved as dictionary at adata.uns['fate_coupling_{source}']")


def fate_map(
    adata,
    selected_fates=None,
    source="transition_map",
    map_backward=True,
    method="norm-sum",
    fate_count=False,
):
    """
    Plot transition probability to given fate/ancestor clusters.

    Given a transition map :math:`T_{ij}`, we explore build
    the fate map :math:`P_i(\mathcal{C})` towards a set of states annotated with
    fate :math:`\mathcal{C}` in the following ways.

    Step 1: Map normalization: :math:`T_{ij}\leftarrow T_{ij}/\sum_j T_{ij}`.

    Step 2: If `map_backward=False`, perform matrix transpose :math:`T_{ij} \leftarrow T_{ji}`.

    Step 3: aggregate fate probabiliteis within a given cluster :math:`\mathcal{C}`:

    * method='sum': :math:`P_i(\mathcal{C})=\sum_{j\in \mathcal{C}} T_{ij}`.
      This gives the intuitive meaning of fate probability.

    * method='norm-sum': We normalize the map from 'sum' method within a cluster, i.e.
      :math:`P_i(\mathcal{C})\leftarrow P_i(\mathcal{C})/\sum_j P_j(\mathcal{C})`.
      This gives the probability that a fate cluster :math:`\mathcal{C}` originates
      from an initial state :math:`i`.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_fates: `list`, optional (default: all fates)
        List of cluster ids consistent with adata.obs['state_info'].
        It allows a nested list, where we merge clusters within
        each sub-list into a mega-fate cluster.
    source: `str`, optional (default: 'transition_map')
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map',...}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    map_backward: `bool`, optional (default: True)
        If `map_backward=True`, show fate properties of initial cell states :math:`i`;
        otherwise, show progenitor properties of later cell states :math:`j`.
        This is used for building the fate map :math:`P_i(\mathcal{C})`. See :func:`.fate_map`.
    method: `str`, optional (default: 'norm-sum')
        Method to obtain the fate probability map :math:`P_i(\mathcal{C})` towards a set
        of states annotated with fate :math:`\mathcal{C}`. Available options:
        {'sum', 'norm-sum'}. See :func:`.fate_map`.
    fate_count: `bool`, optional (default: False)
        Used to determine the method for computing the fate potential of a state.
        If ture, just to count the number of possible fates; otherwise, use the Shannon entropy.

    Returns
    -------
    Fate map for each targeted fate cluster is updated at adata.obs[f'fate_map_{source}_{fate_name}'].
    The accompanying parameters are saved at adata.uns[f"fate_map_{source}_{fate}"]
    """

    hf.check_available_map(adata)

    if source not in adata.uns["available_map"]:
        raise ValueError(f"source should be among {adata.uns['available_map']}")

    else:
        if map_backward:
            cell_id_t1 = adata.uns["Tmap_cell_id_t1"]
        else:
            cell_id_t1 = adata.uns["Tmap_cell_id_t2"]

        if method == "norm-sum":
            color_bar_label = "Progenitor prob."
        else:
            color_bar_label = "Fate probability"

        (
            fate_map,
            mega_cluster_list,
            relative_bias,
            expected_prob,
            valid_fate_list,
            sel_index_list,
            fate_entropy,
        ) = compute_fate_probability_map(
            adata,
            selected_fates=selected_fates,
            used_Tmap=source,
            map_backward=map_backward,
            method=method,
            fate_count=fate_count,
        )

        if len(mega_cluster_list) == 0:
            logg.error("No cells selected. Computation aborted!")
        else:
            for j, fate in enumerate(mega_cluster_list):
                temp_map = np.zeros(adata.shape[0]) + np.nan
                temp_map[cell_id_t1] = fate_map[:, j]
                adata.obs[f"fate_map_{source}_{fate}"] = temp_map
            adata.uns[f"fate_map_{source}"] = {
                "map_backward": map_backward,
                "method": method,
            }

            temp_map = np.zeros(adata.shape[0]) + np.nan
            temp_map[cell_id_t1] = fate_entropy
            adata.uns[f"fate_potency"] = temp_map

        logg.info(f"Results saved at adata.obs['fate_map_{source}_XXX']")


def fate_potency(
    adata,
    selected_fates=None,
    source="transition_map",
    map_backward=True,
    method="norm-sum",
    fate_count=False,
):
    """
    It quantifies how multi-potent a cell state is.

    If fate_count=True, it just to count the number of possible fates; otherwise, use the Shannon entropy.

    It runs :func:`.fate_map` to compute the fate potency. Please see all parameter definitions there.
    """
    fate_map(
        adata,
        selected_fates=selected_fates,
        source=source,
        map_backward=map_backward,
        method=method,
        fate_count=fate_count,
    )
    adata.obs[f"fate_potency_{source}"] = adata.uns["fate_potency"]
    adata.uns[f"fate_potency_{source}"] = {
        "map_backward": map_backward,
        "method": method,
    }
    logg.info(f"Results saved at adata.obs['fate_potency_{source}']")


def fate_bias(
    adata,
    selected_fates=None,
    source="transition_map",
    map_backward=True,
    method="norm-sum",
    sum_fate_prob_thresh=0.05,
    pseudo_count=0,
):
    """
    Plot fate bias to given two fate clusters (A, B).

    Given a fate map :math:`P_i` towards two fate clusters
    :math:`\{\mathcal{A}, \mathcal{B}\}`, constructed according
    to :func:`.fate_map`, we compute the fate bias of state :math:`i` as
    :math:`[P(\mathcal{A})+c_0]/[P(\mathcal{A})+P(\mathcal{B})+2c_0]`,
    where :math:`c_0=a * \max_{i,\mathcal{C}} P_i(\mathcal{C})`
    is a re-scaled pseudocount, with :math:`a` given by pseudo_count.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_fates: `list`
        List of cluster ids consistent with adata.obs['state_info'].
        It allows a nested structure. If so, we merge clusters within
        each sub-list into a mega-fate cluster.
    source: `str`
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map',...}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    map_backward: `bool`, optional (default: True)
        If `map_backward=True`, show fate properties of initial cell states :math:`i`;
        otherwise, show progenitor properties of later cell states :math:`j`.
        This is used for building the fate map :math:`P_i(\mathcal{C})`. See :func:`.fate_map`.
    method: `str`, optional (default: 'norm-sum')
        Method to obtain the fate probability map :math:`P_i(\mathcal{C})` towards a set
        of states annotated with fate :math:`\mathcal{C}`. Available options:
        {'sum', 'norm-sum'}. See :func:`.fate_map`.
    selected_times: `list`, optional (default: all)
        A list of time points to further restrict the cell states to plot.
        The default choice is not to constrain the cell states to show.
    sum_fate_prob_thresh: `float`, optional (default: 0.05)
        The fate bias of a state is plotted only when it has a cumulative fate
        probability to the combined cluster (A+B) larger than this threshold,
        i.e., P(i->A)+P(i+>B) >  sum_fate_prob_thresh.
    mask: `np.array`, optional (default: None)
        A boolean array for available cell states. It should has the length as adata.shape[0].
        Especially useful to constrain the states to show fate bias.
    plot_target_state: `bool`, optional (default: True)
        If true, highlight the target clusters as defined in selected_fates.
    color_bar: `bool`, optional (default: True)
        plot the color bar if True.
    show_histogram: `bool`, optional (default: True)
        If true, show the distribution of inferred fate probability.
    target_transparency: `float`, optional (default: 0.2)
        It controls the transparency of the plotted target cell states,
        for visual effect. Range: [0,1].
    figure_index: `str`, optional (default: '')
        String index for annotate filename for saved figures. Used to distinuigh plots from different conditions.
    pseudo_count: `float`, optional (default: 0)
        Pseudo count to compute the fate bias. See above.
    figure_title: `str`, optional (default: No title)

    Returns
    -------
    Results updated at adata.obs[f'fate_bias_{fate_1}_{fate_2}']
    """

    state_annote = adata.obs["state_info"]
    (
        mega_cluster_list,
        __,
        __,
        sel_index_list,
    ) = hf.analyze_selected_fates(state_annote, selected_fates)

    if len(mega_cluster_list) != 2:
        raise ValueError("selected_fates must have only two valid fates")

    fate_map(
        adata,
        selected_fates=selected_fates,
        source=source,
        map_backward=map_backward,
        method=method,
    )

    if pseudo_count == 0:
        pseudo_count = 10 ** (-10)

    if map_backward:
        cell_id_t1 = adata.uns["Tmap_cell_id_t1"]
    else:
        cell_id_t1 = adata.uns["Tmap_cell_id_t2"]

    fate_vector_1 = np.array(adata.obs[f"fate_map_{source}_{mega_cluster_list[0]}"])[
        cell_id_t1
    ]
    fate_vector_2 = np.array(adata.obs[f"fate_map_{source}_{mega_cluster_list[1]}"])[
        cell_id_t1
    ]
    add_count = pseudo_count * np.max([fate_vector_1, fate_vector_2])
    fate_vector_1 = fate_vector_1 + add_count
    fate_vector_2 = fate_vector_2 + add_count

    tot_prob = fate_vector_1 + fate_vector_2
    valid_idx = tot_prob > sum_fate_prob_thresh  # default 0.05
    fate_bias_vector = fate_vector_1[valid_idx] / (tot_prob[valid_idx])

    temp_map = np.zeros(adata.shape[0]) + 0.5  # initialize to be neutral
    temp_map[cell_id_t1[valid_idx]] = fate_bias_vector
    adata.obs[
        f"fate_bias_{source}_{mega_cluster_list[0]}*{mega_cluster_list[1]}"
    ] = temp_map

    adata.uns[f"fate_bias_{source}"] = {"map_backward": map_backward, "method": method}
    logg.info(f"Results saved at adata.obs['fate_bias_{source}']")


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

    if method not in ["Weinreb", "SW"]:
        logg.warn("method not among [Weinreb, SW]; set method=SW")
        method = "SW"

    if method == "Weinreb":
        cc = np.cov(data.T)
        mm = np.mean(data, axis=0) + 0.0001
        X, Y = np.meshgrid(mm, mm)
        cc = cc / X / Y
        return cc  # /np.max(cc)
    else:
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

    if used_Tmap in adata.uns.keys():
        used_map = adata.uns[used_Tmap]

        fate_map, fate_entropy = compute_state_potential(
            used_map,
            state_annote_BW,
            mega_cluster_list,
            fate_count=fate_count,
            map_backward=map_backward,
            method=method,
        )

        adata.uns["fate_map"] = {
            "fate_array": mega_cluster_list,
            "fate_map": fate_map,
            "fate_entropy": fate_entropy,
        }

    else:
        logg.error(
            f"used_Tmap should be among adata.uns.keys(), with _transition_map as suffix"
        )
    #### finish

    N_macro = len(valid_fate_list)
    #    fate_map=np.zeros((fate_map_0.shape[0],N_macro))
    relative_bias = np.zeros((fate_map.shape[0], N_macro))
    expected_prob = np.zeros(N_macro)
    for jj in range(N_macro):
        # idx=np.in1d(fate_array_flat,valid_fate_list[jj])
        # if method=='max':
        #     fate_map[:,jj]=fate_map_0[:,idx].max(1)
        # elif method=='mean':
        #     fate_map[:,jj]=fate_map_0[:,idx].mean(1)
        # else: # use the sum method
        #     fate_map[:,jj]=fate_map_0[:,idx].sum(1)

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
