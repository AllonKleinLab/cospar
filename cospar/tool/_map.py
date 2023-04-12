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

from cospar.tool import _clone, _utils

from .. import help_functions as hf
from .. import logging as logg
from .. import settings
from ..help_functions import _docs
from ..help_functions._docs import _doc_params


def fate_hierarchy(
    adata,
    selected_fates: list = None,
    source: str = "X_clone",
    selected_times: list = None,
    method: str = "SW",
    normalize: bool=True,
    ignore_cell_number: bool = False,
):
    """
    Build fate hierarchy or lineage trees

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_fates: `list`, optional (default: all fates)
        List of cluster ids consistent with adata.obs['state_info'].
        It allows a nested list, where we merge clusters within
        each sub-list into a mega-fate cluster.
    source: `str`, optional (default: 'transition_map')
        Choices: {'X_clone', 'transition_map',
        'intraclone_transition_map',...}. If set to be 'clone', use only the clonal
        information. If set to be any of the precomputed transition map, use the
        transition map to compute the fate coupling. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    selected_times: `list`, optional (default: all)
        A list of time points to further restrict the cell states to plot.
        The default choice is not to constrain the cell states to show.
    method: `str`, optional (default: 'SW')
        Method to normalize the coupling matrix: {'SW', 'Weinreb'}.
    normalize:
        We normalize the count matrix first within each cluster, then, within each clone separately for each time point. This was set to be off in the original CoSpar publication.
    ignore_cell_number:
        Ignore the cell number of a clone within a cluster. i.e., binarize a clone's
        contribution towards a cluster. This only works when 'source=X_clone'
        This biases towards large clusters, which will host many clones (most of them just
        a few cells)

    Returns
    -------
    Results stored at adata.uns[f"fate_hierarchy_{source}"]
    """

    state_info = np.array(adata.obs["state_info"])
    mega_cluster_list, valid_fate_list, __, __ = hf.analyze_selected_fates(
        state_info, selected_fates
    )

    fate_N = len(valid_fate_list)
    X_history = []
    merged_pairs_history = []
    node_names_history = []
    node_groups = {i: [i] for i in range(fate_N)}

    parent_map = {}
    selected_fates_tmp = []
    for xx in valid_fate_list:
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
            normalize=normalize,
            silence=True,
            ignore_cell_number=ignore_cell_number,
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
    t = _utils.convert_to_tree(parent_map, fate_names)

    adata.uns[f"fate_hierarchy_{source}"] = {
        "parent_map": parent_map,
        "node_mapping": node_mapping,
        "history": history,
        "fate_names": fate_names,
        "tree": t,
    }
    logg.info(f"Results saved as dictionary at adata.uns['fate_hierarchy_{source}']")


def fate_coupling(
    adata,
    selected_fates=None,
    source="transition_map",
    selected_times=None,
    fate_map_method="sum",
    method="SW",
    normalize=True,
    silence=False,
    ignore_cell_number: bool = False,
    fate_normalize_source="X_clone",
    select_clones_with_fates: list = None,
    select_clones_without_fates: list = None,
    select_clones_mode: str = "or",
    **kwargs,
):
    """
    Compute fate coupling determined by the transition map.

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
        Choices: {'X_clone', 'transition_map',
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
    method: `str`, optional (default: 'SW')
        Method to normalize the coupling matrix: {'SW', 'Weinreb','Jaccard'}.
    normalize:
        We normalize the count matrix first within each cluster, then, within each clone separately for each time point. This was set to be off in the original CoSpar publication.
    silence:
        Suppress information printing.
    ignore_cell_number:
        Ignore the cell number of a clone within a cluster. i.e., binarize a clone's
        contribution towards a cluster. This only works when 'source=X_clone'
        This biases towards large clusters, which will host many clones (most of them just
        a few cells)
    fate_normalize_source:
        Source for cluster-wise normalization: {'X_clone','state_info'}. 'X_clone': directly row-normalize coarse_X_clone; 'state_info': compute each cluster size directly, and then normalize coarse_X_clone. The latter method is useful if we have single-cell resolution for each fate.
    select_clones_with_fates: list = None,
        Select clones that labels fates from this list.
    select_clones_without_fates: list = None,
        Exclude clones that labels fates from this list.
    select_clones_mode: str = {'or','and'}
        Logic rule for selection.

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
        coarse_X_clone, fate_names = _clone.coarse_grain_clone_over_cell_clusters(
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
        exclude_idx = coarse_X_clone.sum(1) == 0
        if np.sum(exclude_idx) > 0:
            coarse_X_clone = coarse_X_clone[~exclude_idx]
            fate_names = np.array(fate_names)
            logg.warn(
                f"{fate_names[exclude_idx]} is excluded due to lack of any clones"
            )
            fate_names = fate_names[~exclude_idx]

        if ignore_cell_number:
            coarse_X_clone = (coarse_X_clone > 0).astype(int)
        X_coupling = _utils.get_normalized_covariance(coarse_X_clone.T, method=method)
    else:
        cell_id_t1 = adata.uns["Tmap_cell_id_t1"]
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
        ) = _utils.compute_fate_probability_map(
            adata,
            selected_fates=selected_fates,
            used_Tmap=source,
            map_backward=True,
            method=fate_map_method,
        )
        fate_idx = fate_map.sum(0) > 0
        if (len(mega_cluster_list) == 0) or (np.sum(sp_idx) == 0):
            raise ValueError("No cells selected. Computation aborted!")

        else:
            X_coupling = _utils.get_normalized_covariance(
                fate_map[sp_idx][:, fate_idx], method=method
            )

        if np.sum(~fate_idx) > 0:
            logg.warn(
                f"{mega_cluster_list[~fate_idx]} are ignored due to lack of cells"
            )
        fate_names = mega_cluster_list[fate_idx]

    adata.uns[f"fate_coupling_{source}"] = {
        "X_coupling": X_coupling,
        "fate_names": fate_names,
    }

    if not silence:
        logg.info(f"Results saved as dictionary at adata.uns['fate_coupling_{source}']")


def pvalue_for_fate_coupling(
    adata_orig, selected_fates=None, max_N_simutation=1000, normalize=False
):
    """
    Compute the pvalue for the fate coupling matrix. Pvalues are stored at adata.uns['fate_coupling_X_clone']

    Returns:
        adata: a new adata that contains the pvalue and the observed X_coupling matrix
        X_coupling_random: X_coupling matrix from all permutation test
    """
    (
        mega_cluster_list,
        valid_fate_list,
        fate_array_flat,
        sel_index_list,
    ) = hf.analyze_selected_fates(adata_orig.obs["state_info"], selected_fates)
    all_sel_idx = sel_index_list.sum(0) > 0

    # only select cells with clonal barcodes and within targeted clusters, so that
    # the permutation does not affect the clone size and the number of clonal cells in each cell type
    adata = adata_orig[(adata_orig.obsm["X_clone"].A.sum(1) > 0) & all_sel_idx]
    adata.obs["time_info"] = pd.Categorical(adata.obs["time_info"].astype(str))
    ordered_index = adata.obs["time_info"].sort_values(ascending=True).index
    adata = adata[ordered_index]
    time_info_order = sorted(
        adata.obs["time_info"].unique()
    )  # this is by default in ascending order

    from tqdm import tqdm

    from cospar import logging as logg

    adata_rand = adata.copy()
    adata_rand.uns["data_des"] = ["rand"]
    X_clone = adata.obsm["X_clone"].A

    # compute the actual coupling matrix
    fate_coupling(
        adata, selected_fates=selected_fates, source="X_clone", normalize=normalize
    )

    old_verbosity = settings.verbosity
    settings.verbosity = 1
    X_coupling_true = adata.uns["fate_coupling_X_clone"]["X_coupling"]

    # compute the permuted coupling matrix. Permute separately within each time point, as the normalization is within each time point
    X_coupling_random = []
    for _ in tqdm(range(max_N_simutation)):
        X_clone_list = []
        for t0 in time_info_order:
            t0_index = np.nonzero((adata.obs["time_info"] == t0).to_numpy())[0]
            # mask=np.zeros(X_clone.shape).astype(bool)
            # mask[t0_index]=True
            X_clone_t0 = X_clone[t0_index]
            np.random.shuffle(X_clone_t0)
            # np.putmask(X_clone,mask,X_clone_t0)
            X_clone_list.append(X_clone_t0)
        X_clone = np.vstack(X_clone_list)
        # print(X_clone.sum(0))
        # print(X_clone.sum(1))
        adata_rand.obsm["X_clone"] = ssp.csr_matrix(X_clone)

        fate_coupling(
            adata_rand,
            selected_fates=selected_fates,
            source="X_clone",
            normalize=normalize,
        )  # compute the fate coupling
        X_coupling_random.append(adata_rand.uns["fate_coupling_X_clone"]["X_coupling"])

    X_coupling_random = np.array(X_coupling_random)
    fate_names = adata.uns["fate_coupling_X_clone"]["fate_names"]

    settings.verbosity = old_verbosity

    pvalue = np.zeros((len(fate_names), len(fate_names)))
    pvalue_greater = np.zeros((len(fate_names), len(fate_names)))
    pvalue_less = np.zeros((len(fate_names), len(fate_names)))
    for i in range(len(fate_names)):
        for j in range(len(fate_names)):
            greater = (
                np.sum(np.array(X_coupling_random)[:, i, j] > X_coupling_true[i, j])
                / max_N_simutation
            )
            less = (
                np.sum(np.array(X_coupling_random)[:, i, j] < X_coupling_true[i, j])
                / max_N_simutation
            )
            pvalue_greater[i, j] = greater
            pvalue_less[i, j] = less
            pvalue[i, j] = np.min([greater, less])
            if i == j:
                pvalue[i, j] = np.nan
                pvalue_greater[i, j] = np.nan
                pvalue_less[i, j] = np.nan

    adata.uns["fate_coupling_X_clone"]["pvalue"] = pvalue
    adata.uns["fate_coupling_X_clone"]["pvalue_greater"] = pvalue_greater
    adata.uns["fate_coupling_X_clone"]["pvalue_less"] = pvalue_less
    logg.warn(f"pvalue updated at adata.uns['fate_coupling_X_clone']")

    return adata, X_coupling_random


def fate_map(
    adata,
    selected_fates=None,
    source="transition_map",
    map_backward=True,
    method="norm-sum",
    fate_count=False,
    force_run=False,
):
    """
    Compute transition probability to given fate/ancestor clusters.

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
    force_run: `bool`, optional (default: False)
        Re-compute the fate map.

    Returns
    -------
    Fate map for each targeted fate cluster is updated at adata.obs[f'fate_map_{source}_{fate_name}'].
    The accompanying parameters are saved at adata.uns[f"fate_map_{source}_{fate}"]
    """

    hf.check_available_map(adata)

    if source not in adata.uns["available_map"]:
        raise ValueError(f"source should be among {adata.uns['available_map']}")

    else:

        state_annote = adata.obs["state_info"]
        (
            mega_cluster_list,
            __,
            __,
            sel_index_list,
        ) = hf.analyze_selected_fates(state_annote, selected_fates)

        key_word = f"fate_map_{source}"
        available_choices = hf.parse_output_choices(
            adata, key_word, where="obs", interrupt=False
        )

        # check if we need to recompute the map
        re_compute = True
        condi_0 = len(available_choices) > 0
        condi_1 = set(mega_cluster_list) <= set(available_choices)
        if condi_0 & condi_1:
            map_backward_all = set(
                [
                    adata.uns["fate_map_params"][f"{source}_{x}"]["map_backward"]
                    for x in mega_cluster_list
                ]
            )
            method_all = set(
                [
                    adata.uns["fate_map_params"][f"{source}_{x}"]["method"]
                    for x in mega_cluster_list
                ]
            )
            # check that the parameters are uniform and equals to input
            if len(map_backward_all) == 1 and len(method_all) == 1:
                condi_2 = map_backward == list(map_backward_all)[0]
                condi_3 = method == list(method_all)[0]
                if condi_2 and condi_3:
                    re_compute = False
        if not (re_compute or force_run):
            logg.info("Use pre-computed fate map")
        else:
            (
                fate_map,
                mega_cluster_list,
                relative_bias,
                expected_prob,
                valid_fate_list,
                sel_index_list,
                fate_entropy,
            ) = _utils.compute_fate_probability_map(
                adata,
                selected_fates=selected_fates,
                used_Tmap=source,
                map_backward=map_backward,
                method=method,
                fate_count=fate_count,
            )

            if map_backward:
                cell_id_t1 = adata.uns["Tmap_cell_id_t1"]
            else:
                cell_id_t1 = adata.uns["Tmap_cell_id_t2"]

            if "fate_map_params" not in adata.uns.keys():
                adata.uns[f"fate_map_params"] = {}

            if len(mega_cluster_list) == 0:
                logg.error("No cells selected. Computation aborted!")
            else:
                for j, fate in enumerate(mega_cluster_list):
                    temp_map = np.zeros(adata.shape[0]) + np.nan
                    temp_map[cell_id_t1] = fate_map[:, j]
                    adata.obs[f"fate_map_{source}_{fate}"] = temp_map
                    adata.uns[f"fate_map_params"][f"{source}_{fate}"] = {
                        "map_backward": map_backward,
                        "method": method,
                    }
                    logg.info(f"Results saved at adata.obs['fate_map_{source}_{fate}']")

                temp_map = np.zeros(adata.shape[0]) + np.nan
                temp_map[cell_id_t1] = fate_entropy
                adata.uns[f"fate_potency_tmp"] = temp_map


@_doc_params(
    selected_fates=_docs.selected_fates,
    all_source=_docs.all_source,
    map_source=_docs.map_source,
    map_backward=_docs.map_backward,
    selected_times=_docs.selected_times,
    fate_method=_docs.fate_method,
)
def fate_potency(
    adata,
    selected_fates=None,
    source="transition_map",
    map_backward=True,
    method="norm-sum",
    fate_count=False,
):
    """
    Quantify how multi-potent a cell state is.

    It runs :func:`.fate_map` to compute the fate potency. Please see all parameter definitions there.

    Parameters
    ----------
    {selected_fates}
    {map_source}
    {map_backward}
    {fate_method}
    fate_count:
        True: count the number of possible fates; otherwise, use the Shannon entropy.

    Returns
    -------
    Results saved at adata.obs[f"fate_potency_{{source}}"].
    """

    fate_map(
        adata,
        selected_fates=selected_fates,
        source=source,
        map_backward=map_backward,
        method=method,
        fate_count=fate_count,
        force_run=True,
    )
    adata.obs[f"fate_potency_{source}"] = adata.uns["fate_potency_tmp"]
    if "fate_potency_params" not in adata.uns.keys():
        adata.uns[f"fate_potency_params"] = {}
    adata.uns[f"fate_potency_params"][f"{source}"] = {
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
    sum_fate_prob_thresh=0,
    pseudo_count=0,
):
    """
    Compute fate bias to given two fate clusters (A, B).

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
    sum_fate_prob_thresh: `float`, optional (default: 0.05)
        The fate bias of a state is plotted only when it has a cumulative fate
        probability to the combined cluster (A+B) larger than this threshold,
        i.e., P(i->A)+P(i+>B) >  sum_fate_prob_thresh.

    Returns
    -------
    Results updated at adata.obs[f'fate_bias_{source}_{fate_1}_{fate_2}']
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
        force_run=True,
    )

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

    if pseudo_count == 0:
        pseudo_count = 10 ** (-10)

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

    if "fate_bias_params" not in adata.uns.keys():
        adata.uns[f"fate_bias_params"] = {}
    adata.uns[f"fate_bias_params"][
        f"{source}_{mega_cluster_list[0]}*{mega_cluster_list[1]}"
    ] = {"map_backward": map_backward, "method": method}

    logg.info(
        f"Results saved at adata.obs['fate_bias_{source}_{mega_cluster_list[0]}*{mega_cluster_list[1]}']"
    )


def progenitor(
    adata,
    selected_fates=None,
    source="transition_map",
    map_backward=True,
    method="norm-sum",
    bias_threshold_A=0.5,
    bias_threshold_B=0.5,
    sum_fate_prob_thresh=0,
    pseudo_count=0,
    avoid_target_states=False,
):
    """
    Identify trajectories towards/from two given clusters.

    Given fate bias :math:`Q_i` for a state :math:`i` as defined in :func:`.fate_bias`,
    the selected ancestor population satisfies:

       * :math:`P_i(\mathcal{A})+P_i(\mathcal{B})` > sum_fate_prob_thresh;

       * Ancestor population for fate :math:`\mathcal{A}` satisfies :math:`Q_i` > bias_threshold_A

       * Ancestor population for fate :math:`\mathcal{B}` satisfies :math:`Q_i` < bias_threshold_B

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_fates: `list`
        List of cluster ids consistent with adata.obs['state_info'].
        It allows a nested structure.
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
    bias_threshold_A: `float`, optional (default: 0), range: (0,1)
        The threshold for selecting ancestor population for fate A.
    bias_threshold_B: `float`, optional (default: 0), range: (0,1)
        The threshold for selecting ancestor population for fate B.
    sum_fate_prob_thresh: `float`, optional (default: 0), range: (0,1)
        Minimum cumulative probability towards joint cluster (A,B)
        to qualify for ancestor selection.
    pseudo_count: `float`, optional (default: 0)
        Pseudo count to compute the fate bias. The bias = (Pa+c0)/(Pa+Pb+2*c0),
        where c0=pseudo_count*(maximum fate probability) is a rescaled pseudo count.
    avoid_target_states:
        If True, exclude target states for computing fate bias.

    Returns
    -------
    Results saved at adata.obs[f'progenitor_{source}_{fate_name}'] and adata.obs[f'diff_trajectory_{source}_{fate_name}']"
    """

    state_info = np.array(adata.obs["state_info"])
    (
        mega_cluster_list,
        valid_fate_list,
        __,
        sel_index_list,
    ) = hf.analyze_selected_fates(state_info, selected_fates)

    if len(mega_cluster_list) != 2:
        raise ValueError("selected_fates must have only two valid fates")

    fate_bias(
        adata,
        selected_fates=selected_fates,
        source=source,
        map_backward=map_backward,
        method=method,
        sum_fate_prob_thresh=sum_fate_prob_thresh,
        pseudo_count=pseudo_count,
    )

    if map_backward:
        cell_id_t1 = adata.uns["Tmap_cell_id_t1"]
    else:
        cell_id_t1 = adata.uns["Tmap_cell_id_t2"]

    fate_key = f"{mega_cluster_list[0]}*{mega_cluster_list[1]}"
    fate_vector = np.array(adata.obs[f"fate_bias_{source}_{fate_key}"])[cell_id_t1]

    idx_for_group_A = fate_vector > (bias_threshold_A)
    idx_for_group_B = fate_vector < (bias_threshold_B)

    if avoid_target_states:
        for zz in valid_fate_list[0]:
            id_A_t1 = np.nonzero(state_info[cell_id_t1] == zz)[0]
            idx_for_group_A[id_A_t1] = False

        for zz in valid_fate_list[1]:
            id_B_t1 = np.nonzero(state_info[cell_id_t1] == zz)[0]
            idx_for_group_B[id_B_t1] = False

    group_A_idx_full = np.zeros(adata.shape[0], dtype=bool)
    group_A_idx_full[cell_id_t1] = idx_for_group_A
    group_B_idx_full = np.zeros(adata.shape[0], dtype=bool)
    group_B_idx_full[cell_id_t1] = idx_for_group_B

    # store the trajectory
    key_word = "progenitor"
    if f"{key_word}_params" not in adata.uns.keys():
        adata.uns[f"{key_word}_params"] = {}

    temp_list = [group_A_idx_full, group_B_idx_full]
    for j, fate_name in enumerate(mega_cluster_list):
        adata.obs[f"{key_word}_{source}_{fate_name}"] = temp_list[j]
        selected_idx = sel_index_list[j]
        combined_prob_temp = temp_list[j].astype(int) + selected_idx.astype(int)
        adata.obs[f"diff_trajectory_{source}_{fate_name}"] = combined_prob_temp
        adata.obs[f"progenitor_{source}_{fate_name}"] = temp_list[j].astype(int)
        adata.uns[f"{key_word}_params"][f"{source}_{fate_name}"] = {
            "map_backward": map_backward,
            "method": method,
        }
        logg.info(
            f"Results saved at adata.obs[f'progenitor_{source}_{fate_name}'] and adata.obs[f'diff_trajectory_{source}_{fate_name}']"
        )


def iterative_differentiation(
    adata,
    selected_fates=None,
    source="transition_map",
    map_backward=True,
    map_threshold=0.1,
    apply_time_constaint=False,
):
    """
    Infer trajectory towards/from a cluster

    If map_backward=True, infer the trajectory backward in time.
    Using inferred transition map, the inference is applied recursively. It
    starts with the cell states for the selected fate and uses the selected
    map to infer the immediate ancestor states. Then, using these putative
    ancestor states as the secondary input, it finds their immediate ancestors
    again. This goes on until all time points are exhausted.

    It only works for transition map from multi-time clones.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_fate: `str`, or `list`
        Targeted cluster of the trajectory, as consistent with adata.obs['state_info']
        When it is a list, the listed clusters are combined into a single fate cluster.
    used_Tmap: `str`
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map',...}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    map_backward: `bool`, optional (default: True)
        If `map_backward=True`, show fate properties of initial cell states :math:`i`;
        otherwise, show progenitor properties of later cell states :math:`j`.
        This is used for building the fate map :math:`P_i(\mathcal{C})`. See :func:`.fate_map`.
    map_threshold: `float`, optional (default: 0.1)
        Relative threshold in the range [0,1] for truncating the fate map
        towards the cluster. Only states above the threshold will be selected.
    apply_time_constaint: `bool`, optional (default: False)
        If true, in each iteration of finding the immediate ancestor states, select cell states
        at the corresponding time point.

    Returns
    -------
    adata.obs[f'traj_{source}_{fate_name}']: `np.array`
        The probability of each state to belong to a trajectory.
    """

    # We always use the probabilistic map, which is more realiable. Otherwise, the result is very sensitive to thresholding
    hf.check_available_map(adata)
    if source not in adata.uns["available_map"]:
        raise ValueError(f"source should be among {adata.uns['available_map']}")
    else:
        # we normalize the map in advance to avoid normalization later in mapout_trajectories
        used_map = adata.uns[source]
        resol = 10 ** (-10)
        used_map = hf.sparse_rowwise_multiply(
            used_map, 1 / (resol + np.sum(used_map, 1).A.flatten())
        )

        if not map_backward:
            used_map = used_map.T

        state_annote = adata.obs["state_info"]
        (
            mega_cluster_list,
            __,
            __,
            sel_index_list,
        ) = hf.analyze_selected_fates(state_annote, selected_fates)

        if len(mega_cluster_list) == 0:
            logg.error("No cells selected. Computation aborted!")
        else:
            for k0, fate_key in enumerate(mega_cluster_list):
                selected_idx = sel_index_list[k0]

                time_info = np.array(adata.obs["time_info"])
                if map_backward:
                    cell_id_t1 = adata.uns["Tmap_cell_id_t1"]
                    cell_id_t2 = adata.uns["Tmap_cell_id_t2"]
                    sort_time_info = np.sort(list(set(time_info)))[::-1]
                else:
                    cell_id_t2 = adata.uns["Tmap_cell_id_t1"]
                    cell_id_t1 = adata.uns["Tmap_cell_id_t2"]
                    sort_time_info = np.sort(list(set(time_info)))

                if apply_time_constaint:
                    selected_idx = selected_idx * (time_info == sort_time_info[0])
                prob_array = [selected_idx]

                # used_map=hf.sparse_column_multiply(used_map,1/(resol+used_map.sum(0)))
                for j, t_0 in enumerate(sort_time_info[1:]):
                    prob_1r_full = np.zeros(adata.shape[0])
                    prob_1r_full[cell_id_t1] = _utils.mapout_trajectories(
                        used_map,
                        prob_array[j],
                        threshold=map_threshold,
                        cell_id_t1=cell_id_t1,
                        cell_id_t2=cell_id_t2,
                    )

                    ## thresholding the result
                    prob_1r_full = prob_1r_full * (
                        prob_1r_full > map_threshold * np.max(prob_1r_full)
                    )

                    if apply_time_constaint:
                        prob_1r_full = prob_1r_full * (time_info == t_0)

                    # rescale
                    prob_1r_full = prob_1r_full / (np.max(prob_1r_full) + resol)

                    prob_array.append(prob_1r_full)

                cumu_prob = np.array(prob_array).sum(0)
                key_word = f"iterative_diff_{source}_{fate_key}"
                adata.uns[key_word] = {
                    "diff_prob_list": prob_array,
                    "sorted_time_info": sort_time_info,
                    "map_backward": map_backward,
                }
                adata.obs[f"diff_trajectory_{source}_{fate_key}"] = cumu_prob
                logg.info(
                    f"Results saved at adata.obs[f'diff_trajectory_{source}_{fate_key}']"
                )
