import os
import time
from logging import raiseExceptions

import numpy as np
import pandas as pd
import scipy.sparse as ssp
import scipy.stats as stats
import seaborn as sns
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


def get_coarse_grained_X_clone_for_clone_assignment(adata, cluster_key="leiden"):
    """
    This is used in the context of clonal data demultiplexing, like in the multi-seq protocol, where
    we each cell is labeled by one (or several) hash tag, and the barcoding process is noisy. We can look
    at the clusters at the barcode space, and we can assume that, if a cluster is clearly distinct, we
    simply assume that it belongs to a unique clone.
    """
    cell_type_N = []
    selected_fates = list(set(adata.obs[cluster_key]))
    for x in selected_fates:
        temp = np.sum(np.array(adata.obs[cluster_key]) == x)
        cell_type_N.append(temp)

    df_data = pd.DataFrame(adata.obsm["X_clone"].A)
    df_data["cluster"] = list(adata.obs[cluster_key])
    df_X_cluster_0 = df_data.groupby("cluster").sum()

    coarse_X_clone = df_X_cluster_0.to_numpy()

    # normalize cluster wise
    sum_X = 10 ** (-10) + np.array(
        cell_type_N
    )  # we use the cells with/without clonal information
    norm_X_cluster = coarse_X_clone / sum_X[:, np.newaxis]

    # # normalize clone wise
    # sum_X_1 = 10**(-10)+norm_X_cluster.sum(0)
    # norm_X_cluster_clone=norm_X_cluster/sum_X_1[np.newaxis,:]

    # if normalize_per_cluster:
    #     # Finally, normalize cluster wise, as we are interested where each cluster comes from
    #     # The risk here is that if a cluster is uniquely labled by only one clone, but the number
    #     # of clonal cells there is very small, it could lead to errors.
    print("only normlaize per cluster")
    sum_X_2 = 10 ** (-10) + norm_X_cluster.sum(1)
    norm_X_cluster_clone = norm_X_cluster / sum_X_2[:, np.newaxis]

    df_X_cluster = pd.DataFrame(
        norm_X_cluster_clone,
        columns=[f"clone_{j}" for j in range(norm_X_cluster_clone.shape[1])],
    )
    df_X_cluster.index = df_X_cluster_0.index

    return df_X_cluster


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
        columns=[f"clone_{j}" for j in range(norm_X_cluster_clone.shape[1])],
    )
    df_X_cluster.index = new_fate_list

    return df_X_cluster


def clonal_trajectory(adata, selected_fates):
    """
    Assign each cell a fate choice that belongs to its clone, and the
    clonal fate choice is determined by the coarse-grained and normalized
    barcode count matrix according to get_normalized_coarse_X_clone.
    The output is stored at adata.obs[f"clonal_traj_{sel_fates[j]}"]
    """

    df_X_cluster = get_normalized_coarse_X_clone(adata, selected_fates)
    coarse_X_clone = df_X_cluster.to_numpy()
    X_clone = adata.obsm["X_clone"].A
    fate_map = np.zeros((adata.shape[0], coarse_X_clone.shape[0]))
    cell_id_list = []
    fate_list = []
    for j in np.arange(coarse_X_clone.shape[1]):
        sel_ids = np.nonzero(X_clone[:, j] > 0)[0]
        cell_id_list += list(sel_ids)
        for i in sel_ids:
            fate_list.append(
                list(coarse_X_clone[:, j])
            )  # fate choice of corresponding cells in the selected cell list

    fate_map[cell_id_list, :] = np.array(fate_list)
    sel_fates = list(df_X_cluster.index)
    for j, x in enumerate(fate_map.transpose()):
        adata.obs[f"clonal_traj_{sel_fates[j]}"] = x
        logg.info(f'Results saved at adata.obs[f"clonal_traj_{sel_fates[j]}"]')


def get_normalized_coarse_X_clone_v1(adata, selected_fates):
    """
    We first normalize per clone within a time point, then per cluster.
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

    return df_X_cluster


def add_clone_id_for_each_cell(adata):
    clone_id = list(np.zeros(adata.shape[0], dtype=str))
    X_clone_old = adata.obsm["X_clone"].A
    for j in range(adata.shape[0]):
        clone_id[j] = ",".join(np.nonzero(X_clone_old[j, :])[0].astype(str))
    adata.obs["clone_id"] = clone_id
    logg.info("add information at obs['clone_id']")


def remove_multiclone_cell(
    adata, clone_threshold=2, keep_discarded_clones=True, update_X_clone=True
):
    """
    cells with clones >= clone_threshold will be removed (set to 0) from the X_clone.
    """
    barcode_N_per_cell = adata.obsm["X_clone"].sum(1).A.flatten()
    print("Multiclone cell fraction:", (barcode_N_per_cell > 1).mean())
    print(
        "Fraction of clones related to a multiclone cell:",
        (adata.obsm["X_clone"][barcode_N_per_cell > 1].sum(0) > 0).mean(),
    )
    X_clone_new = adata.obsm["X_clone"].A
    for x in np.nonzero(barcode_N_per_cell >= clone_threshold)[0]:
        for j in range(X_clone_new.shape[1]):
            X_clone_new[x, j] = 0

    absent_clone_before = np.nonzero(adata.obsm["X_clone"].A.sum(0) == 0)[0]
    lost_clones = np.nonzero(X_clone_new.sum(0) == 0)[0]
    removed_clones = set(lost_clones).difference(absent_clone_before)
    removed_clones = list(removed_clones)

    if keep_discarded_clones:
        logg.info("Add back dispeared clones due to cell removal")
        X_clone_new[:, removed_clones] = adata.obsm["X_clone"].A[:, removed_clones]

    adata.obsm["X_clone_old"] = adata.obsm["X_clone"]
    if update_X_clone:
        adata.obsm["X_clone"] = ssp.csr_matrix(X_clone_new)
        logg.info("Updated X_clone")
        logg.info(f"Check removed clones with adata.obsm['X_clone_old']")

    barcode_N_per_cell_new = X_clone_new.sum(1)
    df = pd.DataFrame(
        {
            "Clone_N_per_cell_old": barcode_N_per_cell,
            "Clone_N_per_cell_new": barcode_N_per_cell_new,
        }
    ).melt()
    fig, ax = plt.subplots()
    ax = sns.histplot(data=df, x="value", hue="variable")
    ax.set_xlabel("Clone number per cell")

    return removed_clones


def clone_statistics(adata):
    """
    Extract the number of clones and clonal cells for each time point
    """

    df = (
        pd.DataFrame(adata.obsm["X_clone"].A)
        .reset_index()
        .rename(columns={"index": "cell_id"})
        .assign(time_info=list(adata.obs["time_info"]))
        .set_index(["cell_id", "time_info"])
        .melt(ignore_index=False)
        .reset_index()
        .rename(columns={"variable": "clone_id", "value": "count"})
        .query("count>0")
    )

    df_cell = (
        df.groupby("time_info")
        .agg(cell_N=("cell_id", lambda x: len(set(x))))
        .assign(clonal_cell_fraction=lambda x: x["cell_N"] / x["cell_N"].sum())
    )

    df = df.groupby("clone_id").agg(
        cell_number=("cell_id", "count"),
        time_points=("time_info", lambda x: ",".join(list(set(x)))),
        time_point_N=("time_info", lambda x: len(set(x))),
    )

    df_clone = (
        df.groupby("time_points")
        .agg(clone_N=("time_points", "count"))
        .assign(clone_fraction=lambda x: x["clone_N"] / x["clone_N"].sum())
    )

    print(df_clone)
    print("-----------")
    print(df_cell)
    return df.reset_index()


def computer_sister_cell_distance(
    adata, selected_time=None, method="2D", key="X_emb", neighbor_number=10
):
    """
    Parameters
    ----------
        selected_time:
            selected time point
        method:
            {'2D','high'} using 2D embedding or high-dimension state measurement X_pca.
        key:
            If method='2D', decide which 2-D embedding to use. Could be 'X_umap', or 'X_emb' etc.
        neighbor_number:
            If method='high', decide the neighbor_number to construct the KNN graph.

    Returns
    -------
        df_distance:
            A dataframe for cell-cell distance
    """

    from scipy.spatial.distance import pdist, squareform

    if selected_time is None:
        adata_t1 = adata
    else:
        adata_t1 = adata[adata.obs["time_info"] == selected_time]

    if method == "2D":
        print("Use 2-dimension embedding")
        X = adata_t1.obsm[key]
        norm_distance = squareform(pdist(X))

    elif method == "high":
        print("Use high-dimension")
        norm_distance = hf.compute_shortest_path_distance(
            adata_t1, normalize=False, num_neighbors_target=neighbor_number
        )
    else:
        raise ValueError("method should be among {'2D' or 'high'}")

    t1_clone_size = adata_t1.obsm["X_clone"].A.sum(0)
    selected_clone_idx = np.nonzero(t1_clone_size >= 2)[0]
    distance_list = []
    for x in selected_clone_idx:
        ids_all = np.nonzero(adata_t1.obsm["X_clone"][:, x].A.flatten() > 0)[0]
        distance_tmp = []
        for i in range(len(ids_all)):
            for j in range(i + 1, len(ids_all)):
                dis = norm_distance[ids_all[i], ids_all[j]]
                distance_tmp.append(dis)
        distance_list.append(np.mean(distance_tmp))

    df_distance = pd.DataFrame(
        {
            "clone_id": selected_clone_idx,
            "clone_distance": distance_list,
            "random_distance": np.random.choice(
                norm_distance.flatten(), len(distance_list)
            ),
        }
    )

    fig, ax = plt.subplots()
    ax = sns.histplot(df_distance["clone_distance"], bins=20)
    ax.set_xlabel("Sister cell distance")
    x = np.mean(norm_distance.flatten())
    plt.plot([x, x], [0, 12], "-r")
    # ax.text(x,10, 'mean random dist',fontsize=15,color='r')
    below_random = np.mean(df_distance["clone_distance"] < x)
    ax.set_title(f"t={selected_time}, below random: {below_random:.2f}")
    return df_distance
