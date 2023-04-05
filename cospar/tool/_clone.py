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
from .. import settings


def clonal_fate_bias(
    adata,
    selected_fate="",
    alternative="two-sided",
    multiple_hypothesis_correction=False,
):
    """
    Compute clonal fate bias towards a cluster.

    The clonal fate bias is -log(Q-value). We calculated a P-value that
    that a clone is enriched (or depleted) in a fate, using Fisher-Exact
    test (accounting for clone size). The P-value is then corrected to
    give a Q-value by Benjamini-Hochberg procedure. The alternative
    hypothesis options are: {'two-sided', 'greater', 'less'}.
    The default is 'two-sided'. With two-sided, we simply take less likely
    scenario either 'greater' and 'less' hypothesis.

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
        target_idx = sel_index_list[0]  # whether or not a cell is in a target cluster

        ## target clone
        target_ratio_array = np.zeros(clone_N)
        P_value = np.zeros(clone_N) + 0.5

        for m in tqdm(range(clone_N)):
            clone_cell_idx = (
                X_clone[:, m].sum(1).A > 0
            ).flatten()  # cell index of clone m
            clone_size = np.sum(clone_cell_idx)  # size of this clone

            if clone_size > 0:
                target_ratio = (
                    np.sum(target_idx[clone_cell_idx]) / clone_size
                )  # fraction of cells from this clone in the target cluster
                target_ratio_array[m] = target_ratio
                cell_N_in_target = np.sum(
                    target_idx[clone_cell_idx]
                )  # number of cells from this clone that are in the target cluster
                target_size = np.sum(target_idx)
                clone_size = np.sum(clone_cell_idx)

                p1 = stats.hypergeom(M=cell_N, n=target_size, N=clone_size).cdf(
                    cell_N_in_target
                )  # observing events <=x
                p2 = stats.hypergeom(M=cell_N, n=target_size, N=clone_size).sf(
                    cell_N_in_target - 1
                )  # observing events >=x
                if alternative == "greater":
                    pvalue = p1
                elif alternative == "less":
                    pvalue = p2
                else:
                    pvalue = np.min(
                        [p1, p2]
                    )  # comparing the two hypothesis, and selecting the less likely one

                P_value[m] = pvalue

        clone_size_array = X_clone.sum(0).A.flatten()
        result = pd.DataFrame(
            {
                "clone_id": np.arange(len(clone_size_array)),
                "clone_size": clone_size_array,
                "P_value": P_value,
                "clonal_fraction_in_target_fate": target_ratio_array,
            }
        )

        if multiple_hypothesis_correction:

            def hypothesis_testing(pv):
                qv = statsmodels.sandbox.stats.multicomp.multipletests(
                    list(pv), alpha=0.05, method="fdr_bh"
                )[1]
                return qv

            df_list = []
            for x in result["clone_size"].unique():
                # display(result[result['P_value']==1])
                df_tmp = result[(result["clone_size"] == x)]
                if len(df_tmp) > 0:
                    df_tmp["Q_value"] = hypothesis_testing(df_tmp["P_value"])
                    df_list.append(df_tmp)
            df_final = pd.concat(df_list)
            df_final["Fate_bias"] = -np.log10(df_final["Q_value"])
        else:
            df_final = result
            df_final["Fate_bias"] = -np.log10(df_final["P_value"])

        df_final = df_final.sort_values("Fate_bias", ascending=False)

        adata.uns["clonal_fate_bias"] = df_final[df_final["clone_size"] > 0]
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
    logg.info("only normlaize per cluster")
    sum_X_2 = 10 ** (-10) + norm_X_cluster.sum(1)
    norm_X_cluster_clone = norm_X_cluster / sum_X_2[:, np.newaxis]

    df_X_cluster = pd.DataFrame(
        norm_X_cluster_clone,
        columns=[f"clone_{j}" for j in range(norm_X_cluster_clone.shape[1])],
    )
    df_X_cluster.index = df_X_cluster_0.index

    return df_X_cluster


def coarse_grain_clone_over_cell_clusters(
    adata,
    selected_times=None,
    selected_fates=None,
    normalize=False,
    fate_normalize_source="X_clone",
    pseudocount=10 ** (-10),
    select_clones_with_fates: list = None,
    select_clones_without_fates: list = None,
    select_clones_mode: str = "or",
):
    """
    Compute the coarse-grained X_clone matrix over specified fates and
    time points, with the option to also normalize the resulting matrix
    first cluster-wise then clone-wise.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    selected_times: `list`, optional (default: None)
        Time points to select the cell states.
    selected_fates: `list`, optional (default: all)
        List of fate clusters to use. If set to be [], use all.
    normalize:
        To perform cluster-wise, then clone-wise normalization
    fate_normalize_source:
        Source for cluster-wise normalization: {'X_clone','state_info'}. 'X_clone': directly row-normalize coarse_X_clone; 'state_info': compute each cluster size directly, and then normalize coarse_X_clone. The latter method is useful if we have single-cell resolution for each fate.
    select_clones_with_fates: list = None,
        Select clones that labels fates from this list.
    select_clones_without_fates: list = None,
        Exclude clones that labels fates from this list.
    select_clones_mode: str = {'or','and'}
        Logic rule for selection.

    Returns:
    --------
    coarse_X_clone:
        The coarse-grained X_clone matrix

    mega_cluster_list:
         The updated cluster name list
    """

    time_info = np.array(adata.obs["time_info"])
    if selected_times is not None:
        if type(selected_times) is not list:
            selected_times = [selected_times]
    sp_idx = hf.selecting_cells_by_time_points(time_info, selected_times)
    X_clone_0 = adata[sp_idx].obsm["X_clone"]
    state_annote = adata[sp_idx].obs["state_info"]

    if fate_normalize_source not in ["X_clone", "state_info"]:
        raise ValueError("fate_normalize_source not in ['X_clone','state_info']")

    if np.sum(sp_idx) == 0:
        logg.error("No cells selected. Computation aborted!")
    else:
        (
            mega_cluster_list,
            valid_fate_list,
            fate_array_flat,
            sel_index_list,
        ) = hf.analyze_selected_fates(state_annote, selected_fates)
        if len(mega_cluster_list) == 0:
            logg.error("No cells selected. Computation aborted!")
        else:
            coarse_X_clone = np.zeros((len(mega_cluster_list), X_clone_0.shape[1]))
            for j, idx in enumerate(sel_index_list):
                coarse_X_clone[j, :] = X_clone_0[idx].sum(0)

        # map original fate name to mega_cluster name
        name_map = {
            list(set(state_annote.to_numpy()[x]))[0]: mega_cluster_list[j]
            for j, x in enumerate(sel_index_list)
        }

        # perform cluster-wise, then clone-wise normalization
        if normalize:
            # normalize cluster wise
            if fate_normalize_source == "state_info":
                logg.info("normalize by state_info (in case clonal data is sparse)")
                cell_type_N = np.array([np.sum(x) for x in sel_index_list])
                norm_X_cluster = coarse_X_clone / cell_type_N[:, np.newaxis]
            elif fate_normalize_source == "X_clone":
                logg.info("normalize by X_clone")
                norm_X_cluster = coarse_X_clone / (
                    coarse_X_clone.sum(1)[:, np.newaxis] + pseudocount
                )

            # normalize clone wise within each time point
            # first check if each cluster has only a single time point
            df_time_state = (
                pd.DataFrame(
                    {
                        "state_info": adata.obs["state_info"].to_list(),
                        "time_info": adata.obs["time_info"].to_list(),
                    }
                )
                .groupby(["state_info", "time_info"])
                .count()
                .reset_index()
            )
            df_time_state = df_time_state[
                df_time_state["state_info"].isin(fate_array_flat)
            ]
            df_time_state["mega_state"] = df_time_state["state_info"].map(name_map)

            each_cluster_has_unique_time = (
                df_time_state.groupby("state_info")["time_info"].count().to_numpy() == 1
            ).all()
            if each_cluster_has_unique_time:
                # each cluster has a single time point. normalize by time_info
                logg.info(
                    "each selected cluster has a unique time point. Normalize per time point"
                )
                X_list = []
                new_fate_list = []
                for t in sorted(df_time_state.time_info.unique()):
                    fates_t = df_time_state[df_time_state["time_info"] == t][
                        "mega_state"
                    ].unique()
                    sel_idx = np.in1d(mega_cluster_list, fates_t)

                    norm_X_cluster_clone_t = norm_X_cluster[sel_idx] / (
                        norm_X_cluster[sel_idx].sum(0)[np.newaxis, :] + pseudocount
                    )  # normalize by columns (i.e. clones)

                    X_list.append(norm_X_cluster_clone_t)
                    new_fate_list += list(mega_cluster_list[sel_idx])
                norm_X_cluster_clone = np.vstack(X_list)

                # re-order the matrix
                df_X_cluster = pd.DataFrame(norm_X_cluster_clone, index=new_fate_list)
                coarse_X_clone = df_X_cluster.loc[mega_cluster_list].to_numpy()

            else:
                logg.info(
                    "each cluster do not have a unique time point. Simply column-normalize the matrix"
                )
                norm_X_cluster_clone = norm_X_cluster / (
                    norm_X_cluster.sum(0)[np.newaxis, :] + pseudocount
                )  # normalize by columns (i.e. clones)

                coarse_X_clone = norm_X_cluster_clone

        if (select_clones_with_fates is not None) or (
            select_clones_without_fates is not None
        ):
            fate_names = np.array(mega_cluster_list)
            if select_clones_mode == "and":
                valid_clone_idx = np.ones(coarse_X_clone.shape[1]).astype(bool)
                if select_clones_with_fates is not None:
                    for x_name in select_clones_with_fates:
                        valid_clone_idx_tmp = (
                            coarse_X_clone[fate_names == name_map[x_name]].sum(0) > 0
                        )
                        valid_clone_idx = valid_clone_idx & valid_clone_idx_tmp

                if select_clones_without_fates is not None:
                    for x_name in select_clones_without_fates:
                        valid_clone_idx_tmp = (
                            coarse_X_clone[fate_names == name_map[x_name]].sum(0) > 0
                        )
                        valid_clone_idx = valid_clone_idx & ~valid_clone_idx_tmp
            elif select_clones_mode == "or":
                valid_clone_idx = np.zeros(coarse_X_clone.shape[1]).astype(bool)
                if select_clones_with_fates is not None:
                    for x_name in select_clones_with_fates:
                        valid_clone_idx_tmp = (
                            coarse_X_clone[fate_names == name_map[x_name]].sum(0) > 0
                        )
                        valid_clone_idx = valid_clone_idx | valid_clone_idx_tmp

                if select_clones_without_fates is not None:
                    for x_name in select_clones_without_fates:
                        valid_clone_idx_tmp = (
                            coarse_X_clone[fate_names == name_map[x_name]].sum(0) > 0
                        )
                        valid_clone_idx = valid_clone_idx | ~valid_clone_idx_tmp

            coarse_X_clone = coarse_X_clone[:, valid_clone_idx]

        return coarse_X_clone, mega_cluster_list


def get_normalized_coarse_X_clone(
    adata,
    selected_fates,
    fate_normalize_source="X_clone",
    pseudocount=10 ** (-10),
    select_clones_with_fates: list = None,
    select_clones_without_fates: list = None,
    select_clones_mode: str = "or",
):
    """
    We first normalize per cluster, then within each time point, normalize within the clone.
    In this case, the normalized coarse_X_clone matrix sums to 1 for each clone, thereby directly
    highlighting which cell type is more preferred by a clone. Note that the cluster-cluster correlation
    will be affected by both the cluster and clone normalization.

    Parameters
    ----------
    adata:
        adata object
    selected_fates:
        Selected cell clusters to perform coarse-graining
    fate_normalize_source:
        Source for cluster-wise normalization: {'X_clone','state_info'}. 'X_clone': directly row-normalize coarse_X_clone; 'state_info': compute each cluster size directly, and then normalize coarse_X_clone. The latter method is useful if we have single-cell resolution for each fate.
    """

    if fate_normalize_source not in ["X_clone", "state_info"]:
        raise ValueError(
            "fate_normalize_source must come from {'X_clone', 'state_info'"
        )

    coarse_X_clone, mega_cluster_list = coarse_grain_clone_over_cell_clusters(
        adata,
        selected_fates=selected_fates,
        normalize=True,
        fate_normalize_source=fate_normalize_source,
        pseudocount=pseudocount,
        select_clones_with_fates=select_clones_with_fates,
        select_clones_without_fates=select_clones_without_fates,
        select_clones_mode=select_clones_mode,
    )

    df_X_cluster = pd.DataFrame(
        coarse_X_clone,
        columns=[f"clone_{j}" for j in range(coarse_X_clone.shape[1])],
        index=mega_cluster_list,
    )

    return df_X_cluster


def clonal_trajectory(
    adata,
    selected_fates,
    fate_normalize_source="X_clone",
    select_clones_with_fates: list = None,
    select_clones_without_fates: list = None,
    select_clones_mode: str = "or",
    **kwargs,
):
    """
    Assign each cell a fate choice that belongs to its clone, and the
    clonal fate choice is determined by the coarse-grained and normalized
    barcode count matrix according to coarse_grain_clone_over_cell_clusters.
    The output is stored at adata.obs[f"clonal_traj_{sel_fates[j]}"]
    """

    df_X_cluster = coarse_grain_clone_over_cell_clusters(
        adata,
        selected_fates,
        normalize=True,
        fate_normalize_source=fate_normalize_source,
        select_clones_with_fates=select_clones_with_fates,
        select_clones_without_fates=select_clones_without_fates,
        select_clones_mode=select_clones_mode,
        **kwargs,
    )
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


def add_clone_id_for_each_cell(adata):
    clone_id = list(np.zeros(adata.shape[0], dtype=str))
    X_clone_old = adata.obsm["X_clone"].A
    for j in range(adata.shape[0]):
        clone_id[j] = ",".join(np.nonzero(X_clone_old[j, :])[0].astype(str))
    adata.obs["clone_id"] = clone_id
    logg.info("add information at obs['clone_id']")


def remove_multiclone_cell(adata, **kwargs):
    return filter_cells(adata, **kwargs)


def filter_cells(
    adata, clone_threshold=2, keep_discarded_clones=True, update_X_clone=True
):
    """
    cells with clone number >= clone_threshold will be removed (set to 0) from the X_clone.
    """
    barcode_N_per_cell = adata.obsm["X_clone"].sum(1).A.flatten()
    logg.info("Multiclone cell fraction:", (barcode_N_per_cell > 1).mean())
    logg.info(
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


def remove_multicell_clone(adata, **kwargs):
    return filter_clones(adata, **kwargs)


def filter_clones(adata, clone_size_threshold=2, filter_larger_clones=False):
    """
    if filter_larger_clones=True,
        clones with clone_size >= clone_size_threshold will be removed from the X_clone.
    else:
        filter smaller clones
    """
    clone_size = adata.obsm["X_clone"].sum(0).A.flatten()
    if filter_larger_clones:
        clone_idx = clone_size < clone_size_threshold
        logg.info(
            f"Removed clone with size >= {clone_size_threshold}; fraction {1-clone_idx.mean():.2f}"
        )
    else:
        clone_idx = clone_size >= clone_size_threshold
        logg.info(
            f"Removed clone with size < {clone_size_threshold}; fraction {1-clone_idx.mean():.2f}"
        )

    X_clone_new = adata.obsm["X_clone"][:, clone_idx]

    adata.obsm["X_clone_old"] = adata.obsm["X_clone"]
    if "clone_id" not in adata.uns:
        adata.uns["clone_id"] = np.arange(adata.obsm["X_clone"].shape[1])
    adata.uns["clone_id"] = np.array(adata.uns["clone_id"])[clone_idx]
    adata.obsm["X_clone"] = ssp.csr_matrix(X_clone_new)
    logg.info("Updated X_clone")
    logg.info(f"Check removed clones with adata.obsm['X_clone_old']")


def clone_statistics(adata, joint_variable="time_info", display_clone_stat=True):
    """
    Extract the number of clones and clonal cells for each time point
    """

    if "clone_id" not in adata.uns:
        adata.uns["clone_id"] = np.arange(adata.obsm["X_clone"].shape[1])
    adata.obs[joint_variable] = adata.obs[joint_variable].astype(str)

    df = (
        pd.DataFrame(adata.obsm["X_clone"].A)
        .reset_index()
        .rename(columns={"index": "cell_id"})
        .assign(time_info=list(adata.obs[joint_variable]))
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

    if display_clone_stat:
        logg.info(
            df_clone.reset_index()
            .rename(columns={"time_points": joint_variable})
            .set_index(joint_variable)
        )
        logg.info("-----------")
        logg.info(
            df_cell.reset_index()
            .rename(columns={"time_info": joint_variable})
            .set_index(joint_variable)
        )
    return df.reset_index().rename(
        columns={
            "time_point_N": f"{joint_variable}_N",
            "time_points": joint_variable,
            "cell_number": "clone_size",
        }
    )


def get_distance_within_each_clone(X_clone, norm_distance):
    # get distance vector for each clone
    t1_clone_size = (X_clone).sum(0)
    selected_clone_idx = np.nonzero(t1_clone_size >= 2)[0]
    distance_list = []
    for x in selected_clone_idx:
        ids_all = np.nonzero(X_clone[:, x] > 0)[0]
        distance_tmp = []
        for i in range(len(ids_all)):
            distance_tmp.append(
                [
                    norm_distance[ids_all[i], ids_all[j]]
                    for j in range(len(ids_all))
                    if j != i
                ]
            )
        distance_list.append(np.array(distance_tmp).flatten().max())  # min.mean
    return distance_list, selected_clone_idx


def compute_sister_cell_distance(
    adata,
    selected_time=None,
    method="2D",
    precomputed_distance_matrix=None,
    key="X_emb",
    neighbor_number=10,
    title=None,
    color_random="#a6bddb",
    color_data="#fdbb84",
    max_N_simutation=1000,
    alternative="less",
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
        max_N_simutation:
            Max number to perform simulation to generate a null distribution. A higher number may give more stronger
            statistical statement.
        alternative:
            'less', 'greater','two-sided'

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

    if precomputed_distance_matrix is not None:
        logg.info("use precomputed distance matrix")
        norm_distance = precomputed_distance_matrix
    else:
        if method == "2D":
            logg.info("Use 2-dimension embedding")
            X = adata_t1.obsm[key]
            norm_distance = squareform(pdist(X))

        elif method == "high":
            logg.info("Use high-dimension")
            norm_distance = hf.compute_shortest_path_distance(
                adata_t1,
                normalize=False,
                num_neighbors_target=neighbor_number,
            )
        else:
            raise ValueError("method should be among {'2D' or 'high'}")

    X_clone = (adata_t1.obsm["X_clone"].A.copy() > 0).astype(int)
    logg.info(np.sum(X_clone.sum(0) >= 2), " clones with >=2 cells selected")

    # observed distances
    distance_list, selected_clone_idx = get_distance_within_each_clone(
        X_clone, norm_distance
    )

    # randomized distances
    random_dis = []
    random_dis_stat = []
    for _ in tqdm(range(max_N_simutation)):
        np.random.shuffle(X_clone)
        temp, __ = get_distance_within_each_clone(X_clone, norm_distance)
        random_dis += temp
        random_dis_stat.append(
            [np.mean(temp), np.min(temp), np.median(temp), np.max(temp)]
        )
    random_dis_stat = np.array(random_dis_stat)

    df_obs = pd.DataFrame(
        {
            "clone_id": selected_clone_idx,
            "distance": distance_list,
        }
    )
    df_obs["source"] = "Observed"

    df_rand = pd.DataFrame(
        {
            "clone_id": np.arange(len(random_dis)).astype(str),
            "distance": random_dis,
        }
    )
    df_rand["clone_id"] = "rand_" + df_rand["clone_id"]
    df_rand["source"] = "Random"

    df_distance = pd.concat([df_obs, df_rand], ignore_index=True)
    df_distance["source"] = pd.Categorical(df_distance["source"]).set_categories(
        ["Random", "Observed"], ordered=True
    )

    bins = np.linspace(df_distance["distance"].min(), df_distance["distance"].max(), 50)
    ax = sns.histplot(
        data=df_distance[df_distance["source"] == "Random"],
        label="Random",
        bins=bins,
        stat="probability",
        color=color_random,
    )
    ax = sns.histplot(
        data=df_distance[df_distance["source"] == "Observed"],
        x="distance",
        label="Observed",
        bins=bins,
        stat="probability",
        color=color_data,
        alpha=0.5,
    )
    ax.legend()
    ax.set_xlabel("Sister-cell distance")

    stat_score, pvalue = stats.ranksums(
        df_obs["distance"].to_numpy(), random_dis, alternative=alternative
    )

    if title is None:
        ax.set_title(f"t={selected_time}, Pvalue: {pvalue:.2f}")
    else:
        ax.set_title(title)
    plt.tight_layout()
    if "data_des" in adata.uns.keys():
        data_des = adata.uns["data_des"][-1]
    else:
        data_des = ""
    plt.savefig(f"{settings.figure_path}/transcriptome_memory{data_des}.pdf")

    return df_distance, pvalue
