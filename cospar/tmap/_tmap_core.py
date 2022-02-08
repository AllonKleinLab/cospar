import os
import time

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as ssp
from tqdm import tqdm

from cospar.tmap import _utils as tmap_util

from .. import help_functions as hf
from .. import logging as logg
from .. import settings
from .. import tool as tl
from .optimal_transport import optimal_transport_duality_gap, transport_stablev2


def refine_Tmap_through_cospar(
    MultiTime_cell_id_array_t1,
    MultiTime_cell_id_array_t2,
    proportion,
    transition_map,
    X_clone,
    initial_similarity,
    final_similarity,
    sparsity_threshold=0.1,
    normalization_mode=1,
):
    """
    This performs one iteration of coherent sparsity optimization.

    This is our core algorithm for Tmap inference. It updates a map
    by considering clones spanning multiple time points.

    Parameters
    ----------
    MultiTime_cell_id_array_t1: `np.array`
        An array of cell id sub_array, where each sub_array consists of
        clonally-related cell id's at different time points
    MultiTime_cell_id_array_t2: `np.array`
        An corresponding array of sub_array, where each sub_array are id's of
        cells that are clonally related to the corresponding sub_array at
        MultiTime_cell_id_array_t1.
    proportion: `list`
        A weight factor for each time point.
    transition_map: `np.array` or `sp.spmatrix`
        initialized transition map, or map from a previous iteration.
    X_clone: `sp.spmatrix`
        clonal matrix
    initial_similarity: `np.array`
        similarity matrix for all cells belonging
        to MultiTime_cell_id_array_t1
    final_similarity: `np.array`
        similarity matrix for all cells belonging
        to MultiTime_cell_id_array_t2
    sparsity_threshold: `float`, optional (default: 0.1)
        The relative threshold to remove noises in the updated transition map,
        in the range [0,1].
    normalization_mode: `int`, optional (default: 1)
        Normalization method. Choice: [0,1].
        0, single-cell normalization; 1, Clone normalization. The clonal
        normalization suppresses the contribution of large
        clones, and is much more robust.

    Returns
    -------
    smoothed_new_transition_map: `np.array`
    un_SM_transition_map: `np.array`
    """

    resol = 10 ** (-10)
    transition_map = hf.matrix_row_or_column_thresholding(
        transition_map, sparsity_threshold, row_threshold=True
    )

    if normalization_mode == 0:
        logg.hint("Single-cell normalization")
    if normalization_mode == 1:
        logg.hint("Clone normalization")

    if ssp.issparse(X_clone):
        X_clone = ssp.csr_matrix(X_clone)

    cell_N, clone_N = X_clone.shape
    N1, N2 = transition_map.shape
    new_coupling_matrix = ssp.lil_matrix((N1, N2))

    # cell id order in the similarity matrix is obtained by concatenating the cell id
    # list in MultiTime_cell_id_array_t1. So, we need to offset the id if we move to the next list
    offset_N1 = 0
    offset_N2 = 0
    for j in range(len(MultiTime_cell_id_array_t1)):

        logg.hint("Relative time point pair index:", j)
        cell_id_array_t1 = MultiTime_cell_id_array_t1[j]
        cell_id_array_t2 = MultiTime_cell_id_array_t2[j]

        for clone_id in range(clone_N):
            # pdb.set_trace()

            if clone_id % 1000 == 0:
                logg.hint("Clone id:", clone_id)
            idx1 = X_clone[cell_id_array_t1, clone_id].A.flatten()
            idx2 = X_clone[cell_id_array_t2, clone_id].A.flatten()
            if idx1.sum() > 0 and idx2.sum() > 0:
                ## update the new_coupling matrix
                id_1 = offset_N1 + np.nonzero(idx1)[0]
                id_2 = offset_N2 + np.nonzero(idx2)[0]
                prob = transition_map[id_1][:, id_2]

                ## try row normalization
                if normalization_mode == 0:
                    prob = hf.sparse_rowwise_multiply(
                        prob, 1 / (resol + np.sum(prob, 1))
                    )  # cell-level normalization
                else:
                    prob = prob / (
                        resol + np.sum(prob)
                    )  # clone level normalization, account for proliferation

                weight_factor = np.sqrt(
                    np.mean(idx1[idx1 > 0]) * np.mean(idx2[idx2 > 0])
                )  # the contribution of a particular clone can be tuned by its average entries
                if weight_factor > 1:
                    logg.hint(
                        "X_clone has entries not 0 or 1. Using weight modulation",
                        weight_factor,
                    )

                # Use the add mode, add up contributions from each clone
                new_coupling_matrix[id_1[:, np.newaxis], id_2] = (
                    new_coupling_matrix[id_1[:, np.newaxis], id_2]
                    + proportion[j] * prob * weight_factor
                )

        ## update offset
        offset_N1 = offset_N1 + len(cell_id_array_t1)
        offset_N2 = offset_N2 + len(cell_id_array_t2)

    ## rescale
    new_coupling_matrix = new_coupling_matrix / (new_coupling_matrix.A.max())

    ## convert to sparse matrix form
    new_coupling_matrix = new_coupling_matrix.tocsr()

    logg.hint("Start to smooth the refined clonal map")
    t = time.time()
    temp = new_coupling_matrix * final_similarity

    logg.hint("Phase I: time elapsed -- ", time.time() - t)
    smoothed_new_transition_map = initial_similarity.dot(temp)

    logg.hint("Phase II: time elapsed -- ", time.time() - t)

    # both return are numpy array
    un_SM_transition_map = new_coupling_matrix.A
    return smoothed_new_transition_map, un_SM_transition_map


def refine_Tmap_through_cospar_noSmooth(
    MultiTime_cell_id_array_t1,
    MultiTime_cell_id_array_t2,
    proportion,
    transition_map,
    X_clone,
    sparsity_threshold=0.1,
    normalization_mode=1,
):
    """
    This performs one iteration of coherent sparsity optimization

    This is the same as 'refine_Tmap_through_cospar', except that
    there is no smoothing in the end.

    Parameters
    ----------
    MultiTime_cell_id_array_t1: `np.array`
        an array of cell id sub_array, where each sub_array consists of
        clonally-related cell id's at different time points
    MultiTime_cell_id_array_t2: `np.array`
        an corresponding array of sub_array, where each sub_array are id's of
        cells that are clonally related to the corresponding sub_array at
        MultiTime_cell_id_array_t1.
    proportion: `list`
        A weight factor for each time point.
    transition_map: `np.array` or `sp.spmatrix`
        initialized transition map, or map from a previous iteration.
    X_clone: `sp.spmatrix`
        clonal matrix
    initial_similarity: `np.array`
        similarity matrix for all cells belonging
        to MultiTime_cell_id_array_t1
    final_similarity: `np.array`
        similarity matrix for all cells belonging
        to MultiTime_cell_id_array_t2
    sparsity_threshold: `float`, optional (default: 0.1)
        noise threshold to remove noises in the updated transition map,
        in the range [0,1]
    normalization_mode: `int`, optional (default: 1)
        Normalization method. Choice: [0,1].
        0, single-cell normalization; 1, Clone normalization. The clonal
        normalization suppresses the contribution of large
        clones, and is much more robust.

    Returns
    -------
    un_SM_transition_map: `np.array`
    """

    if not isinstance(X_clone[0, 0], bool):
        X_clone = X_clone.astype(bool)

    resol = 10 ** (-10)

    if normalization_mode == 0:
        logg.hint("Single-cell normalization")
    if normalization_mode == 1:
        logg.hint("Clone normalization")

    transition_map = hf.matrix_row_or_column_thresholding(
        transition_map, sparsity_threshold, row_threshold=True
    )

    if not ssp.issparse(transition_map):
        transition_map = ssp.csr_matrix(transition_map)
    if not ssp.issparse(X_clone):
        X_clone = ssp.csr_matrix(X_clone)

    cell_N, clone_N = X_clone.shape
    N1, N2 = transition_map.shape
    new_coupling_matrix = ssp.lil_matrix((N1, N2))

    offset_N1 = 0
    offset_N2 = 0
    for j in range(len(MultiTime_cell_id_array_t1)):

        logg.hint("Relative time point pair index:", j)
        cell_id_array_t1 = MultiTime_cell_id_array_t1[j]
        cell_id_array_t2 = MultiTime_cell_id_array_t2[j]

        for clone_id in range(clone_N):

            if clone_id % 1000 == 0:
                logg.hint("Clone id:", clone_id)
            idx1 = X_clone[cell_id_array_t1, clone_id].A.flatten()
            idx2 = X_clone[cell_id_array_t2, clone_id].A.flatten()
            if idx1.sum() > 0 and idx2.sum() > 0:
                ## update the new_coupling matrix
                id_1 = offset_N1 + np.nonzero(idx1)[0]
                id_2 = offset_N2 + np.nonzero(idx2)[0]
                prob = transition_map[id_1][:, id_2].A

                ## try row normalization
                if normalization_mode == 0:
                    prob = hf.sparse_rowwise_multiply(
                        prob, 1 / (resol + np.sum(prob, 1))
                    )  # cell-level normalization
                else:
                    prob = prob / (
                        resol + np.sum(prob)
                    )  # clone level normalization, account for proliferation

                weight_factor = np.sqrt(
                    np.mean(idx1[idx1 > 0]) * np.mean(idx2[idx2 > 0])
                )  # the contribution of a particular clone can be tuned by its average entries
                if weight_factor > 1:
                    logg.hint("marker gene weight", weight_factor)

                # Use the add mode, add up contributions from each clone
                new_coupling_matrix[id_1[:, np.newaxis], id_2] = (
                    new_coupling_matrix[id_1[:, np.newaxis], id_2]
                    + proportion[j] * prob * weight_factor
                )

        ## update offset
        offset_N1 = offset_N1 + len(cell_id_array_t1)
        offset_N2 = offset_N2 + len(cell_id_array_t2)

    ## convert to sparse matrix form
    new_coupling_matrix = new_coupling_matrix.tocsr()
    #
    un_SM_transition_map = new_coupling_matrix
    return un_SM_transition_map


def infer_Tmap_from_multitime_clones_private(
    adata,
    smooth_array=[15, 10, 5],
    neighbor_N=20,
    sparsity_threshold=0.1,
    intraclone_threshold=0.05,
    normalization_mode=1,
    save_subset=True,
    use_full_Smatrix=True,
    trunca_threshold=[0.001, 0.01],
    compute_new_Smatrix=False,
    max_iter_N=5,
    epsilon_converge=0.05,
):
    """
    Internal function for Tmap inference from multi-time clonal data.

    Same as :func:`.infer_Tmap_from_multitime_clones` except that it
    assumes that the adata object has been prepared for targeted
    time points. It generates the similarity matrix
    via :func:`.tmap_util.generate_similarity_matrix`, and iteratively calls
    the core function :func:`.refine_Tmap_through_cospar` to update
    the transition map.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Should be prepared by :func:`cospar.tmap._utils.select_time_points`
    smooth_array: `list`, optional (default: [15,10,5])
        List of smooth rounds at initial runs of iteration.
        Suppose that it has a length N. For iteration n<N, the n-th entry of
        smooth_array determines the kernel exponent to build the S matrix at the n-th
        iteration. When n>N, we use the last entry of smooth_array to compute
        the S matrix. We recommend starting with more smoothing depth and gradually
        reduce the depth, as inspired by simulated annealing. Data with higher
        clonal dispersion should start with higher smoothing depth. The final depth should
        depend on the manifold itself. For fewer cells, it results in a small KNN graph,
        and a small final depth should be used. We recommend to use a number at
        the multiple of 5 for computational efficiency i.e.,
        smooth_array=[20, 15, 10, 5], or [20,15,10]
    max_iter_N: `int`, optional (default: 5)
        The maximum iterations used to compute the transition map, regardless of epsilon_converge.
    epsilon_converge: `float`, optional (default: 0.05)
        The convergence threshold for the change of map correlations between consecutive iterations.
        This convergence test is activated only when CoSpar has iterated for 3 times.
    neighbor_N: `int`, optional (default: 20)
        The number of neighbors for KNN graph used for computing the similarity matrix.
    trunca_threshold: `list`, optional (default: [0.001,0.01])
        Threshold to reset entries of a matrix to zero. The first entry is for
        Similarity matrix; the second entry is for the Tmap.
        This is only for computational and storage efficiency.
    sparsity_threshold: `float`, optional (default: 0.1)
        The relative threshold to remove noises in the updated transition map,
        in the range [0,1].
    intraclone_threshold: `float`, optional (default: 0.05)
        The threshold to remove noises in the demultiplexed (un-smoothed) map,
        in the range [0,1]
    normalization_mode: `int`, optional (default: 1)
        Normalization method. Choice: [0,1].
        0, single-cell normalization; 1, Clone normalization. The clonal
        normalization suppresses the contribution of large
        clones, and is much more robust.
    save_subset: `bool`, optional (default: True)
        If true, save only Smatrix at smooth round [5,10,15,...];
        Otherwise, save Smatrix at each round.
    use_full_Smatrix: `bool`, optional (default: True)
        If true, extract the relevant Smatrix from the full Smatrix defined by all cells.
        This tends to be more accurate. The package is optimized around this choice.
    compute_new_Smatrix: `bool`, optional (default: False)
        If True, compute Smatrix from scratch, whether it was
        computed and saved before or not. This is activated only when
        `use_full_Smatrix=False`.

    Returns
    -------
    None. Inferred transition map updated at adata.uns['transition_map']
    and adata.uns['intraclone_transition_map']
    """

    ########## extract data
    clone_annot = adata.obsm["X_clone"]
    clonal_cell_id_t1 = adata.uns["clonal_cell_id_t1"]
    clonal_cell_id_t2 = adata.uns["clonal_cell_id_t2"]
    Tmap_cell_id_t1 = adata.uns["Tmap_cell_id_t1"]
    Tmap_cell_id_t2 = adata.uns["Tmap_cell_id_t2"]
    sp_idx = adata.uns["sp_idx"]
    data_des = adata.uns["data_des"][0]  # original label
    data_des_1 = adata.uns["data_des"][
        -1
    ]  # current label, sensitive to selected time points
    multiTime_cell_id_t1 = adata.uns["multiTime_cell_id_t1"]
    multiTime_cell_id_t2 = adata.uns["multiTime_cell_id_t2"]
    proportion = adata.uns["proportion"]
    data_path = settings.data_path

    ######### check whether we need to extend the map space
    ratio_t1 = np.sum(np.in1d(Tmap_cell_id_t1, clonal_cell_id_t1)) / len(
        Tmap_cell_id_t1
    )
    ratio_t2 = np.sum(np.in1d(Tmap_cell_id_t2, clonal_cell_id_t2)) / len(
        Tmap_cell_id_t2
    )
    if (ratio_t1 == 1) and (ratio_t2 == 1):
        extend_Tmap_space = False  # no need to extend the map space
    else:
        extend_Tmap_space = True

    ########################### Compute the transition map
    temp_str = "0" + str(trunca_threshold[0])[2:]

    if use_full_Smatrix:
        similarity_file_name = os.path.join(
            data_path,
            f"{data_des}_Similarity_matrix_with_all_cell_states_kNN{neighbor_N}_Truncate{temp_str}",
        )
        for round_of_smooth in smooth_array:
            if not os.path.exists(similarity_file_name + f"_SM{round_of_smooth}.npz"):
                raise ValueError(
                    f"Similarity matrix at given parameters have not been computed before! File name: {similarity_file_name}\n"
                    "Please re-run the function with: compute_new=True. If you want to use smooth round not the multiples of 5, set save_subset=False"
                )

    else:
        similarity_file_name = os.path.join(
            data_path,
            f"{data_des_1}_Similarity_matrix_with_selected_states_kNN{neighbor_N}_Truncate{temp_str}",
        )

    initial_similarity_array = []
    final_similarity_array = []
    initial_similarity_array_ext = []
    final_similarity_array_ext = []

    logg.info("Load pre-computed similarity matrix")

    if "Smatrix" not in adata.uns.keys():
        logg.hint("Load from hard disk--------")
        for round_of_smooth in smooth_array:
            # we cannot force it to compute new at this time. Otherwise, if we use_full_Smatrix, the resulting similarity is actually from adata, thus not full similarity.

            re_compute = (not use_full_Smatrix) and (
                compute_new_Smatrix
            )  # re-compute only when not using full similarity
            similarity_matrix_full = tmap_util.generate_similarity_matrix(
                adata,
                similarity_file_name,
                round_of_smooth=round_of_smooth,
                neighbor_N=neighbor_N,
                truncation_threshold=trunca_threshold[0],
                save_subset=save_subset,
                compute_new_Smatrix=re_compute,
            )

            ## add dimensionality check
            if (similarity_matrix_full.shape[0] != len(sp_idx)) or (
                similarity_matrix_full.shape[1] != len(sp_idx)
            ):
                raise ValueError(
                    "The pre-computed similarity matrix does not have the right dimension.\n"
                    "Possible reason: this is computed from a different dataset, but with the same data label in adata.uns['data_des']\n"
                    "You can fix this issue by running the Tmap inference with compute_new=True, which compute everything from scratch.\n"
                    "To avoid this from happening again, please use a different data_des for each new data, which can be set with adata.uns['data_des']=['Your_data_des']."
                )

            if use_full_Smatrix:
                # pdb.set_trace()
                similarity_matrix_full_sp = similarity_matrix_full[sp_idx][:, sp_idx]

                ### minimum similarity matrix that only involves the multi-time clones
                initial_similarity = tmap_util.generate_initial_similarity(
                    similarity_matrix_full_sp, clonal_cell_id_t1, clonal_cell_id_t1
                )
                final_similarity = tmap_util.generate_final_similarity(
                    similarity_matrix_full_sp, clonal_cell_id_t2, clonal_cell_id_t2
                )

                if extend_Tmap_space:
                    initial_similarity_ext = tmap_util.generate_initial_similarity(
                        similarity_matrix_full_sp, Tmap_cell_id_t1, clonal_cell_id_t1
                    )
                    final_similarity_ext = tmap_util.generate_final_similarity(
                        similarity_matrix_full_sp, clonal_cell_id_t2, Tmap_cell_id_t2
                    )

            else:
                initial_similarity = tmap_util.generate_initial_similarity(
                    similarity_matrix_full, clonal_cell_id_t1, clonal_cell_id_t1
                )
                final_similarity = tmap_util.generate_final_similarity(
                    similarity_matrix_full, clonal_cell_id_t2, clonal_cell_id_t2
                )

                if extend_Tmap_space:
                    initial_similarity_ext = tmap_util.generate_initial_similarity(
                        similarity_matrix_full, Tmap_cell_id_t1, clonal_cell_id_t1
                    )
                    final_similarity_ext = tmap_util.generate_final_similarity(
                        similarity_matrix_full, clonal_cell_id_t2, Tmap_cell_id_t2
                    )

            initial_similarity_array.append(initial_similarity)
            final_similarity_array.append(final_similarity)
            if extend_Tmap_space:
                initial_similarity_array_ext.append(initial_similarity_ext)
                final_similarity_array_ext.append(final_similarity_ext)

        # loading the map is too costly. We attach it to adata, and remove that after Tmap inference
        # This is useful only for the joint optimization.
        adata.uns["Smatrix"] = {}
        adata.uns["Smatrix"]["initial_similarity_array"] = initial_similarity_array
        adata.uns["Smatrix"]["final_similarity_array"] = final_similarity_array
        adata.uns["Smatrix"][
            "initial_similarity_array_ext"
        ] = initial_similarity_array_ext
        adata.uns["Smatrix"]["final_similarity_array_ext"] = final_similarity_array_ext

    else:
        logg.hint("Copy from adata (pre-loaded)--------")
        initial_similarity_array = adata.uns["Smatrix"]["initial_similarity_array"]
        final_similarity_array = adata.uns["Smatrix"]["final_similarity_array"]
        initial_similarity_array_ext = adata.uns["Smatrix"][
            "initial_similarity_array_ext"
        ]
        final_similarity_array_ext = adata.uns["Smatrix"]["final_similarity_array_ext"]

    #### Compute the core of the transition map that involve multi-time clones, then extend to other cell states
    transition_map = np.ones((len(clonal_cell_id_t1), len(clonal_cell_id_t2)))
    # transition_map_array=[transition_map_v1]

    X_clone = clone_annot.copy()
    if not ssp.issparse(X_clone):
        X_clone = ssp.csr_matrix(X_clone)

    # smooth_iter_N=len(smooth_array)
    for j in range(max_iter_N):

        # transition_map=Tmap_temp
        if j < len(smooth_array):

            logg.info(f"Iteration {j+1}, Use smooth_round={smooth_array[j]}")
            used_initial_similarity = initial_similarity_array[j]
            used_final_similarity = final_similarity_array[j]
        else:

            logg.info(f"Iteration {j+1}, Use smooth_round={smooth_array[-1]}")
            used_initial_similarity = initial_similarity_array[-1]
            used_final_similarity = final_similarity_array[-1]

        transition_map_new, unSM_sc_coupling = refine_Tmap_through_cospar(
            multiTime_cell_id_t1,
            multiTime_cell_id_t2,
            proportion,
            transition_map,
            X_clone,
            used_initial_similarity,
            used_final_similarity,
            sparsity_threshold=sparsity_threshold,
            normalization_mode=normalization_mode,
        )

        ########################### Convergency test
        # sample cell states to convergence test
        sample_N_x = 50
        sample_N_y = 100
        t0 = time.time()
        cell_N_tot_x = transition_map.shape[0]
        if cell_N_tot_x < sample_N_x:
            sample_id_temp_x = np.arange(cell_N_tot_x)
        else:
            xx = np.arange(cell_N_tot_x)
            yy = (
                list(np.nonzero(xx % 3 == 0)[0])
                + list(np.nonzero(xx % 3 == 1)[0])
                + list(np.nonzero(xx % 3 == 2)[0])
            )
            sample_id_temp_x = yy[:sample_N_x]

        cell_N_tot_y = transition_map.shape[1]
        if cell_N_tot_y < sample_N_y:
            sample_id_temp_y = np.arange(cell_N_tot_y)
        else:
            xx = np.arange(cell_N_tot_y)
            yy = (
                list(np.nonzero(xx % 3 == 0)[0])
                + list(np.nonzero(xx % 3 == 1)[0])
                + list(np.nonzero(xx % 3 == 2)[0])
            )
            sample_id_temp_y = yy[:sample_N_y]

        # transition_map is changed bytmap_core.refine_Tmap_through_cospar (thresholding). So, we only use transition_map_new to update
        if j == 0:
            X_map_0 = transition_map[sample_id_temp_x, :][:, sample_id_temp_y]
        else:
            X_map_0 = X_map_1.copy()

        X_map_1 = transition_map_new[sample_id_temp_x, :][:, sample_id_temp_y].copy()
        transition_map = transition_map_new

        if (j >= 2) and (
            j + 1 >= len(smooth_array)
        ):  # only perform convergency test after at least 3 iterations
            verbose = logg._settings_verbosity_greater_or_equal_than(3)
            corr_X = np.diag(hf.corr2_coeff(X_map_0, X_map_1)).mean()
            if verbose:
                from matplotlib import pyplot as plt

                fig = plt.figure()
                ax = plt.subplot(1, 1, 1)
                ax.plot(X_map_0.flatten(), X_map_1.flatten(), ".r")
                ax.set_xlabel("$T_{ij}$: previous iteration")
                ax.set_ylabel("$T_{ij}$: current iteration")
                ax.set_title(f"CoSpar, iter_N={j+1}, R={int(100*corr_X)/100}")
                plt.show()
            else:
                logg.info(
                    f"Convergence (CoSpar, iter_N={j+1}): corr(previous_T, current_T)={int(1000*corr_X)/1000}"
                )
                # logg.info(f"Convergence (CoSpar, iter_N={j+1}): corr(previous_T, current_T)={corr_X}; cost time={time.time()-t0}")

            if (1 - corr_X) < epsilon_converge:
                break
        #############################

    ### expand the map to other cell states
    if not extend_Tmap_space:

        logg.hint(
            "No need for Final Smooth (i.e., clonally-labeled states are the final state space for Tmap)"
        )

        transition_map = hf.matrix_row_or_column_thresholding(
            transition_map, threshold=trunca_threshold[1], row_threshold=True
        )
        adata.uns["transition_map"] = ssp.csr_matrix(transition_map)
    else:

        logg.hint(
            "Final round of Smooth (to expand the state space of Tmap to include non-clonal states)"
        )

        if j < len(smooth_array):
            used_initial_similarity_ext = initial_similarity_array_ext[j]
            used_final_similarity_ext = final_similarity_array_ext[j]
        else:
            used_initial_similarity_ext = initial_similarity_array_ext[-1]
            used_final_similarity_ext = final_similarity_array_ext[-1]

        unSM_sc_coupling = ssp.csr_matrix(unSM_sc_coupling)
        t = time.time()
        temp = unSM_sc_coupling * used_final_similarity_ext

        logg.hint("Phase I: time elapsed -- ", time.time() - t)
        transition_map_1 = used_initial_similarity_ext.dot(temp)

        logg.hint("Phase II: time elapsed -- ", time.time() - t)

        transition_map_1 = hf.matrix_row_or_column_thresholding(
            transition_map_1, threshold=trunca_threshold[1], row_threshold=True
        )
        adata.uns["transition_map"] = ssp.csr_matrix(transition_map_1)

    logg.hint("----Intraclone transition map----")

    demultiplexed_map_0 = refine_Tmap_through_cospar_noSmooth(
        multiTime_cell_id_t1,
        multiTime_cell_id_t2,
        proportion,
        transition_map,
        X_clone,
        sparsity_threshold=intraclone_threshold,
        normalization_mode=normalization_mode,
    )

    idx_t1 = hf.converting_id_from_fullSpace_to_subSpace(
        clonal_cell_id_t1, Tmap_cell_id_t1
    )[0]
    idx_t2 = hf.converting_id_from_fullSpace_to_subSpace(
        clonal_cell_id_t2, Tmap_cell_id_t2
    )[0]
    demultiplexed_map = np.zeros((len(Tmap_cell_id_t1), len(Tmap_cell_id_t2)))
    demultiplexed_map[idx_t1[:, np.newaxis], idx_t2] = demultiplexed_map_0.A
    adata.uns["intraclone_transition_map"] = ssp.csr_matrix(demultiplexed_map)


## This is just used for testing WOT
def infer_Tmap_from_optimal_transport_v0(
    adata,
    OT_epsilon=0.02,
    OT_dis_KNN=None,
    OT_solver=None,
    OT_cost=None,
    compute_new=None,
    use_existing_KNN_graph=None,
):
    """
    Test WOT

    In order to use non-uniform cell_grwoth_rate,
    we implicitly assume that you should pass a variable
    to adata at .obs['cell_growth_rate']

    Also note that only the variable `adata`, `OT_epsilon`, and .obs['cell_growth_rate']
    affects the result.

    Returns
    -------
    None. Results are stored at adata.uns['OT_transition_map'].
    """

    cell_id_array_t1 = adata.uns["Tmap_cell_id_t1"]
    cell_id_array_t2 = adata.uns["Tmap_cell_id_t2"]
    data_des = adata.uns["data_des"][0]
    data_path = settings.data_path

    logg.warn("-------------Using WOT----------------")
    logg.warn(f"epsilon={OT_epsilon}")
    import wot

    time_info = np.zeros(adata.shape[0])
    time_info[cell_id_array_t1] = 1
    time_info[cell_id_array_t2] = 2
    adata.obs["day"] = time_info

    if "cell_growth_rate" not in adata.obs.keys():
        print("Use uniform growth rate")
        adata.obs["cell_growth_rate"] = np.ones(len(time_info))
    else:
        x0 = np.mean(adata.obs["cell_growth_rate"])
        y0 = np.std(adata.obs["cell_growth_rate"])
        print(f"Use pre-supplied cell_grwoth_rate (mean: {x0:.2f}; std {y0:.2f})")

    ot_model = wot.ot.OTModel(adata, epsilon=OT_epsilon, lambda1=1, lambda2=50)
    OT_transition_map = ot_model.compute_transport_map(1, 2).X

    adata.uns["OT_transition_map"] = ssp.csr_matrix(OT_transition_map)


########### v1, with convergence test, 20210326
# We tested that, for clones of all different sizes, where np.argsort gives unique results,
# this method reproduces the v01, v1 results, when use_fixed_clonesize_t1=True, and when change
# sort_clone=0,1,-1.
def refine_Tmap_through_joint_optimization(
    adata,
    initialized_map,
    smooth_array=[15, 10, 5],
    max_iter_N=[1, 5],
    epsilon_converge=[0.05, 0.05],
    CoSpar_KNN=20,
    normalization_mode=1,
    sparsity_threshold=0.2,
    use_full_Smatrix=True,
    trunca_threshold=[0.001, 0.01],
    compute_new=True,
    use_fixed_clonesize_t1=False,
    sort_clone=1,
    save_subset=True,
):
    """
    Infer Tmap from clones with a single time point

    Starting from an initialized transitin map from state information,
    we jointly infer the initial clonal states and the transition map.

    This method has been optimized to be very fast. Besides, it is
    deterministic.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Should have only two time points.
    initialized_map: `sp.spmatrix`
        Initialized transition map based on state information alone.
    normalization_mode: `int`, optional (default: 1)
        Normalization method. Choice: [0,1].
        0, single-cell normalization; 1, Clone normalization. The clonal
        normalization suppresses the contribution of large
        clones, and is much more robust.
    smooth_array: `list`, optional (default: [15,10,5])
        List of smooth rounds at initial runs of iteration.
        Suppose that it has a length N. For iteration n<N, the n-th entry of
        smooth_array determines the kernel exponent to build the S matrix at the n-th
        iteration. When n>N, we use the last entry of smooth_array to compute
        the S matrix. We recommend starting with more smoothing depth and gradually
        reduce the depth, as inspired by simulated annealing. Data with higher
        clonal dispersion should start with higher smoothing depth. The final depth should
        depend on the manifold itself. For fewer cells, it results in a small KNN graph,
        and a small final depth should be used. We recommend to use a number at
        the multiple of 5 for computational efficiency i.e.,
        smooth_array=[20, 15, 10, 5], or [20,15,10]
    max_iter_N: `list`, optional (default: [1,5])
        A list for maximum iterations for the Joint optimization and CoSpar core function, respectively.
    epsilon_converge: `list`, optional (default: [0.05,0.05])
        A list of convergence threshold for the Joint optimization and CoSpar core function, respectively.
        The convergence threshold is for the change of map correlations between consecutive iterations.
        For CoSpar core function, this convergence test is activated only when CoSpar has iterated for 3 times.
    CoSpar_KNN: `int`, optional (default: 20)
        The number of neighbors for KNN graph used for computing the similarity matrix.
    trunca_threshold: `list`, optional (default: [0.001,0.01])
        Threshold to reset entries of a matrix to zero. The first entry is for
        Similarity matrix; the second entry is for the Tmap.
        This is only for computational and storage efficiency.
    sparsity_threshold: `float`, optional (default: 0.1)
        The relative threshold to remove noises in the updated transition map,
        in the range [0,1].
    save_subset: `bool`, optional (default: True)
        If true, save only Smatrix at smooth round [5,10,15,...];
        Otherwise, save Smatrix at each round.
    use_full_Smatrix: `bool`, optional (default: True)
        If true, extract the relevant Smatrix from the full Smatrix defined by all cells.
        This tends to be more accurate. The package is optimized around this choice.
    use_fixed_clonesize_t1: `bool`, optional (default: False)
        If true, fix the number of initial states as the same for all clones
    sort_clone: `int`, optional (default: 1)
        The order to infer initial states for each clone: {1,-1,others}.
        1, sort clones by size from small to large;
        -1, sort clones by size from large to small;
        others, do not sort.
    compute_new: `bool`, optional (default: False)
        If True, compute everything (ShortestPathDis, OT_map, etc.) from scratch,
        whether it was computed and saved before or not.
    save_subset: `bool`, optional (default: True)
        If true, save only Smatrix at smooth round [5,10,15,...];
        Otherwise, save Smatrix at each round.

    Returns
    ------
    None. Update adata.obsm['X_clone'] and adata.uns['transition_map'],
    as well as adata.uns['OT_transition_map'] or
    adata.uns['HighVar_transition_map'], depending on the initialization.
    """

    # I found the error: 1) we should use clonally related cell number at t2 as a factor to determine the clonally cell number at t1
    #                    2) update the whole t2 clonal info at once

    # logg.info("Joint optimization that consider possibility of clonal overlap")

    cell_id_array_t1 = adata.uns["Tmap_cell_id_t1"]
    cell_id_array_t2 = adata.uns["Tmap_cell_id_t2"]
    data_des = adata.uns["data_des"][-1]
    data_path = settings.data_path
    X_clone = adata.obsm["X_clone"]
    if not ssp.issparse(X_clone):
        X_clone = ssp.csr_matrix(X_clone)

    time_info = np.array(adata.obs["time_info"])
    time_index_t1 = time_info == (time_info[cell_id_array_t1[0]])
    time_index_t2 = time_info == (time_info[cell_id_array_t2[0]])

    if not ssp.issparse(initialized_map):
        map_temp = ssp.csr_matrix(initialized_map)
    else:
        map_temp = initialized_map

    # a clone must has at least 2 cells, to be updated later.
    valid_clone_id = np.nonzero(X_clone[cell_id_array_t2].sum(0).A.flatten() > 0)[0]
    X_clone_temp = X_clone[:, valid_clone_id]
    clonal_cells_t2 = np.sum(X_clone_temp[cell_id_array_t2].sum(1).flatten())

    logg.hint(f"original clone shape: {X_clone.shape}")
    logg.hint(f"After excluding zero-sized clones at t2: {X_clone_temp.shape}")

    flag = True  # to check whether overlapping clones are found or not
    if use_fixed_clonesize_t1:
        logg.hint("Use fixed clone size at t1")

    ##### Partition cells into non-overlapping, combinatorial BC_id.
    # ---------------------------------
    # find the combinatorial barcodes
    clone_idx = np.nonzero(X_clone_temp.A)
    dic = [[] for j in range(X_clone_temp.shape[0])]  # a list of list
    for j in range(clone_idx[0].shape[0]):
        dic[clone_idx[0][j]].append(clone_idx[1][j])

    BC_id = [
        tuple(x) for x in dic
    ]  # a BC_id is a unique barcode combination, does not change the ordering of cells

    # --------------------
    # construct the new X_clone_temp matrix, and the clone_mapping
    unique_BC_id = list(set(BC_id))
    if () in unique_BC_id:  # () is resulted from cells without any barcodes
        unique_BC_id.remove(())

    # construct a X_clone_newBC for the new BC_id
    # also record how the new BC_id is related to the old barcode

    X_clone_newBC = np.zeros((X_clone_temp.shape[0], len(unique_BC_id)))
    for i, BC_0 in enumerate(BC_id):
        for j, BC_1 in enumerate(unique_BC_id):
            if BC_1 == BC_0:
                X_clone_newBC[i, j] = 1  # does not change the ordering of cells

    clone_mapping = np.zeros((X_clone_temp.shape[1], X_clone_newBC.shape[1]))
    for j, BC_1 in enumerate(unique_BC_id):
        for k in BC_1:
            clone_mapping[k, j] = 1

    X_clone_newBC = ssp.csr_matrix(X_clone_newBC)
    clone_mapping = ssp.csr_matrix(clone_mapping)
    # To recover the original X_clone_temp, use 'X_clone_newBC*(clone_mapping.T)'
    # howver, clone_mapping is not invertible. We cannot get from X_clone_temp to
    # X_clone_newBC using matrix multiplification.

    ### select the early states using the grouped distribution of a clone
    ### clones are not overlapping, and all early states should be attached to clones at the end

    # we sort clones according to their sizes. The order of cells are not affected. So, it should not affect downstream analysis
    # small clones tend to be the ones that are barcoded/mutated later, while large clones tend to be early mutations...
    clone_size_t2_temp = X_clone_newBC[cell_id_array_t2].sum(0).A.flatten()

    if sort_clone == 1:
        logg.hint("Sort clones by size (small to large)")

        sort_clone_id = np.argsort(clone_size_t2_temp, kind="stable")
        clone_size_t2 = clone_size_t2_temp[sort_clone_id]
        X_clone_sort = X_clone_newBC[:, sort_clone_id]
        clone_mapping_sort = clone_mapping[:, sort_clone_id]

    elif sort_clone == -1:
        logg.hint("Sort clones by size (large to small)")

        sort_clone_id = np.argsort(clone_size_t2_temp, kind="stable")[::-1]
        clone_size_t2 = clone_size_t2_temp[sort_clone_id]
        X_clone_sort = X_clone_newBC[:, sort_clone_id]
        clone_mapping_sort = clone_mapping[:, sort_clone_id]

    else:
        logg.hint("Do not order clones by size ")
        clone_size_t2 = clone_size_t2_temp
        X_clone_sort = X_clone_newBC
        clone_mapping_sort = clone_mapping

    logg.hint("Infer the number of initial cells to extract for each clone in advance")
    clone_N1 = X_clone_sort.shape[1]
    ave_clone_size_t1 = int(np.ceil(len(cell_id_array_t1) / clone_N1))
    cum_cell_N = np.ceil(
        np.cumsum(clone_size_t2) * len(cell_id_array_t1) / clonal_cells_t2
    )
    cell_N_to_extract = np.zeros(len(cum_cell_N), dtype=int)
    if use_fixed_clonesize_t1:
        cell_N_to_extract += ave_clone_size_t1
    else:
        cell_N_to_extract[0] = cum_cell_N[0]
        cell_N_to_extract[1:] = np.diff(cum_cell_N)

    for x0 in range(max_iter_N[0]):
        logg.info(f"-----JointOpt Iteration {x0+1}: Infer initial clonal structure")

        # update initial state probability matrix based on the current map
        initial_prob_matrix = (
            map_temp * X_clone_sort[cell_id_array_t2]
        ).A  # a initial probability matrix for t1 cells, shape (n_t1_cell,n_clone)

        ########## begin: update clones
        remaining_ids_t1 = list(np.arange(len(cell_id_array_t1), dtype=int))

        X_clone_new = np.zeros(X_clone_sort.shape, dtype=bool)
        X_clone_new[cell_id_array_t2] = X_clone_sort[cell_id_array_t2].A.astype(
            bool
        )  # update the whole t2 clones at once

        for j in range(clone_N1):
            if j % 100 == 0:
                # pdb.set_trace()
                logg.hint(f"Inferring early clonal states: current clone id {j}")

            # infer the earlier clonal states for each clone
            ### select the early states using the grouped distribution of a clone
            sorted_id_array = np.argsort(
                initial_prob_matrix[remaining_ids_t1, j], kind="stable"
            )[::-1]

            sel_id_t1 = sorted_id_array[: cell_N_to_extract[j]]
            temp_t1_idx = np.zeros(len(cell_id_array_t1), dtype=bool)
            temp_t1_idx[np.array(remaining_ids_t1)[sel_id_t1]] = True
            X_clone_new[cell_id_array_t1, j] = temp_t1_idx
            for kk in np.array(remaining_ids_t1)[sel_id_t1]:
                remaining_ids_t1.remove(kk)

            if (len(remaining_ids_t1) == 0) and ((j + 1) < clone_N1):
                logg.hint(f"Early break; current clone id: {j+1}")
                break

        ########### end: update clones
        cell_id_array_t1_new = np.nonzero((X_clone_new.sum(1) > 0) & (time_index_t1))[0]
        cell_id_array_t2_new = np.nonzero((X_clone_new.sum(1) > 0) & (time_index_t2))[0]

        adata.obsm["X_clone"] = ssp.csr_matrix(X_clone_new) * (
            clone_mapping_sort.T
        )  # convert back to the original clone structure
        adata.uns["multiTime_cell_id_t1"] = [
            cell_id_array_t1_new
        ]  # For CoSpar, clonally-related states
        adata.uns["multiTime_cell_id_t2"] = [cell_id_array_t2_new]
        adata.uns[
            "clonal_cell_id_t1"
        ] = cell_id_array_t1_new  # for prepare the similarity matrix with same cell states
        adata.uns["clonal_cell_id_t2"] = cell_id_array_t2_new
        adata.uns["proportion"] = [1]

        logg.info(
            f"-----JointOpt Iteration {x0+1}: Update the transition map by CoSpar"
        )
        infer_Tmap_from_multitime_clones_private(
            adata,
            smooth_array=smooth_array,
            neighbor_N=CoSpar_KNN,
            sparsity_threshold=sparsity_threshold,
            normalization_mode=normalization_mode,
            save_subset=save_subset,
            use_full_Smatrix=use_full_Smatrix,
            trunca_threshold=trunca_threshold,
            compute_new_Smatrix=compute_new,
            max_iter_N=max_iter_N[1],
            epsilon_converge=epsilon_converge[1],
        )

        # update, for the next iteration
        if "transition_map" in adata.uns.keys():

            # sample cell states to perform the accuracy test
            sample_N_x = 50
            sample_N_y = 100
            t0 = time.time()
            cell_N_tot_x = map_temp.shape[0]
            if cell_N_tot_x < sample_N_x:
                sample_id_temp_x = np.arange(cell_N_tot_x)
            else:
                xx = np.arange(cell_N_tot_x)
                yy = (
                    list(np.nonzero(xx % 3 == 0)[0])
                    + list(np.nonzero(xx % 3 == 1)[0])
                    + list(np.nonzero(xx % 3 == 2)[0])
                )
                sample_id_temp_x = yy[:sample_N_x]

            cell_N_tot_y = map_temp.shape[1]
            if cell_N_tot_y < sample_N_y:
                sample_id_temp_y = np.arange(cell_N_tot_y)
            else:
                xx = np.arange(cell_N_tot_y)
                yy = (
                    list(np.nonzero(xx % 3 == 0)[0])
                    + list(np.nonzero(xx % 3 == 1)[0])
                    + list(np.nonzero(xx % 3 == 2)[0])
                )
                sample_id_temp_y = yy[:sample_N_y]

            if x0 == 0:
                X_map_0 = map_temp[sample_id_temp_x, :][:, sample_id_temp_y].A
            else:
                X_map_0 = X_map_1.copy()

            X_map_1 = adata.uns["transition_map"][sample_id_temp_x, :][
                :, sample_id_temp_y
            ].A

            verbose = logg._settings_verbosity_greater_or_equal_than(3)
            corr_X = np.diag(hf.corr2_coeff(X_map_0, X_map_1)).mean()
            if verbose:
                from matplotlib import pyplot as plt

                fig = plt.figure()
                ax = plt.subplot(1, 1, 1)
                ax.plot(X_map_0.flatten(), X_map_1.flatten(), ".r")
                ax.set_xlabel("$T_{ij}$: previous iteration")
                ax.set_ylabel("$T_{ij}$: current iteration")
                ax.set_title(f"Joint Opt., iter_N={x0+1}, R={int(100*corr_X)/100}")
                plt.show()
            else:
                # logg.info(f"Convergence (JointOpt, iter_N={x0+1}): corr(previous_T, current_T)={int(1000*corr_X)/1000}; cost time={time.time()-t0}")
                logg.info(
                    f"Convergence (JointOpt, iter_N={x0+1}): corr(previous_T, current_T)={int(1000*corr_X)/1000}"
                )

            if abs(1 - corr_X) < epsilon_converge[0]:
                break

            map_temp = adata.uns["transition_map"]
        else:
            raise ValueError(
                "transition_map not updated in infer_Tmap_from_multitime_clones_private."
            )


def infer_Tmap_from_HighVar(
    adata,
    min_counts=3,
    min_cells=3,
    min_gene_vscore_pctl=85,
    smooth_array=[15, 10, 5],
    neighbor_N=20,
    sparsity_threshold=0.2,
    normalization_mode=1,
    use_full_Smatrix=True,
    trunca_threshold=[0.001, 0.01],
    compute_new_Smatrix=True,
    max_iter_N=5,
    epsilon_converge=0.05,
    save_subset=True,
):
    """
    Generate Tmap based on state information using HighVar.

    We convert differentially expressed genes into `pseudo-clones`,
    and run coherent sparsity optimization to infer the transition map.
    Each clone occupies a different set of cells.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assumed to be preprocessed, only has two time points.
    min_counts: int, optional (default: 3)
        Minimum number of UMIs per cell to be considered for selecting highly variable genes.
    min_cells: int, optional (default: 3)
        Minimum number of cells per gene to be considered for selecting highly variable genes.
    min_gene_vscore_pctl: int, optional (default: 85)
        Genes with a variability percentile higher than this threshold are marked as
        highly variable genes for constructing pseudo-clones. Range: [0,100].
    smooth_array: `list`, optional (default: [15,10,5])
        List of smooth rounds at initial runs of iteration.
        Suppose that it has a length N. For iteration n<N, the n-th entry of
        smooth_array determines the kernel exponent to build the S matrix at the n-th
        iteration. When n>N, we use the last entry of smooth_array to compute
        the S matrix. We recommend starting with more smoothing depth and gradually
        reduce the depth, as inspired by simulated annealing. Data with higher
        clonal dispersion should start with higher smoothing depth. The final depth should
        depend on the manifold itself. For fewer cells, it results in a small KNN graph,
        and a small final depth should be used. We recommend to use a number at
        the multiple of 5 for computational efficiency i.e.,
        smooth_array=[20, 15, 10, 5], or [20,15,10]
    max_iter_N: `int`, optional (default: 5)
        The maximum iterations used to compute the transition map,
        regardless of epsilon_converge.
    epsilon_converge: `float`, optional (default: 0.05)
        The convergence threshold for the change of map
        correlations between consecutive iterations.
        This convergence test is activated only when
        CoSpar has iterated for 3 times.
    neighbor_N: `int`, optional (default: 20)
        The number of neighbors for KNN graph used for computing the similarity matrix.
    trunca_threshold: `list`, optional (default: [0.001,0.01])
        Threshold to reset entries of a matrix to zero. The first entry is for
        Similarity matrix; the second entry is for the Tmap.
        This is only for computational and storage efficiency.
    sparsity_threshold: `float`, optional (default: 0.1)
        The relative threshold to remove noises in the updated transition map,
        in the range [0,1].
    normalization_mode: `int`, optional (default: 1)
        Normalization method. Choice: [0,1].
        0, single-cell normalization; 1, Clone normalization. The clonal
        normalization suppresses the contribution of large
        clones, and is much more robust.
    save_subset: `bool`, optional (default: True)
        If true, save only Smatrix at smooth round [5,10,15,...];
        Otherwise, save Smatrix at each round.
    use_full_Smatrix: `bool`, optional (default: True)
        If true, extract the relevant Smatrix from the full Smatrix defined by all cells.
        This tends to be more accurate. The package is optimized around this choice.
    compute_new_Smatrix: `bool`, optional (default: False)
        If True, compute Smatrix from scratch, whether it was
        computed and saved before or not.

    Returns
    -------
    None. Results are stored at adata.uns['HighVar_transition_map'].
    """

    # logg.info("HighVar-v0: avoid cells that have been selected")
    weight = 1  # wehight of each gene.

    cell_id_array_t1 = adata.uns["Tmap_cell_id_t1"]
    cell_id_array_t2 = adata.uns["Tmap_cell_id_t2"]
    real_clone_annot = adata.obsm["X_clone"]

    time_info = np.array(adata.obs["time_info"])
    selected_time_points = [
        time_info[cell_id_array_t1][0],
        time_info[cell_id_array_t2][0],
    ]

    # logg.info("----------------")
    logg.info("Step a: find the commonly shared highly variable genes------")
    adata_t1 = sc.AnnData(adata.X[cell_id_array_t1])
    adata_t2 = sc.AnnData(adata.X[cell_id_array_t2])

    ## use marker genes
    gene_list = adata.var_names

    verbose = logg._settings_verbosity_greater_or_equal_than(3)

    ## the scanpy version
    # def filter_gene_custom(adata_tmp, gene_list_tmp):
    #     sc.pp.normalize_total(adata_tmp, target_sum=1e4)
    #     sc.pp.log1p(adata_tmp)
    #     sc.pp.highly_variable_genes(
    #         adata_tmp, min_mean=0.0125, max_mean=3, min_disp=0.5
    #     )
    #     return list(gene_list_tmp[adata_tmp.var["highly_variable"]])
    # highvar_genes_t1 = filter_gene_custom(adata_t1, gene_list)
    # highvar_genes_t2 = filter_gene_custom(adata_t2, gene_list)

    gene_idx_t1 = hf.filter_genes(
        adata_t1.X,
        min_counts=min_counts,
        min_cells=min_cells,
        min_vscore_pctl=min_gene_vscore_pctl,
        show_vscore_plot=verbose,
    )
    if gene_idx_t1 is not None:
        highvar_genes_t1 = gene_list[gene_idx_t1]
    else:
        raise ValueError("No variable genes selected at t1")

    gene_idx_t2 = hf.filter_genes(
        adata_t2.X,
        min_counts=min_counts,
        min_cells=min_cells,
        min_vscore_pctl=min_gene_vscore_pctl,
        show_vscore_plot=verbose,
    )
    if gene_idx_t2 is not None:
        highvar_genes_t2 = gene_list[gene_idx_t2]
    else:
        raise ValueError("No variable genes selected at t2")

    common_gene = sorted(list(set(highvar_genes_t1).intersection(highvar_genes_t2)))

    logg.info(
        f"Highly varable gene number: {len(highvar_genes_t1)} (t1); {len(highvar_genes_t2)} (t2). Common set: {len(common_gene)}"
    )

    # logg.info("----------------")
    logg.info("Step b: convert the shared highly variable genes into clonal info------")

    sel_marker_gene_list = common_gene.copy()
    clone_annot_gene = np.zeros((adata.shape[0], len(sel_marker_gene_list)))
    N_t1 = len(cell_id_array_t1)
    N_t2 = len(cell_id_array_t2)
    cumu_sel_idx_t1 = np.zeros(N_t1, dtype=bool)
    cumu_sel_idx_t2 = np.zeros(N_t2, dtype=bool)
    cell_fraction_per_gene = 1 / len(
        sel_marker_gene_list
    )  # fraction of cells as clonally related by this gene
    cutoff_t1 = int(np.ceil(len(cell_id_array_t1) * cell_fraction_per_gene))
    cutoff_t2 = int(np.ceil(len(cell_id_array_t2) * cell_fraction_per_gene))
    gene_exp_matrix = adata[:, sel_marker_gene_list].X.A
    for j in tqdm(range(gene_exp_matrix.shape[1])):
        temp_t1 = gene_exp_matrix[:, j][cell_id_array_t1]
        temp_t1[cumu_sel_idx_t1] = 0  # set selected cell id to have zero expression
        sel_id_t1 = np.argsort(temp_t1, kind="stable")[::-1][:cutoff_t1]
        clone_annot_gene[cell_id_array_t1[sel_id_t1], j] = weight
        cumu_sel_idx_t1[sel_id_t1] = True

        temp_t2 = gene_exp_matrix[:, j][cell_id_array_t2]
        temp_t2[cumu_sel_idx_t2] = 0  # set selected cell id to have zero expression
        sel_id_t2 = np.argsort(temp_t2, kind="stable")[::-1][:cutoff_t2]
        clone_annot_gene[cell_id_array_t2[sel_id_t2], j] = weight
        cumu_sel_idx_t2[sel_id_t2] = True

        if (np.sum(~cumu_sel_idx_t1) == 0) or (np.sum(~cumu_sel_idx_t2) == 0):
            logg.info(f"Total used genes={j} (no cells left)")
            break

    logg.info(
        "Step c: compute the transition map based on clonal info from highly variable genes------"
    )

    adata.obsm["X_clone"] = ssp.csr_matrix(clone_annot_gene)
    adata.uns["multiTime_cell_id_t1"] = [cell_id_array_t1]
    adata.uns["multiTime_cell_id_t2"] = [cell_id_array_t2]
    adata.uns["proportion"] = [1]

    infer_Tmap_from_multitime_clones_private(
        adata,
        smooth_array=smooth_array,
        neighbor_N=neighbor_N,
        sparsity_threshold=sparsity_threshold,
        normalization_mode=normalization_mode,
        save_subset=save_subset,
        use_full_Smatrix=use_full_Smatrix,
        trunca_threshold=trunca_threshold,
        compute_new_Smatrix=compute_new_Smatrix,
        max_iter_N=max_iter_N,
        epsilon_converge=epsilon_converge,
    )

    adata.uns["HighVar_transition_map"] = adata.uns["transition_map"]
    adata.obsm[
        "X_clone"
    ] = real_clone_annot  # This entry has been changed previously. Note correct the clonal matrix
    # data_des_1=data_des_0+'_HighVar1' # to record which initialization is used
    # adata.uns['data_des']=[data_des_orig,data_des_1]

    if "Smatrix" in adata.uns.keys():
        adata.uns.pop("Smatrix")


# this is the new version: v1, finally used
def infer_Tmap_from_optimal_transport(
    adata,
    OT_epsilon=0.02,
    OT_dis_KNN=5,
    OT_solver="duality_gap",
    OT_cost="SPD",
    compute_new=True,
    use_existing_KNN_graph=False,
):
    """
    Compute Tmap from state info using optimal transport (OT).

    We provide the options for the OT solver, and also the cost function.
    The OT solver does not seem to matter, although 'duality_gap' is faster.
    The cost function could affect the OT map results. Using shortest path
    distance ('SPD') is slower but more accurate, while using gene expression
    distance ('GED') is faster but less accurate. The performance of cospar
    is robust to the initialized map (this is especially so in terms of fate
    bias, not so much for the fate map alone)

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assumed to be preprocessed, only has two time points.
    OT_epsilon: `float`, optional (default: 0.02)
        The entropic regularization, >0. A larger value increases
        uncertainty of the transition.
    OT_dis_KNN: `int`, optional (default: 5)
        Number of nearest neighbors to construct the KNN graph for
        computing the shortest path distance.
    OT_solver: `str`, optional (default: `duality_gap`)
        The method used to compute the optimal transport map. Available choices:
        {'duality_gap','fixed_iters'}. Our test shows that they produce the same
        results, while 'duality_gap' is almost twice faster.
    OT_cost: `str`, optional (default: `SPD`), options {'GED','SPD'}
        The cost metric. We provide gene expression distance (GED), and also
        shortest path distance (SPD). GED is much faster, but SPD is more accurate.
        However, coherent sparsity optimization is robust to the initialization.
    compute_new: `bool`, optional (default: False)
        If True, compute OT_map and also the shortest path distance from scratch,
        whether it was computed and saved before or not.
    use_existing_KNN_graph: `bool`, optional (default: False)
        If true and adata.obsp['connectivities'], use the existing knn graph for constructing
        the shortest-path distance. This overrides all other parameters.

    Returns
    -------
    None. Results are stored at adata.uns['OT_transition_map'].
    """

    cell_id_array_t1 = adata.uns["Tmap_cell_id_t1"]
    cell_id_array_t2 = adata.uns["Tmap_cell_id_t2"]
    data_des = adata.uns["data_des"][-1]
    data_path = settings.data_path

    ######## apply optimal transport
    CustomOT_file_name = os.path.join(
        data_path,
        f"{data_des}_CustomOT_map_epsilon{OT_epsilon}_KNN{OT_dis_KNN}_OTcost{OT_cost}.npz",
    )
    if os.path.exists(CustomOT_file_name) and (not compute_new):

        logg.info("Load pre-computed custom OT matrix")
        OT_transition_map = ssp.load_npz(CustomOT_file_name)

        ## add dimensionality check
        if (OT_transition_map.shape[0] != len(cell_id_array_t1)) or (
            OT_transition_map.shape[1] != len(cell_id_array_t2)
        ):
            raise ValueError(
                "The pre-computed OT transition map does not have the right dimension.\n"
                "Possible reason: this is computed from a different dataset, but with the same data label in adata.uns['data_des']\n"
                "You can fix this issue by running the Tmap inference with compute_new=True, which compute everything from scratch.\n"
                "To avoid this from happening again, please use a different data_des for each new data, which can be set with adata.uns['data_des']=['Your_data_des']."
            )

    else:

        ############ Compute shorted-path distance
        # use sklearn KNN graph construction method and select the connectivity option, not related to UMAP
        # use the mode 'distance' to obtain the shortest-path *distance*, rather than 'connectivity'
        if OT_cost == "SPD":
            SPD_file_name = os.path.join(
                data_path,
                f"{data_des}_ShortestPathDistanceMatrix_t0t1_KNN{OT_dis_KNN}.npy",
            )
            if os.path.exists(SPD_file_name) and (not compute_new):

                logg.info("Load pre-computed shortest path distance matrix")
                OT_cost_matrix = np.load(SPD_file_name)

                ## add dimensionality check
                if (OT_cost_matrix.shape[0] != len(cell_id_array_t1)) or (
                    OT_cost_matrix.shape[1] != len(cell_id_array_t2)
                ):
                    raise ValueError(
                        "The pre-computed OT cost matrix does not have the right dimension.\n"
                        "Possible reason: this is computed from a different dataset, but with the same data label in adata.uns['data_des']\n"
                        "You can fix this issue by running the Tmap inference with compute_new=True, which compute everything from scratch.\n"
                        "To avoid this from happening again, please use a different data_des for each new data, which can be set with adata.uns['data_des']=['Your_data_des']."
                    )

            else:

                logg.info("Compute new shortest path distance matrix")
                t = time.time()
                # data_matrix=adata.obsm['X_pca']
                # ShortPath_dis=hf.compute_shortest_path_distance_from_raw_matrix(data_matrix,num_neighbors_target=OT_dis_KNN,mode='distance')
                ShortPath_dis = hf.compute_shortest_path_distance(
                    adata,
                    num_neighbors_target=OT_dis_KNN,
                    mode="distances",
                    method="umap",
                    use_existing_KNN_graph=use_existing_KNN_graph,
                )

                idx0 = cell_id_array_t1
                idx1 = cell_id_array_t2
                ShortPath_dis_t0t1 = ShortPath_dis[idx0[:, np.newaxis], idx1]
                OT_cost_matrix = ShortPath_dis_t0t1 / ShortPath_dis_t0t1.max()

                np.save(
                    SPD_file_name, OT_cost_matrix
                )  # This is not a sparse matrix at all.

                logg.info(
                    f"Finishing computing shortest-path distance, used time {time.time()-t}"
                )
        else:
            t = time.time()
            pc_n = adata.obsm["X_pca"].shape[1]
            OT_cost_matrix = hf.compute_gene_exp_distance(
                adata, cell_id_array_t1, cell_id_array_t2, pc_n=pc_n
            )
            logg.info(
                f"Finishing computing gene expression distance, used time {time.time()-t}"
            )

        ##################
        logg.info("Compute new custom OT matrix")
        t = time.time()

        if "cell_growth_rate" not in adata.obs.keys():
            logg.info("Use uniform growth rate")
            adata.obs["cell_growth_rate"] = np.ones(adata.shape[0])
        else:
            logg.info("Use pre-supplied cell_grwoth_rate")
            x0 = np.mean(adata.obs["cell_growth_rate"])
            y0 = np.std(adata.obs["cell_growth_rate"])
            print(f"Use pre-supplied cell_grwoth_rate (mean: {x0:.2f}; std {y0:.2f})")

        #############
        OT_solver = "duality_gap"
        logg.info(f"OT solver: {OT_solver}")
        if (
            OT_solver == "fixed_iters"
        ):  # This takes 50s for the subsampled hematopoietic data. The result is the same.
            ot_config = {
                "C": OT_cost_matrix,
                "G": np.array(adata.obs["cell_growth_rate"])[cell_id_array_t1],
                "epsilon": OT_epsilon,
                "lambda1": 1,
                "lambda2": 50,
                "epsilon0": 1,
                "scaling_iter": 3000,
                "tau": 10000,
                "inner_iter_max": 50,
                "extra_iter": 1000,
            }

            OT_transition_map = transport_stablev2(**ot_config)

        elif (
            OT_solver == "duality_gap"
        ):  # This takes 30s for the subsampled hematopoietic data. The result is the same.
            ot_config = {
                "C": OT_cost_matrix,
                "G": np.array(adata.obs["cell_growth_rate"])[cell_id_array_t1],
                "epsilon": OT_epsilon,
                "lambda1": 1,
                "lambda2": 50,
                "epsilon0": 1,
                "tau": 10000,
                "tolerance": 1e-08,
                "max_iter": 1e7,
                "batch_size": 5,
            }

            OT_transition_map = optimal_transport_duality_gap(**ot_config)

        else:
            raise ValueError("Unknown solver")

        OT_transition_map = hf.matrix_row_or_column_thresholding(
            OT_transition_map, threshold=0.01
        )
        if not ssp.issparse(OT_transition_map):
            OT_transition_map = ssp.csr_matrix(OT_transition_map)
        # ssp.save_npz(CustomOT_file_name, OT_transition_map)

        logg.info(
            f"Finishing computing optial transport map, used time {time.time()-t}"
        )

    adata.uns["OT_transition_map"] = OT_transition_map
