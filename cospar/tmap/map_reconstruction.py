# -*- coding: utf-8 -*-

import os
import time

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as ssp

from cospar.tmap import _tmap_core as tmap_core
from cospar.tmap import _utils as tmap_util

from .. import help_functions as hf
from .. import logging as logg
from .. import settings
from .. import tool as tl


# v1 version, allows to set later time point
def infer_Tmap_from_multitime_clones(
    adata_orig,
    clonal_time_points=None,
    later_time_point=None,
    smooth_array=[15, 10, 5],
    CoSpar_KNN=20,
    sparsity_threshold=0.1,
    intraclone_threshold=0.05,
    normalization_mode=1,
    extend_Tmap_space=False,
    save_subset=True,
    use_full_Smatrix=True,
    trunca_threshold=[0.001, 0.01],
    compute_new=False,
    max_iter_N=5,
    epsilon_converge=0.05,
):
    """
    Infer transition map for clonal data with multiple time points.

    It prepares adata object for cells of targeted time points by
    :func:`cospar.tmap._utils.select_time_points`, generates the similarity matrix
    via :func:`cospar.tmap._utils.generate_similarity_matrix`, and iteratively calls
    the core function :func:`.refine_Tmap_through_cospar` to update
    the transition map.

    * If `later_time_point=None`, the inferred map allows transitions
      between neighboring time points. For example, if
      clonal_time_points=['day1','day2','day3'], then it computes transitions
      for pairs (day1, day2) and (day2, day3), but not (day1, day3).

    * If `later_time_point` is specified, the function produces a map
      between earlier time points and this later time point. For example, if
      `later_time_point='day3`, the map allows transitions for pairs (day1, day3)
      and (day2, day3), but not (day1,day2).

    Parameters
    ------------
    adata_orig: :class:`~anndata.AnnData` object
        Should be prepared from our anadata initialization.
    clonal_time_points: `list` of `str`, optional (default: all time points)
        List of time points to be included for analysis.
        We assume that each selected time point has clonal measurements.
    later_time_points: `list`, optional (default: None)
        If specified, the function will produce a map T between these early
        time points among `clonal_time_points` and the `later_time_point`.
        If not specified, it produces a map T between neighboring clonal time points.
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
    CoSpar_KNN: `int`, optional (default: 20)
        The number of neighbors for KNN graph used for computing the
        similarity matrix.
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
    extend_Tmap_space: `bool` optional (default: `False`)
        If true, the initial states for Tmap will include all states at initial time points,
        and the later states for Tmap will include all states at later time points.
        Otherwise, the initial and later state space of the Tmap will be
        restricted to cells with multi-time clonal information
        alone. The latter case speeds up the computation, which is recommended.
        This option is ignored when `later_time_points` is not None.
    save_subset: `bool`, optional (default: True)
        If true, save only Smatrix at smooth round [5,10,15,...];
        Otherwise, save Smatrix at each round.
    use_full_Smatrix: `bool`, optional (default: True)
        If true, extract the relevant Smatrix from the full Smatrix defined by all cells.
        This tends to be more accurate. The package is optimized around this choice.
    Compute_new: `bool`, optional (default: False)
        If True, compute Smatrix from scratch, whether it was
        computed and saved before or not. This is activated only when
        `use_full_Smatrix=False`.

    Returns
    -------
    adata: :class:`~anndata.AnnData` object
        Store results at adata.uns['transition_map']
        and adata.uns['intraclone_transition_map']. This adata is different
        from the input adata_orig due to subsampling cells.
    """

    t0 = time.time()
    hf.check_available_clonal_info(adata_orig)
    clonal_time_points_0 = np.array(adata_orig.uns["clonal_time_points"])
    if len(clonal_time_points_0) < 2:
        raise ValueError("There are no multi-time clones. Abort the inference.")

    if clonal_time_points is None:
        clonal_time_points = clonal_time_points_0

    if type(later_time_point) == list:
        later_time_point = later_time_point[0]

    if later_time_point is not None:
        clonal_time_points = list(clonal_time_points) + [later_time_point]
        clonal_time_points = list(set(clonal_time_points))

    hf.check_input_parameters(
        adata_orig,
        later_time_point=later_time_point,
        clonal_time_points=clonal_time_points,
        smooth_array=smooth_array,
        save_subset=save_subset,
    )
    # order the clonal time points
    time_ordering = adata_orig.uns["time_ordering"]
    sel_idx_temp = np.in1d(time_ordering, clonal_time_points)
    clonal_time_points = time_ordering[sel_idx_temp]

    logg.info("------Compute the full Similarity matrix if necessary------")
    data_path = settings.data_path
    if (
        use_full_Smatrix
    ):  # prepare the similarity matrix with all state info, all subsequent similarity will be down-sampled from this one.

        temp_str = "0" + str(trunca_threshold[0])[2:]
        round_of_smooth = np.max(smooth_array)
        data_des = adata_orig.uns["data_des"][0]
        similarity_file_name = os.path.join(
            data_path,
            f"{data_des}_Similarity_matrix_with_all_cell_states_kNN{CoSpar_KNN}_Truncate{temp_str}",
        )
        if not (
            os.path.exists(similarity_file_name + f"_SM{round_of_smooth}.npz")
            and (not compute_new)
        ):
            similarity_matrix_full = tmap_util.generate_similarity_matrix(
                adata_orig,
                similarity_file_name,
                round_of_smooth=round_of_smooth,
                neighbor_N=CoSpar_KNN,
                truncation_threshold=trunca_threshold[0],
                save_subset=save_subset,
                compute_new_Smatrix=compute_new,
            )

    # compute transition map between neighboring time points
    if later_time_point is None:
        logg.info("----Infer transition map between neighboring time points-----")
        logg.info("Step 1: Select time points")
        adata = tmap_util.select_time_points(
            adata_orig,
            time_point=clonal_time_points,
            extend_Tmap_space=extend_Tmap_space,
        )

        logg.info("Step 2: Optimize the transition map recursively")
        tmap_core.infer_Tmap_from_multitime_clones_private(
            adata,
            smooth_array=smooth_array,
            neighbor_N=CoSpar_KNN,
            sparsity_threshold=sparsity_threshold,
            intraclone_threshold=intraclone_threshold,
            normalization_mode=normalization_mode,
            save_subset=save_subset,
            use_full_Smatrix=use_full_Smatrix,
            trunca_threshold=trunca_threshold,
            compute_new_Smatrix=compute_new,
            max_iter_N=max_iter_N,
            epsilon_converge=epsilon_converge,
        )

        if "Smatrix" in adata.uns.keys():
            adata.uns.pop("Smatrix")

        logg.info(f"-----------Total used time: {time.time()-t0} s ------------")
        return adata

    else:
        # compute transition map between initial time points and the later time point
        sel_id = np.nonzero(np.in1d(clonal_time_points, later_time_point))[0][0]
        initial_time_points = clonal_time_points[:sel_id]

        time_info_orig = np.array(adata_orig.obs["time_info"])
        sp_idx = np.zeros(adata_orig.shape[0], dtype=bool)
        all_time_points = list(initial_time_points) + [later_time_point]
        label = "t"
        for xx in all_time_points:
            id_array = np.nonzero(time_info_orig == xx)[0]
            sp_idx[id_array] = True
            label = label + "*" + str(xx)

        adata = adata_orig[sp_idx]
        data_des_orig = adata_orig.uns["data_des"][0]
        data_des_0 = adata_orig.uns["data_des"][-1]
        data_des = (
            data_des_0
            + f"_MultiTimeClone_Later_FullSpace{int(extend_Tmap_space)}_{label}"
        )
        adata.uns["data_des"] = [data_des_orig, data_des]

        time_info = np.array(adata.obs["time_info"])
        time_index_t2 = time_info == later_time_point
        time_index_t1 = ~time_index_t2

        #### used for similarity matrix generation
        Tmap_cell_id_t1 = np.nonzero(time_index_t1)[0]
        Tmap_cell_id_t2 = np.nonzero(time_index_t2)[0]
        adata.uns["Tmap_cell_id_t1"] = Tmap_cell_id_t1
        adata.uns["Tmap_cell_id_t2"] = Tmap_cell_id_t2
        adata.uns["clonal_cell_id_t1"] = Tmap_cell_id_t1
        adata.uns["clonal_cell_id_t2"] = Tmap_cell_id_t2
        adata.uns["sp_idx"] = sp_idx
        data_path = settings.data_path

        transition_map = np.zeros((len(Tmap_cell_id_t1), len(Tmap_cell_id_t2)))
        intraclone_transition_map = np.zeros(
            (len(Tmap_cell_id_t1), len(Tmap_cell_id_t2))
        )

        logg.info(
            "------Infer transition map between initial time points and the later time one------"
        )
        for yy in initial_time_points:

            logg.info(f"--------Current initial time point: {yy}--------")

            logg.info("Step 1: Select time points")
            adata_temp = tmap_util.select_time_points(
                adata_orig, time_point=[yy, later_time_point], extend_Tmap_space=True
            )  # for this to work, we need to set extend_Tmap_space=True, otherwise for different initial time points, the later Tmap_cell_id_t2 may be different

            logg.info("Step 2: Optimize the transition map recursively")
            tmap_core.infer_Tmap_from_multitime_clones_private(
                adata_temp,
                smooth_array=smooth_array,
                neighbor_N=CoSpar_KNN,
                sparsity_threshold=sparsity_threshold,
                intraclone_threshold=intraclone_threshold,
                normalization_mode=normalization_mode,
                save_subset=save_subset,
                use_full_Smatrix=use_full_Smatrix,
                trunca_threshold=trunca_threshold,
                compute_new_Smatrix=compute_new,
                max_iter_N=max_iter_N,
                epsilon_converge=epsilon_converge,
            )

            temp_id_t1 = np.nonzero(time_info == yy)[0]
            sp_id_t1 = hf.converting_id_from_fullSpace_to_subSpace(
                temp_id_t1, Tmap_cell_id_t1
            )[0]

            transition_map[sp_id_t1, :] = adata_temp.uns["transition_map"].A
            intraclone_transition_map[sp_id_t1, :] = adata_temp.uns[
                "intraclone_transition_map"
            ].A

            if "Smatrix" in adata_temp.uns.keys():
                adata_temp.uns.pop("Smatrix")

        adata.uns["transition_map"] = ssp.csr_matrix(transition_map)
        adata.uns["intraclone_transition_map"] = ssp.csr_matrix(
            intraclone_transition_map
        )

        logg.info(f"-----------Total used time: {time.time()-t0} s ------------")
        return adata


def infer_intraclone_Tmap(adata, intraclone_threshold=0.05, normalization_mode=1):
    """
    Infer intra-clone transition map.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Should be prepared by :func:`cospar.tmap._utils.select_time_points`
    intraclone_threshold: `float`, optional (default: 0.05)
        The threshold to remove noises in the demultiplexed (un-smoothed) map,
        in the range [0,1]
    normalization_mode: `int`, optional (default: 1)
        Normalization method. Choice: [0,1].
        0, single-cell normalization; 1, Clone normalization. The clonal
        normalization suppresses the contribution of large
        clones, and is much more robust.

    Returns
    -------
    None. Update/generate adata.uns['intraclone_transition_map']

    """

    ########## extract data
    if "transition_map" not in adata.uns.keys():
        logg.error(
            "Please run ---- CS.tmap.infer_Tmap_from_multitime_clones ---- first"
        )

    else:

        clone_annot = adata.obsm["X_clone"]

        multiTime_cell_id_t1 = [adata.uns["Tmap_cell_id_t1"]]
        multiTime_cell_id_t2 = [adata.uns["Tmap_cell_id_t2"]]
        proportion = adata.uns["proportion"]

        transition_map = adata.uns["transition_map"]

        X_clone = clone_annot.copy()
        if not ssp.issparse(X_clone):
            X_clone = ssp.csr_matrix(X_clone)

        demultiplexed_map = tmap_core.refine_Tmap_through_cospar_noSmooth(
            multiTime_cell_id_t1,
            multiTime_cell_id_t2,
            proportion,
            transition_map,
            X_clone,
            sparsity_threshold=intraclone_threshold,
            normalization_mode=normalization_mode,
        )

        adata.uns["intraclone_transition_map"] = ssp.csr_matrix(demultiplexed_map)


def infer_Tmap_from_one_time_clones(
    adata_orig,
    initial_time_points=None,
    later_time_point=None,
    initialize_method="OT",
    OT_epsilon=0.02,
    OT_dis_KNN=5,
    OT_cost="SPD",
    HighVar_gene_pctl=85,
    padding_X_clone=False,
    normalization_mode=1,
    sparsity_threshold=0.2,
    CoSpar_KNN=20,
    use_full_Smatrix=True,
    smooth_array=[15, 10, 5],
    trunca_threshold=[0.001, 0.01],
    compute_new=False,
    max_iter_N=[1, 5],
    epsilon_converge=[0.05, 0.05],
    use_fixed_clonesize_t1=False,
    sort_clone=1,
    save_subset=True,
    use_existing_KNN_graph=False,
):
    """
    Infer transition map from clones with a single time point

    We jointly infer a transition map and the initial clonal observation
    through iteration. The inferred map is between each of the initial
    time points ['day_1','day_2',...,] and the time point with clonal
    observation. We initialize the transition map by either the OT
    method or HighVar method.

    **Summary**

    * Parameters relevant for cell state selection:  initial_time_points,
      later_time_point.

    * Initialization methods:

        * 'OT': optional transport based method. Key parameters: `OT_epsilon, OT_dis_KNN`.
          See :func:`.infer_Tmap_from_optimal_transport`.

        * 'HighVar': a customized approach, assuming that cells similar in gene
          expression across time points share clonal origin. Key parameter: `HighVar_gene_pctl`.
          See :func:`.infer_Tmap_from_HighVar`.

    * Key parameters relevant for joint optimization itself (which relies on coherent sparse optimization):
      `smooth_array, CoSpar_KNN, sparsity_threshold`. See :func:`.refine_Tmap_through_joint_optimization`.


    Parameters
    ----------
    adata_orig: :class:`~anndata.AnnData` object
        It is assumed to be preprocessed and has multiple time points.
    initial_time_points: `list`, optional (default, all time points)
        List of initial time points to be included for the transition map.
        Like ['day_1','day_2']. Entries consistent with adata.obs['time_info'].
    later_time_point: `str`, optional (default, the last time point)
        The time point with clonal observation. Its value should be
        consistent with adata.obs['time_info'].
    initialize_method: `str`, optional (default 'OT')
        Method to initialize the transition map from state information.
        Choice: {'OT', 'HighVar'}.
    OT_epsilon: `float`, optional (default: 0.02)
        The entropic regularization, >0. A larger value increases
        uncertainty of the transition. Relevant when `initialize_method='OT'`.
    OT_dis_KNN: `int`, optional (default: 5)
        Number of nearest neighbors to construct the KNN graph for
        computing the shortest path distance. Relevant when `initialize_method='OT'`.
    OT_cost: `str`, optional (default: `SPD`), options {'GED','SPD'}
        The cost metric. We provide gene expression distance (GED), and also
        shortest path distance (SPD). GED is much faster, but SPD is more accurate.
        However, cospar is robust to the initialization.
    HighVar_gene_pctl: `int`, optional (default: 85)
        Percentile threshold to select highly variable genes to construct pseudo-clones.
        A higher value selects more variable genes. Range: [0,100].
        Relevant when `initialize_method='HighVar'`.
    padding_X_clone: `bool`, optional (default: False)
        If true, select cells at the `later_time_point` yet without any clonal label, and
        generate a unique clonal label for each of them. This adds artificial clonal data.
        However, it will make the best use of the state information, especially when there
        are very few clonal barcodes in the data.
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
        whether it was computed and saved before or not. Regarding the Smatrix, it is
        recomputed only when `use_full_Smatrix=False`.
    use_existing_KNN_graph: `bool`, optional (default: False)
        If true and adata.obsp['connectivities'], use the existing knn graph
        to compute the shortest-path distance. Revelant if initialize_method='OT'.
        This overrides all other relevant parameters for building shortest-path distance.

    Returns
    -------
    adata: :class:`~anndata.AnnData` object
        Update adata.obsm['X_clone'] and adata.uns['transition_map'],
        as well as adata.uns['OT_transition_map'] or
        adata.uns['HighVar_transition_map'], depending on the initialization.
        adata_orig.obsm['X_clone'] remains the same.
    """

    t0 = time.time()
    hf.check_available_clonal_info(adata_orig)
    clonal_time_points_0 = np.array(adata_orig.uns["clonal_time_points"])
    time_ordering = adata_orig.uns["time_ordering"]
    if len(clonal_time_points_0) == 0:
        raise ValueError(
            "No clonal time points available for this dataset. Please run cs.tmap.infer_Tmap_from_state_info_alone."
        )

    if later_time_point is None:
        sel_idx_temp = np.in1d(time_ordering, clonal_time_points_0)
        later_time_point = time_ordering[sel_idx_temp][-1]

    if type(later_time_point) == list:
        later_time_point = later_time_point[0]

    # use the last clonal later time point

    if initial_time_points is None:
        sel_id_temp = np.nonzero(np.in1d(time_ordering, [later_time_point]))[0][0]
        initial_time_points = time_ordering[:sel_id_temp]

    sel_idx_temp = np.in1d(time_ordering, initial_time_points)
    initial_time_points = list(time_ordering[sel_idx_temp])
    if later_time_point in initial_time_points:
        logg.warn(f"remove {later_time_point} from initial_time_points")
        initial_time_points.remove(later_time_point)

    hf.check_input_parameters(
        adata_orig,
        later_time_point=later_time_point,
        initial_time_points=initial_time_points,
        smooth_array=smooth_array,
        save_subset=save_subset,
    )

    if initialize_method not in ["OT", "HighVar"]:
        logg.warn(
            "initialize_method not among ['OT','HighVar']. Use initialize_method='OT'"
        )
        initialize_method = "OT"

    if OT_cost not in ["GED", "SPD"]:
        logg.warn("OT_cost not among ['GED','SPD']. Use OT_cost='SPD'")
        OT_cost = "SPD"

    sp_idx = np.zeros(adata_orig.shape[0], dtype=bool)
    time_info_orig = np.array(adata_orig.obs["time_info"])
    all_time_points = list(initial_time_points) + [later_time_point]

    label = "t"
    for xx in all_time_points:
        id_array = np.nonzero(time_info_orig == xx)[0]
        sp_idx[id_array] = True
        label = label + "*" + str(xx)

    adata = adata_orig[sp_idx]

    clone_annot_orig = adata_orig.obsm["X_clone"].copy()
    data_des_orig = adata_orig.uns["data_des"][0]
    data_des_0 = adata_orig.uns["data_des"][-1]
    data_des = data_des_0 + f"_OneTimeClone_{label}"
    adata.uns["data_des"] = [data_des_orig, data_des]

    time_info = np.array(adata.obs["time_info"])
    time_index_t2 = time_info == later_time_point
    time_index_t1 = ~time_index_t2

    ## set cells without a clone ID to have a unique clone ID
    if padding_X_clone:
        logg.info("Generate a unique clonal label for each clonally unlabeled cell.")
        time_index_t2_orig = time_info_orig == later_time_point
        zero_clone_idx = clone_annot_orig[time_index_t2_orig].sum(1).A.flatten() == 0
        clone_annot_t2_padding = np.diag(np.ones(np.sum(zero_clone_idx)))
        non_zero_clones_idx = (
            clone_annot_orig[time_index_t2_orig].sum(0).A.flatten() > 0
        )
        M0 = np.sum(non_zero_clones_idx)
        M1 = clone_annot_t2_padding.shape[1]
        clone_annot_new = np.zeros((clone_annot_orig.shape[0], M0 + M1))
        clone_annot_new[:, :M0] = clone_annot_orig[:, non_zero_clones_idx].A
        sp_id_t2 = np.nonzero(time_index_t2_orig)[0]
        clone_annot_new[sp_id_t2[zero_clone_idx], M0:] = clone_annot_t2_padding
    else:
        clone_annot_new = clone_annot_orig

    # remove clones without a cell at t2
    valid_clone_id = np.nonzero(
        clone_annot_new[time_info_orig == later_time_point].sum(0).A.flatten() > 0
    )[0]
    X_clone_temp = clone_annot_new[:, valid_clone_id]
    adata_orig.obsm["X_clone"] = ssp.csr_matrix(X_clone_temp)

    #### used for similarity matrix generation
    Tmap_cell_id_t1 = np.nonzero(time_index_t1)[0]
    Tmap_cell_id_t2 = np.nonzero(time_index_t2)[0]
    adata.uns["Tmap_cell_id_t1"] = Tmap_cell_id_t1
    adata.uns["Tmap_cell_id_t2"] = Tmap_cell_id_t2
    adata.uns["clonal_cell_id_t1"] = Tmap_cell_id_t1
    adata.uns["clonal_cell_id_t2"] = Tmap_cell_id_t2
    adata.uns["sp_idx"] = sp_idx
    data_path = settings.data_path

    transition_map = np.zeros((len(Tmap_cell_id_t1), len(Tmap_cell_id_t2)))
    ini_transition_map = np.zeros((len(Tmap_cell_id_t1), len(Tmap_cell_id_t2)))
    X_clone_updated = adata_orig.obsm["X_clone"][
        sp_idx
    ].A  # this does not work well if there are empty clones to begin with

    logg.info(
        "--------Infer transition map between initial time points and the later time one-------"
    )
    for yy in initial_time_points:
        logg.info(f"--------Current initial time point: {yy}--------")

        adata_temp = infer_Tmap_from_one_time_clones_twoTime(
            adata_orig,
            selected_two_time_points=[yy, later_time_point],
            initialize_method=initialize_method,
            OT_epsilon=OT_epsilon,
            OT_dis_KNN=OT_dis_KNN,
            OT_cost=OT_cost,
            HighVar_gene_pctl=HighVar_gene_pctl,
            normalization_mode=normalization_mode,
            sparsity_threshold=sparsity_threshold,
            CoSpar_KNN=CoSpar_KNN,
            use_full_Smatrix=use_full_Smatrix,
            smooth_array=smooth_array,
            trunca_threshold=trunca_threshold,
            compute_new=compute_new,
            use_fixed_clonesize_t1=use_fixed_clonesize_t1,
            sort_clone=sort_clone,
            save_subset=save_subset,
            use_existing_KNN_graph=use_existing_KNN_graph,
            max_iter_N=max_iter_N,
            epsilon_converge=epsilon_converge,
        )

        temp_id_t1 = np.nonzero(time_info == yy)[0]
        sp_id_t1 = hf.converting_id_from_fullSpace_to_subSpace(
            temp_id_t1, Tmap_cell_id_t1
        )[0]

        transition_map_temp = adata_temp.uns["transition_map"].A
        transition_map[sp_id_t1, :] = transition_map_temp

        if initialize_method == "OT":
            transition_map_ini_temp = adata_temp.uns["OT_transition_map"]
        else:
            transition_map_ini_temp = adata_temp.uns["HighVar_transition_map"]

        ini_transition_map[sp_id_t1, :] = transition_map_ini_temp.A

        # Update clonal prediction. This does not work well if there are empty clones to begin with
        time_info_idx = np.array(adata_temp.obs["time_info"]) == yy
        X_clone_updated[temp_id_t1, :] = adata_temp.obsm["X_clone"][time_info_idx].A

    adata.uns["transition_map"] = ssp.csr_matrix(transition_map)
    adata.obsm["X_clone"] = ssp.csr_matrix(X_clone_updated)

    if initialize_method == "OT":
        adata.uns["OT_transition_map"] = ssp.csr_matrix(ini_transition_map)
    else:
        adata.uns["HighVar_transition_map"] = ssp.csr_matrix(ini_transition_map)

    adata_orig.obsm["X_clone"] = clone_annot_orig  # reset to the original clonal matrix
    logg.info(f"-----------Total used time: {time.time()-t0} s ------------")
    return adata


# updated version: v1, we initialize the X_clone as isolated cells
def infer_Tmap_from_state_info_alone(
    adata_orig,
    initial_time_points=None,
    later_time_point=None,
    initialize_method="OT",
    OT_epsilon=0.02,
    OT_dis_KNN=5,
    OT_cost="SPD",
    HighVar_gene_pctl=85,
    normalization_mode=1,
    sparsity_threshold=0.2,
    CoSpar_KNN=20,
    use_full_Smatrix=True,
    smooth_array=[15, 10, 5],
    trunca_threshold=[0.001, 0.01],
    compute_new=False,
    max_iter_N=[1, 5],
    epsilon_converge=[0.05, 0.05],
    use_fixed_clonesize_t1=False,
    sort_clone=1,
    save_subset=True,
    use_existing_KNN_graph=False,
):
    """
    Infer transition map from state information alone.

    After initializing the clonal matrix as such that each cell has a unique barcode,
    it runs :func:`.infer_Tmap_from_one_time_clones` to infer the transition map. Please see :func:`.infer_Tmap_from_one_time_clones` for the meaning of each parameter.

    Returns
    -------
    adata will include both the inferred transition map, and also the updated X_clone matrix.
    The input, adata_orig, will maintain the original X_clone matrix.
    """

    ##--------------- check input parameters
    if "data_des" not in adata_orig.uns.keys():
        adata_orig.uns["data_des"] = ["cospar"]
    logg.info(
        "Step I: Generate pseudo clones where each cell has a unique barcode-----"
    )

    if type(later_time_point) == list:
        later_time_point = later_time_point[0]

    hf.update_time_ordering(adata_orig, mode="auto")
    time_ordering = adata_orig.uns["time_ordering"]

    # use the last time point
    if later_time_point is None:
        later_time_point = time_ordering[-1]

    if initial_time_points is None:
        # use the time points preceding the last one.
        sel_id_temp = np.nonzero(np.in1d(time_ordering, [later_time_point]))[0][0]
        initial_time_points = time_ordering[:sel_id_temp]
    else:
        # re-order time points. This also gets rid of invalid time points
        sel_idx_temp = np.in1d(time_ordering, initial_time_points)
        if np.sum(sel_idx_temp) > 0:
            initial_time_points = time_ordering[sel_idx_temp]
        else:
            raise ValueError(
                f"The 'initial_time_points' are not valid. Please select from {time_ordering}"
            )

    ##--------------- use the artifical clonal matrix
    X_clone_0 = adata_orig.obsm["X_clone"].copy()
    # adata_orig.obsm['X_clone_old']=adata_orig.obsm['X_clone'].copy()
    X_clone = np.diag(np.ones(adata_orig.shape[0]))
    adata_orig.obsm["X_clone"] = ssp.csr_matrix(X_clone)

    logg.info("Step II: Perform joint optimization-----")
    adata = infer_Tmap_from_one_time_clones(
        adata_orig,
        initial_time_points=initial_time_points,
        later_time_point=later_time_point,
        initialize_method=initialize_method,
        OT_epsilon=OT_epsilon,
        OT_dis_KNN=OT_dis_KNN,
        OT_cost=OT_cost,
        HighVar_gene_pctl=HighVar_gene_pctl,
        normalization_mode=normalization_mode,
        sparsity_threshold=sparsity_threshold,
        CoSpar_KNN=CoSpar_KNN,
        use_full_Smatrix=use_full_Smatrix,
        smooth_array=smooth_array,
        trunca_threshold=trunca_threshold,
        compute_new=compute_new,
        max_iter_N=max_iter_N,
        epsilon_converge=epsilon_converge,
        use_fixed_clonesize_t1=use_fixed_clonesize_t1,
        sort_clone=sort_clone,
        save_subset=save_subset,
        use_existing_KNN_graph=use_existing_KNN_graph,
    )

    # only restore the original X_clone information to adata_orig.
    # adata will carry the new structure
    adata_orig.obsm["X_clone"] = X_clone_0

    time_info_orig = np.array(adata_orig.obs["time_info"])
    all_time_points = list(initial_time_points) + [later_time_point]
    label = "t"
    for xx in all_time_points:
        id_array = np.nonzero(time_info_orig == xx)[0]
        label = label + "*" + str(xx)

    data_des_orig = adata_orig.uns["data_des"][0]
    data_des_0 = adata_orig.uns["data_des"][-1]
    data_des = data_des_0 + f"_StateInfo_{label}"
    adata.uns["data_des"] = [data_des_orig, data_des]

    return adata


def infer_Tmap_from_one_time_clones_twoTime(
    adata_orig,
    selected_two_time_points=["1", "2"],
    initialize_method="OT",
    OT_epsilon=0.02,
    OT_dis_KNN=5,
    OT_cost="SPD",
    HighVar_gene_pctl=80,
    normalization_mode=1,
    sparsity_threshold=0.2,
    CoSpar_KNN=20,
    use_full_Smatrix=True,
    smooth_array=[15, 10, 5],
    max_iter_N=[1, 5],
    epsilon_converge=[0.05, 0.05],
    trunca_threshold=[0.001, 0.01],
    compute_new=True,
    use_fixed_clonesize_t1=False,
    sort_clone=1,
    save_subset=True,
    joint_optimization=True,
    use_existing_KNN_graph=False,
):
    """
    Infer transition map from clones with a single time point

    It is the same as :func:`.infer_Tmap_from_one_time_clones`, except that
    it assumes that the input adata_orig has only two time points.

    joint_optimization: `bool`, optional (default: True).
    """

    time_info_orig = np.array(adata_orig.obs["time_info"])
    sort_time_point = np.sort(list(set(time_info_orig)))
    N_valid_time = np.sum(np.in1d(sort_time_point, selected_two_time_points))
    if N_valid_time != 2:
        logg.error(f"Must select only two time points among the list {sort_time_point}")
        # The second time point in this list (not necessarily later time point) is assumed to have clonal data.")
    else:
        ####################################

        logg.info("Step 0: Pre-processing and sub-sampling cells-------")
        # select cells from the two time points, and sub-sampling, create the new adata object with these cell states
        sp_idx = (time_info_orig == selected_two_time_points[0]) | (
            time_info_orig == selected_two_time_points[1]
        )

        adata = adata_orig[sp_idx]
        data_des_0 = adata_orig.uns["data_des"][-1]
        data_des_orig = adata_orig.uns["data_des"][0]
        data_des = (
            data_des_0
            + f"_t*{selected_two_time_points[0]}*{selected_two_time_points[1]}"
        )
        adata.uns["data_des"] = [data_des_orig, data_des]

        time_info = np.array(adata.obs["time_info"])
        time_index_t1 = time_info == selected_two_time_points[0]
        time_index_t2 = time_info == selected_two_time_points[1]

        #### used for similarity matrix generation
        Tmap_cell_id_t1 = np.nonzero(time_index_t1)[0]
        Tmap_cell_id_t2 = np.nonzero(time_index_t2)[0]
        adata.uns["Tmap_cell_id_t1"] = Tmap_cell_id_t1
        adata.uns["Tmap_cell_id_t2"] = Tmap_cell_id_t2
        adata.uns["clonal_cell_id_t1"] = Tmap_cell_id_t1
        adata.uns["clonal_cell_id_t2"] = Tmap_cell_id_t2
        adata.uns["sp_idx"] = sp_idx
        data_path = settings.data_path

        ###############################
        # prepare the similarity matrix with all state info, all subsequent similarity will be down-sampled from this one.
        if use_full_Smatrix and (joint_optimization or (initialize_method != "OT")):

            temp_str = "0" + str(trunca_threshold[0])[2:]
            round_of_smooth = np.max(smooth_array)
            data_des = adata_orig.uns["data_des"][0]
            similarity_file_name = os.path.join(
                data_path,
                f"{data_des}_Similarity_matrix_with_all_cell_states_kNN{CoSpar_KNN}_Truncate{temp_str}",
            )
            if not (
                os.path.exists(similarity_file_name + f"_SM{round_of_smooth}.npz")
                and (not compute_new)
            ):
                similarity_matrix_full = tmap_util.generate_similarity_matrix(
                    adata_orig,
                    similarity_file_name,
                    round_of_smooth=round_of_smooth,
                    neighbor_N=CoSpar_KNN,
                    truncation_threshold=trunca_threshold[0],
                    save_subset=save_subset,
                    compute_new_Smatrix=compute_new,
                )

        if initialize_method == "OT":

            # logg.info("----------------")
            logg.info("Step 1: Use OT method for initialization-------")

            tmap_core.infer_Tmap_from_optimal_transport(
                adata,
                OT_epsilon=OT_epsilon,
                OT_cost=OT_cost,
                OT_dis_KNN=OT_dis_KNN,
                compute_new=compute_new,
                use_existing_KNN_graph=use_existing_KNN_graph,
            )

            OT_transition_map = adata.uns["OT_transition_map"]
            initialized_map = OT_transition_map

        else:

            # logg.info("----------------")
            logg.info("Step 1: Use the HighVar method for initialization-------")

            t = time.time()
            tmap_core.infer_Tmap_from_HighVar(
                adata,
                min_counts=3,
                min_cells=3,
                min_gene_vscore_pctl=HighVar_gene_pctl,
                sparsity_threshold=sparsity_threshold,
                neighbor_N=CoSpar_KNN,
                normalization_mode=normalization_mode,
                use_full_Smatrix=use_full_Smatrix,
                smooth_array=smooth_array,
                trunca_threshold=trunca_threshold,
                compute_new_Smatrix=compute_new,
                max_iter_N=max_iter_N[1],
                epsilon_converge=epsilon_converge[1],
            )

            HighVar_transition_map = adata.uns["HighVar_transition_map"]
            initialized_map = HighVar_transition_map
            logg.info(
                f"Finishing initialization using HighVar, used time {time.time()-t}"
            )

        if joint_optimization:
            ########### Jointly optimize the transition map and the initial clonal states
            if selected_two_time_points[1] in adata_orig.uns["clonal_time_points"]:

                # logg.info("----------------")
                logg.info(
                    "Step 2: Jointly optimize the transition map and the initial clonal states-------"
                )

                t = time.time()

                tmap_core.refine_Tmap_through_joint_optimization(
                    adata,
                    initialized_map,
                    normalization_mode=normalization_mode,
                    sparsity_threshold=sparsity_threshold,
                    CoSpar_KNN=CoSpar_KNN,
                    use_full_Smatrix=use_full_Smatrix,
                    smooth_array=smooth_array,
                    max_iter_N=max_iter_N,
                    epsilon_converge=epsilon_converge,
                    trunca_threshold=trunca_threshold,
                    compute_new=compute_new,
                    use_fixed_clonesize_t1=use_fixed_clonesize_t1,
                    sort_clone=sort_clone,
                    save_subset=save_subset,
                )

                logg.info(f"Finishing Joint Optimization, used time {time.time()-t}")
            else:
                logg.warn(
                    "No clonal information available. Skip the joint optimization of clone and scRNAseq data"
                )

        if "Smatrix" in adata.uns.keys():
            adata.uns.pop("Smatrix")
        return adata


def infer_Tmap_from_clonal_info_alone_private(
    adata_orig, method="naive", clonal_time_points=None, selected_fates=None
):
    """
    Compute transition map using only the lineage information.

    Here, we compute the transition map between neighboring time points.

    We simply average transitions across all clones (or selected clones when method='Weinreb'),
    assuming that the intra-clone transition is uniform within the same clone.

    Parameters
    ----------
    adata_orig: :class:`~anndata.AnnData` object
    method: `str`, optional (default: 'naive')
        Method used to compute the transition map. Choice: {'naive',
        'weinreb'}. For the naive method, we simply average transitions
        across all clones, assuming that the intra-clone transitions are
        uniform within the same clone. For the 'weinreb' method, we first
        find uni-potent clones, then compute the transition map by simply
        averaging across all clonal transitions as the naive method.
    selected_fates: `list`, optional (default: all selected)
        List of targeted fate clusters to define uni-potent clones for the
        weinreb method, which are used to compute the transition map.
    clonal_time_points: `list` of `str`, optional (default: all time points)
        List of time points to be included for analysis.
        We assume that each selected time point has clonal measurements.
    later_time_points: `list`, optional (default: None)
        If specified, the function will produce a map T between these early
        time points among `clonal_time_points` and the `later_time_point`.
        If not specified, it produces a map T between neighboring time points.

    Returns
    -------
    adata: :class:`~anndata.AnnData` object
        The transition map is stored at adata.uns['clonal_transition_map']
    """

    adata_1 = tmap_util.select_time_points(
        adata_orig, time_point=clonal_time_points, extend_Tmap_space=True
    )
    if method not in ["naive", "weinreb"]:
        logg.warn("method not in ['naive','weinreb']; set to be 'weinreb'")
        method = "weinreb"

    cell_id_t2_all = adata_1.uns["Tmap_cell_id_t2"]
    cell_id_t1_all = adata_1.uns["Tmap_cell_id_t1"]

    T_map = np.zeros((len(cell_id_t1_all), len(cell_id_t2_all)))
    clone_annot = adata_1.obsm["X_clone"]

    N_points = len(adata_1.uns["multiTime_cell_id_t1"])
    for k in range(N_points):

        cell_id_t1_temp = adata_1.uns["multiTime_cell_id_t1"][k]
        cell_id_t2_temp = adata_1.uns["multiTime_cell_id_t2"][k]
        if method == "naive":
            logg.info("Use all clones (naive method)")
            T_map_temp = clone_annot[cell_id_t1_temp] * clone_annot[cell_id_t2_temp].T

        else:
            logg.info("Use only uni-potent clones (weinreb et al., 2020)")
            state_annote = np.array(adata_1.obs["state_info"])
            if selected_fates == None:
                selected_fates = list(set(state_annote))
            potential_vector_clone, fate_entropy_clone = tl.compute_state_potential(
                clone_annot[cell_id_t2_temp].T,
                state_annote[cell_id_t2_temp],
                selected_fates,
                fate_count=True,
            )

            sel_unipotent_clone_id = np.array(
                list(set(np.nonzero(fate_entropy_clone == 1)[0]))
            )
            clone_annot_unipotent = clone_annot[:, sel_unipotent_clone_id]
            T_map_temp = (
                clone_annot_unipotent[cell_id_t1_temp]
                * clone_annot_unipotent[cell_id_t2_temp].T
            )
            logg.info(
                f"Used uni-potent clone fraction {len(sel_unipotent_clone_id)/clone_annot.shape[1]}"
            )

        idx_t1 = np.nonzero(np.in1d(cell_id_t1_all, cell_id_t1_temp))[0]
        idx_t2 = np.nonzero(np.in1d(cell_id_t2_all, cell_id_t2_temp))[0]
        idx_t1_temp = np.nonzero(np.in1d(cell_id_t1_temp, cell_id_t1_all))[0]
        idx_t2_temp = np.nonzero(np.in1d(cell_id_t2_temp, cell_id_t2_all))[0]
        T_map[idx_t1[:, np.newaxis], idx_t2] = T_map_temp[idx_t1_temp][:, idx_t2_temp].A

    T_map = T_map.astype(int)
    adata_1.uns["clonal_transition_map"] = ssp.csr_matrix(T_map)
    return adata_1


# the v2 version, it is the same format as infer_Tmap_from_multiTime_clones.
# We return a new adata object that will throw away existing annotations in uns.
def infer_Tmap_from_clonal_info_alone(
    adata_orig,
    method="naive",
    clonal_time_points=None,
    later_time_point=None,
    selected_fates=None,
):
    """
    Compute transition map using only the lineage information.

    As in :func:`.infer_Tmap_from_multitime_clones`, we provide two modes of inference:

    * If `later_time_point=None`, the inferred map allows transitions
      between neighboring time points. For example, if
      clonal_time_points=['day1','day2','day3'], then it computes transitions
      for pairs (day1, day2) and (day2, day3), but not (day1, day3).

    * If `later_time_point` is specified, the function produces a map
      between earlier time points and this later time point. For example, if
      `later_time_point='day3`, the map allows transitions for pairs (day1, day3)
      and (day2, day3), but not (day1,day2).

    Parameters
    ----------
    adata_orig: :class:`~anndata.AnnData` object
    method: `str`, optional (default: 'naive')
        Method used to compute the transition map. Choice: {'naive',
        'weinreb'}. For the naive method, we simply average transitions
        across all clones, assuming that the intra-clone transitions are
        uniform within the same clone. For the 'weinreb' method, we first
        find uni-potent clones, then compute the transition map by simply
        averaging across all clonal transitions as the naive method.
    selected_fates: `list`, optional (default: all selected)
        List of targeted fate clusters to define uni-potent clones for the
        weinreb method, which are used to compute the transition map.
    clonal_time_points: `list` of `str`, optional (default: all time points)
        List of time points to be included for analysis.
        We assume that each selected time point has clonal measurements.
    later_time_points: `list`, optional (default: None)
        If specified, the function will produce a map T between these early
        time points among `clonal_time_points` and the `later_time_point`.
        If not specified, it produces a map T between neighboring time points.

    Returns
    -------
    adata: :class:`~anndata.AnnData` object
        The transition map is stored at adata.uns['clonal_transition_map']
    """

    hf.check_available_clonal_info(adata_orig)
    clonal_time_points_0 = np.array(adata_orig.uns["clonal_time_points"])
    if len(clonal_time_points_0) < 2:
        raise ValueError("There are no multi-time clones. Abort the inference.")

    if clonal_time_points is None:
        clonal_time_points = clonal_time_points_0

    if type(later_time_point) == list:
        later_time_point = later_time_point[0]

    if later_time_point is not None:
        clonal_time_points = list(clonal_time_points) + [later_time_point]
        clonal_time_points = list(set(clonal_time_points))

    hf.check_input_parameters(
        adata_orig,
        later_time_point=later_time_point,
        clonal_time_points=clonal_time_points,
    )
    # order the clonal time points
    time_ordering = adata_orig.uns["time_ordering"]
    sel_idx_temp = np.in1d(time_ordering, clonal_time_points)
    clonal_time_points = time_ordering[sel_idx_temp]

    if later_time_point is None:
        logg.info("Infer transition map between neighboring time points.")
        adata = infer_Tmap_from_clonal_info_alone_private(
            adata_orig,
            method=method,
            clonal_time_points=clonal_time_points,
            selected_fates=selected_fates,
        )

        return adata
    else:
        logg.info(
            f"Infer transition map between initial time points and the later time point."
        )
        # compute transition map between initial time points and the later time point
        sel_id = np.nonzero(np.in1d(clonal_time_points, later_time_point))[0][0]
        initial_time_points = clonal_time_points[:sel_id]

        time_info_orig = np.array(adata_orig.obs["time_info"])
        sp_idx = np.zeros(adata_orig.shape[0], dtype=bool)
        all_time_points = list(initial_time_points) + [later_time_point]
        label = "t"
        for xx in all_time_points:
            id_array = np.nonzero(time_info_orig == xx)[0]
            sp_idx[id_array] = True
            label = label + "*" + str(xx)

        adata = adata_orig[sp_idx]
        data_des_orig = adata_orig.uns["data_des"][0]
        data_des_0 = adata_orig.uns["data_des"][-1]
        data_des = data_des_0 + f"_ClonalMap_Later_{label}"
        adata_orig.uns["data_des"] = [data_des_orig, data_des]

        time_info = np.array(adata_orig.obs["time_info"])
        time_index_t2 = time_info == later_time_point
        time_index_t1 = ~time_index_t2

        #### used for similarity matrix generation
        Tmap_cell_id_t1 = np.nonzero(time_index_t1)[0]
        Tmap_cell_id_t2 = np.nonzero(time_index_t2)[0]
        adata.uns["Tmap_cell_id_t1"] = Tmap_cell_id_t1
        adata.uns["Tmap_cell_id_t2"] = Tmap_cell_id_t2
        adata.uns["clonal_cell_id_t1"] = Tmap_cell_id_t1
        adata.uns["clonal_cell_id_t2"] = Tmap_cell_id_t2
        adata.uns["sp_idx"] = sp_idx
        data_path = settings.data_path

        transition_map = np.zeros((len(Tmap_cell_id_t1), len(Tmap_cell_id_t2)))

        # logg.info("------Infer transition map between initial time points and the later time one-------")
        for yy in initial_time_points:
            logg.info(f"--------Current initial time point: {yy}--------")

            # by default, we extend the state space to all cells at the given time point.
            adata_temp = infer_Tmap_from_clonal_info_alone_private(
                adata_orig,
                method=method,
                clonal_time_points=[yy, later_time_point],
                selected_fates=selected_fates,
            )

            temp_id_t1 = np.nonzero(time_info == yy)[0]
            sp_id_t1 = hf.converting_id_from_fullSpace_to_subSpace(
                temp_id_t1, Tmap_cell_id_t1
            )[0]

            # by default, we extend the state space to all cells at the given time point.
            # so we only need to care about t1.
            transition_map[sp_id_t1, :] = adata_temp.uns["clonal_transition_map"].A

        adata.uns["clonal_transition_map"] = ssp.csr_matrix(transition_map)

        return adata
