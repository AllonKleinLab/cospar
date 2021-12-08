import os
import time

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as ssp

from .. import help_functions as hf
from .. import logging as logg
from .. import plotting as pl
from .. import settings
from .. import tool as tl


def generate_similarity_matrix(
    adata,
    file_name,
    round_of_smooth=10,
    neighbor_N=20,
    beta=0.1,
    truncation_threshold=0.001,
    save_subset=True,
    use_existing_KNN_graph=False,
    compute_new_Smatrix=False,
):
    """
    Generate similarity matrix (Smatrix) through graph diffusion

    It generates the similarity matrix via iterative graph diffusion.
    Similarity matrix from each round of diffusion will be saved, after truncation
    to promote sparsity and save space. If save_subset is activated, only save
    Smatrix for smooth rounds at the multiples of 5 (like 5,10,15,...). If a Smatrix is pre-computed,
    it will be loaded directly if compute_new_Smatrix=Flase.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    file_name: str
        Filename to load pre-computed similarity matrix or save the newly
        computed similarity matrix.
    round_of_smooth: `int`, optional (default: 10)
        The rounds of graph diffusion.
    neighbor_N: `int`, optional (default: 20)
        Neighber number for constructing the KNN graph, using the UMAP method.
    beta: `float`, option (default: 0.1)
        Probability to stay at the origin in a unit diffusion step, in the range [0,1]
    truncation_threshold: `float`, optional (default: 0.001)
        At each iteration, truncate the similarity matrix using
        truncation_threshold. This promotes the sparsity of the matrix,
        thus the speed of computation. We set the truncation threshold to be small,
        to guarantee accracy.
    save_subset: `bool`, optional (default: True)
        If true, save only Smatrix at smooth round [5,10,15,...];
        Otherwise, save Smatrix at each round.
    use_existing_KNN_graph: `bool`, optional (default: False)
        If true and adata.obsp['connectivities'], use the existing knn graph to build
        the similarity matrix, regardless of neighbor_N.
    compute_new_Smatrix: `bool`, optional (default: False)
        If true, compute a new Smatrix, even if there is pre-computed Smatrix with the
        same parameterization.

    Returns
    -------
        similarity_matrix: `sp.spmatrix`
    """

    if os.path.exists(file_name + f"_SM{round_of_smooth}.npz") and (
        not compute_new_Smatrix
    ):

        logg.hint("Compute similarity matrix: load existing data")
        similarity_matrix = ssp.load_npz(file_name + f"_SM{round_of_smooth}.npz")
    else:  # compute now

        logg.hint(f"Compute similarity matrix: computing new; beta={beta}")

        # add a step to compute PCA in case this is not computed

        if (not use_existing_KNN_graph) or ("connectivities" not in adata.obsp.keys()):
            # here, we assume that adata already has pre-computed PCA
            sc.pp.neighbors(adata, n_neighbors=neighbor_N)
        else:
            logg.hint(
                "Use existing KNN graph at adata.obsp['connectivities'] for generating the smooth matrix"
            )
        adjacency_matrix = adata.obsp["connectivities"]

        ############## The new method
        adjacency_matrix = (adjacency_matrix + adjacency_matrix.T) / 2
        ##############

        adjacency_matrix = hf.sparse_rowwise_multiply(
            adjacency_matrix, 1 / adjacency_matrix.sum(1).A.squeeze()
        )
        nrow = adata.shape[0]
        similarity_matrix = ssp.lil_matrix((nrow, nrow))
        similarity_matrix.setdiag(np.ones(nrow))
        transpose_A = adjacency_matrix.T

        if round_of_smooth == 0:
            SM = 0
            similarity_matrix = ssp.csr_matrix(similarity_matrix)
            ssp.save_npz(file_name + f"_SM{SM}.npz", similarity_matrix)

        for iRound in range(round_of_smooth):
            SM = iRound + 1

            logg.info("Smooth round:", SM)
            t = time.time()
            similarity_matrix = (
                beta * similarity_matrix + (1 - beta) * transpose_A * similarity_matrix
            )
            # similarity_matrix =beta*similarity_matrix+(1-beta)*similarity_matrix*adjacency_matrix
            # similarity_matrix_array.append(similarity_matrix)

            logg.hint("Time elapsed:", time.time() - t)

            t = time.time()
            sparsity_frac = (similarity_matrix > 0).sum() / (
                similarity_matrix.shape[0] * similarity_matrix.shape[1]
            )
            if sparsity_frac >= 0.1:
                # similarity_matrix_truncate=similarity_matrix
                # similarity_matrix_truncate_array.append(similarity_matrix_truncate)

                logg.hint(f"Orignal sparsity={sparsity_frac}, Thresholding")
                similarity_matrix = hf.matrix_row_or_column_thresholding(
                    similarity_matrix, truncation_threshold
                )
                sparsity_frac_2 = (similarity_matrix > 0).sum() / (
                    similarity_matrix.shape[0] * similarity_matrix.shape[1]
                )
                # similarity_matrix_truncate_array.append(similarity_matrix_truncate)

                logg.hint(f"Final sparsity={sparsity_frac_2}")

                logg.info(
                    f"similarity matrix truncated (Smooth round={SM}): ",
                    time.time() - t,
                )

            # logg.info("Save the matrix")
            # file_name=f'data/20200221_truncated_similarity_matrix_SM{round_of_smooth}_kNN{neighbor_N}_Truncate{str(truncation_threshold)[2:]}.npz'
            similarity_matrix = ssp.csr_matrix(similarity_matrix)

            ############## The new method
            # similarity_matrix=similarity_matrix.T.copy()
            ##############

            if save_subset:
                if SM % 5 == 0:  # save when SM=5,10,15,20,...

                    logg.hint("Save the matrix at every 5 rounds")
                    ssp.save_npz(file_name + f"_SM{SM}.npz", similarity_matrix)
            else:  # save all

                logg.hint("Save the matrix at every round")
                ssp.save_npz(file_name + f"_SM{SM}.npz", similarity_matrix)

    return similarity_matrix


def generate_initial_similarity(similarity_matrix, initial_index_0, initial_index_1):
    """
    Extract Smatrix at t1 from the full Smatrix

    Parameters
    ----------
    similarity_matrix: `np.array` or `sp.spmatrix`
        full Smatrix
    initial_index_0: `list`
        list of selected t1-cell id among all cells (t1+t2)
    initial_index_1: `list`
        list of selected t1-cell id among all cells (t1+t2)
        It can be the same as initial_index_0. In the case that they are different,
        initial_index_1 is a subset of cells that correspond to multi-time clones,
        while initial_index_0 may be all cells at t1.

    Returns
    -------
    initial Smatrix: `np.array`
    """

    t = time.time()
    initial_similarity = similarity_matrix[initial_index_0][:, initial_index_1]
    # initial_similarity=hf.sparse_column_multiply(initial_similarity,1/(resol+initial_similarity.sum(0)))
    if ssp.issparse(initial_similarity):
        initial_similarity = initial_similarity.A

    logg.hint("Time elapsed: ", time.time() - t)
    return initial_similarity


def generate_final_similarity(similarity_matrix, final_index_0, final_index_1):
    """
    Extract Smatrix at t2 from the full Smatrix

    Parameters
    ----------
    similarity_matrix: `np.array` or `sp.spmatrix`
        full Smatrix
    final_index_0: `list`
        list of selected t2-cell id among all cells (t1+t2)
    final_index_1: `list`
        list of selected t2-cell id among all cells (t1+t2)
        It can be the same as final_index_0. In the case that they are different,
        initial_index_0 is a subset of cells that correspond to multi-time clones,
        while initial_index_1 may be all cells at t2.

    Returns
    -------
    initial Smatrix: `np.array`
    """

    t = time.time()
    final_similarity = similarity_matrix.T[final_index_0][:, final_index_1]
    if ssp.issparse(final_similarity):
        final_similarity = final_similarity.A
    # final_similarity=hf.sparse_rowwise_multiply(final_similarity,1/(resol+final_similarity.sum(1)))

    logg.hint("Time elapsed: ", time.time() - t)
    return final_similarity


def select_time_points(
    adata_orig, time_point=["day_1", "day_2"], extend_Tmap_space=False
):
    """
    Select barcoded cells at given time points for Tmap inference.

    Select cells at given time points, and prepare the right data structure
    for running core cospar function to infer the Tmap.

    Parameters
    ----------
    adata_orig: original :class:`~anndata.AnnData` object
    time_point: `list` optional (default: ['day_1','day_2'])
        Require at least two time points, arranged in ascending order.
    extend_Tmap_space: `bool` optional (default: `False`)
        If true, the initial states for Tmap will include all states at initial time points,
        and the later states for Tmap will include all states at later time points.
        Otherwise, the initial and later state
        space of the Tmap will be restricted to cells with multi-time clonal information
        alone. The latter case speeds up the computation, which is recommended.

    Returns
    -------
    Subsampled :class:`~anndata.AnnData` object
    """

    # x_emb_orig=adata_orig.obsm['X_emb'][:,0]
    # y_emb_orig=adata_orig.obsm['X_emb'][:,1]
    time_info_orig = np.array(adata_orig.obs["time_info"])
    clone_annot_orig = adata_orig.obsm["X_clone"]
    if len(time_point) == 0:  # use all clonally labelled cell states
        time_point = np.sort(
            list(set(time_info_orig))
        )  # this automatic ordering may not work

    if len(time_point) < 2:
        logg.error("Must select more than 1 time point!")
    else:

        At = []
        for j, time_0 in enumerate(time_point):
            At.append(ssp.csr_matrix(clone_annot_orig[time_info_orig == time_0]))

        ### Day t - t+1
        Clonal_cell_ID_FOR_t = []
        for j in range(len(time_point) - 1):
            idx_t = np.array((At[j] * At[j + 1].T).sum(1) > 0).flatten()
            time_index_t = time_info_orig == time_point[j]
            temp = np.nonzero(time_index_t)[0][idx_t]
            Clonal_cell_ID_FOR_t.append(
                temp
            )  # this index is in the original space, without sampling etc

            logg.hint(
                f"Clonal cell fraction (day {time_point[j]}-{time_point[j+1]}):",
                len(temp) / np.sum(time_index_t),
            )

        ### Day t+1 - t
        Clonal_cell_ID_BACK_t = []
        for j in range(len(time_point) - 1):
            idx_t = np.array((At[j + 1] * At[j].T).sum(1) > 0).flatten()
            time_index_t = time_info_orig == time_point[j + 1]
            temp = np.nonzero(time_index_t)[0][idx_t]
            Clonal_cell_ID_BACK_t.append(
                temp
            )  # this index is in the original space, without sampling etc

            logg.hint(
                f"Clonal cell fraction (day {time_point[j+1]}-{time_point[j]}):",
                len(temp) / np.sum(time_index_t),
            )

        for j in range(len(time_point) - 1):
            logg.hint(
                f"Numer of cells that are clonally related -- day {time_point[j]}: {len(Clonal_cell_ID_FOR_t[j])}  and day {time_point[j+1]}: {len(Clonal_cell_ID_BACK_t[j])}"
            )

        proportion = np.ones(len(time_point))
        # flatten the list
        flatten_clonal_cell_ID_FOR = np.array(
            [sub_item for item in Clonal_cell_ID_FOR_t for sub_item in item]
        )
        flatten_clonal_cell_ID_BACK = np.array(
            [sub_item for item in Clonal_cell_ID_BACK_t for sub_item in item]
        )
        valid_clone_N_FOR = np.sum(
            clone_annot_orig[flatten_clonal_cell_ID_FOR].A.sum(0) > 0
        )
        valid_clone_N_BACK = np.sum(
            clone_annot_orig[flatten_clonal_cell_ID_BACK].A.sum(0) > 0
        )

        logg.info(f"Number of multi-time clones post selection: {valid_clone_N_FOR}")
        # logg.info("Valid clone number 'BACK' post selection",valid_clone_N_BACK)

        ###################### select initial and later cell states

        if extend_Tmap_space:
            old_Tmap_cell_id_t1 = []
            for t_temp in time_point[:-1]:
                old_Tmap_cell_id_t1 = old_Tmap_cell_id_t1 + list(
                    np.nonzero(time_info_orig == t_temp)[0]
                )
            old_Tmap_cell_id_t1 = np.array(old_Tmap_cell_id_t1)

            ########
            old_Tmap_cell_id_t2 = []
            for t_temp in time_point[1:]:
                old_Tmap_cell_id_t2 = old_Tmap_cell_id_t2 + list(
                    np.nonzero(time_info_orig == t_temp)[0]
                )
            old_Tmap_cell_id_t2 = np.array(old_Tmap_cell_id_t2)

        else:
            old_Tmap_cell_id_t1 = flatten_clonal_cell_ID_FOR
            old_Tmap_cell_id_t2 = flatten_clonal_cell_ID_BACK

        old_clonal_cell_id_t1 = flatten_clonal_cell_ID_FOR
        old_clonal_cell_id_t2 = flatten_clonal_cell_ID_BACK
        ########################

        sp_id = np.sort(
            list(set(list(old_Tmap_cell_id_t1) + list(old_Tmap_cell_id_t2)))
        )
        sp_idx = np.zeros(clone_annot_orig.shape[0], dtype=bool)
        sp_idx[sp_id] = True

        Tmap_cell_id_t1 = hf.converting_id_from_fullSpace_to_subSpace(
            old_Tmap_cell_id_t1, sp_id
        )[0]
        clonal_cell_id_t1 = hf.converting_id_from_fullSpace_to_subSpace(
            old_clonal_cell_id_t1, sp_id
        )[0]
        clonal_cell_id_t2 = hf.converting_id_from_fullSpace_to_subSpace(
            old_clonal_cell_id_t2, sp_id
        )[0]
        Tmap_cell_id_t2 = hf.converting_id_from_fullSpace_to_subSpace(
            old_Tmap_cell_id_t2, sp_id
        )[0]

        Clonal_cell_ID_FOR_t_new = []
        for temp_id_list in Clonal_cell_ID_FOR_t:
            convert_list = hf.converting_id_from_fullSpace_to_subSpace(
                temp_id_list, sp_id
            )[0]
            Clonal_cell_ID_FOR_t_new.append(convert_list)

        Clonal_cell_ID_BACK_t_new = []
        for temp_id_list in Clonal_cell_ID_BACK_t:
            convert_list = hf.converting_id_from_fullSpace_to_subSpace(
                temp_id_list, sp_id
            )[0]
            Clonal_cell_ID_BACK_t_new.append(convert_list)

        sp_id_0 = np.sort(list(old_clonal_cell_id_t1) + list(old_clonal_cell_id_t2))
        sp_idx_0 = np.zeros(clone_annot_orig.shape[0], dtype=bool)
        sp_idx_0[sp_id_0] = True

        barcode_id = np.nonzero(clone_annot_orig[sp_idx_0].A.sum(0).flatten() > 0)[0]
        # sp_id=np.nonzero(sp_idx)[0]
        clone_annot = clone_annot_orig[sp_idx][:, barcode_id]

        adata = adata_orig[sp_idx]
        adata.obsm["X_clone"] = clone_annot
        adata.uns["clonal_cell_id_t1"] = clonal_cell_id_t1
        adata.uns["clonal_cell_id_t2"] = clonal_cell_id_t2
        adata.uns["Tmap_cell_id_t1"] = Tmap_cell_id_t1
        adata.uns["Tmap_cell_id_t2"] = Tmap_cell_id_t2
        adata.uns["multiTime_cell_id_t1"] = Clonal_cell_ID_FOR_t_new
        adata.uns["multiTime_cell_id_t2"] = Clonal_cell_ID_BACK_t_new
        adata.uns["proportion"] = np.ones(len(time_point) - 1)
        adata.uns["sp_idx"] = sp_idx

        data_des_orig = adata_orig.uns["data_des"][0]
        data_des_0 = adata_orig.uns["data_des"][-1]
        time_label = "t"
        for x in time_point:
            time_label = time_label + f"*{x}"

        data_des = (
            data_des_0
            + f"_MultiTimeClone_FullSpace{int(extend_Tmap_space)}_{time_label}"
        )
        adata.uns["data_des"] = [data_des_orig, data_des]

        if logg._settings_verbosity_greater_or_equal_than(3):
            N_cell, N_clone = clone_annot.shape
            logg.info(f"Cell number={N_cell}, Clone number={N_clone}")
            x_emb = adata.obsm["X_emb"][:, 0]
            y_emb = adata.obsm["X_emb"][:, 1]
            pl.customized_embedding(x_emb, y_emb, -x_emb)

        logg.hint(f"clonal_cell_id_t1: {len(clonal_cell_id_t1)}")
        logg.hint(f"Tmap_cell_id_t1: {len(Tmap_cell_id_t1)}")
        return adata
