import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as ssp
import scipy.stats
import statsmodels.sandbox.stats.multicomp
from fastcluster import linkage
from matplotlib import pyplot as plt
from scanpy import read  # So that we can call this function in cospar directly
from scipy.optimize import fmin
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import pairwise
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from tqdm import tqdm

from .. import logging as logg
from .. import plotting as pl
from .. import settings, tmap

# import scipy.stats


def get_dge_SW(ad, mask1, mask2, min_frac_expr=0.05, pseudocount=1):
    """
    Perform differential gene expression analysis.

    Parameters
    ----------
    ad: :class:`~anndata.AnnData` object
    mask1: `np.array`
        A np.array of `bool` for selecting group_1 cells.
    mask2: `np.array`
        A np.array of `bool` for selecting group_2 cells.
    min_frac_expr: `float`, optional (default: 0.05)
        Minimum expression fraction among selected states for a
        gene to be considered for DGE analysis.
    pseudocount: `int`, optional (default: 1)
        pseudo count for taking the gene expression ratio between the two groups

    Returns
    -------
    df: :class:`pandas.DataFrame`
        A pandas dataFrame, with columns: `gene`, `pv`, `mean_1`, `mean_2`, `ratio`
    """

    gene_mask = (
        (ad.X[mask1, :] > 0).sum(0).A.squeeze() / mask1.sum() > min_frac_expr
    ) | ((ad.X[mask2, :] > 0).sum(0).A.squeeze() / mask2.sum() > min_frac_expr)
    # print(gene_mask.sum())
    E1 = ad.X[mask1, :][:, gene_mask].toarray()
    E2 = ad.X[mask2, :][:, gene_mask].toarray()

    m1 = E1.mean(0) + pseudocount
    m2 = E2.mean(0) + pseudocount
    r = np.log2(m1 / m2)

    pv = np.zeros(gene_mask.sum())
    for ii, iG in enumerate(np.nonzero(gene_mask)[0]):
        pv[ii] = scipy.stats.ranksums(E1[:, ii], E2[:, ii])[1]
    pv = statsmodels.sandbox.stats.multicomp.multipletests(
        pv,
        alpha=0.05,
        method="fdr_bh",
    )[1]
    sort_idx = np.argsort(pv)

    df = pd.DataFrame(
        {
            "gene": ad.var_names.values.astype(str)[gene_mask][sort_idx],
            "Qvalue": pv[sort_idx],
            "mean_1": m1[sort_idx] - pseudocount,
            "mean_2": m2[sort_idx] - pseudocount,
            "ratio": r[sort_idx],
        }
    )

    return df


########## USEFUL SPARSE FUNCTIONS


def sparse_var(E, axis=0):
    """calculate variance across the specified axis of a sparse matrix"""

    mean_gene = E.mean(axis=axis).A.squeeze()
    tmp = E.copy()
    tmp.data **= 2
    return tmp.mean(axis=axis).A.squeeze() - mean_gene**2


def mean_center(E, column_means=None):
    """mean-center columns of a sparse matrix"""

    if column_means is None:
        column_means = E.mean(axis=0)
    return E - column_means


def normalize_variance(E, column_stdevs=None):
    """variance-normalize columns of a sparse matrix"""

    if column_stdevs is None:
        column_stdevs = np.sqrt(sparse_var(E, axis=0))
    return sparse_rowwise_multiply(E.T, 1 / column_stdevs).T


# this is not working well
def sparse_zscore(E, gene_mean=None, gene_stdev=None):
    """z-score normalize each column of a sparse matrix"""
    if gene_mean is None:
        gene_mean = E.mean(0)
    if gene_stdev is None:
        gene_stdev = np.sqrt(sparse_var(E))
    return sparse_rowwise_multiply((E - gene_mean).T, 1 / gene_stdev).T


def corr2_coeff(A, B):
    """
    Calculate Pearson correlation between matrix A and B

    A and B are allowed to have different shapes. This method
    does not work if A and B are constituted of constant row vectors,
    in which case the standard deviation becomes zero.

    Parameters
    ----------
    A: np.array, shape (cell_n0, gene_m)
    B: np.array, shape (cell_n1, gene_m)
    """

    resol = 10 ** (-15)
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / (np.sqrt(np.dot(ssA[:, None], ssB[None])) + resol)


def sparse_rowwise_multiply(E, a):
    """
    Multiply each row of the sparse matrix E by a scalar a

    Parameters
    ----------
    E: `np.array` or `sp.spmatrix`
    a: `np.array`
        A scalar vector.

    Returns
    -------
    Rescaled sparse matrix
    """

    nrow = E.shape[0]
    if nrow != a.shape[0]:
        logg.error("Dimension mismatch, multiplication failed")
        return E
    else:
        w = ssp.lil_matrix((nrow, nrow))
        w.setdiag(a)
        return w * E


def sparse_column_multiply(E, a):
    """
    Multiply each columns of the sparse matrix E by a scalar a

    Parameters
    ----------
    E: `np.array` or `sp.spmatrix`
    a: `np.array`
        A scalar vector.

    Returns
    -------
    Rescaled sparse matrix
    """

    ncol = E.shape[1]
    if ncol != a.shape[0]:
        logg.error("Dimension mismatch, multiplication failed")
        return E
    else:
        w = ssp.lil_matrix((ncol, ncol))
        w.setdiag(a)
        return ssp.csr_matrix(E) * w


# This is faster than v0
def matrix_row_or_column_thresholding(input_matrix, threshold=0.1, row_threshold=True):
    """
    Row or column-wise thresholding a matrix

    Set entries in a given row (column) to be zero, if its value is below threshold*max(row_vector).

    Parameters
    ----------
    input_matrix: `np.array`
    threshold: `float`, optional (default: 0.1)
    row_threshold: `bool`, optional (default: True)
        If true, perform row-wise thresholding; otherwise, column-wise.

    Returns
    -------
    Rescaled np.array matrix
    """

    # print("V1")
    # t1=time.time()
    if ssp.issparse(input_matrix):
        input_matrix = input_matrix.A
        # print("Turn the sparse matrix into numpy array")
        # print(f"Time-1: {time.time()-t1}")

    max_vector = np.max(input_matrix, int(row_threshold))
    for j in range(len(max_vector)):
        if row_threshold:
            idx = input_matrix[j, :] < threshold * max_vector[j]
            input_matrix[j, idx] = 0
        else:
            idx = input_matrix[:, j] < threshold * max_vector[j]
            input_matrix[idx, j] = 0
    # print(f"matrix_row_or_column_thresholding time:{time.time()-t1}")
    return input_matrix


# This is slower due to a step of copying
def matrix_row_or_column_thresholding_v0(
    input_matrix, threshold=0.1, row_threshold=True
):
    """
    Row or column-wise thresholding a matrix

    Set entries in a given row (column) to be zero, if its value is below threshold*max(row_vector).

    Parameters
    ----------
    input_matrix: `np.array`
    threshold: `float`, optional (default: 0.1)
    row_threshold: `bool`, optional (default: True)
        If true, perform row-wise thresholding; otherwise, column-wise.

    Returns
    -------
    Rescaled np.array matrix
    """
    print("V0")
    if ssp.issparse(input_matrix):
        input_matrix = input_matrix.A

    output_matrix = input_matrix.copy()
    max_vector = np.max(input_matrix, int(row_threshold))
    for j in range(len(max_vector)):
        # if j%2000==0: logg.hint(j)
        if row_threshold:
            idx = input_matrix[j, :] < threshold * max_vector[j]
            output_matrix[j, idx] = 0
        else:
            idx = input_matrix[:, j] < threshold * max_vector[j]
            output_matrix[idx, j] = 0

    return output_matrix


def get_pca(E, base_ix=[], numpc=50, keep_sparse=False, normalize=True, random_state=0):
    """
    Run PCA on the count matrix E, gene-level normalizing if desired.

    By default, it performs z-score transformation for each gene across all cells, i.e.,
    a gene normalization, before computing PCA. (There is currently no consensus on doing
    this or not. In scanpy, after count normalization (a per-cell normalization), it assumes
    that the individual gene counts in a cell is log-normally distributed, and performs a
    log-transformation before computing PCA. The z-score transformation is gene-specific,
    while the log-transformation is not.)

    Parameters
    ----------
    E: `sp.spmatrix`
        sparse count matrix
    base_ix: `np.array`
        List of column id's to sub-sample the matrix
    numpc: `int`, optional (default: 50)
        Number of principal components to keep
    keep_sparse: `bool`, optional (default: False)
        If true, do not subtract the mean, but just divide by
        standard deviation, before running PCA.
        If false, subtract the mean and then divide by standard deviation,
        thus performing Zscore transformation, before running PCA
    normalize: `bool`, optional (default: True)
        Perform Zscore transformation if keep_sparse=True,
        Otherwise, only rescale by the standard deviation.
    random_state: `int`, optional (default: 0)
        Random seed for PCA

    Returns
    -------
    PCA coordinates
    """

    # If keep_sparse is True, gene-level normalization maintains sparsity
    #     (no centering) and TruncatedSVD is used instead of normal PCA.

    if len(base_ix) == 0:
        base_ix = np.arange(E.shape[0])

    if keep_sparse:
        if normalize:  # normalize variance
            zstd = np.sqrt(sparse_var(E[base_ix, :]))
            Z = sparse_rowwise_multiply(E.T, 1 / zstd).T
        else:
            Z = E
        pca = TruncatedSVD(n_components=numpc, random_state=random_state)

    else:
        if normalize:
            zmean = E[base_ix, :].mean(0)
            zstd = np.sqrt(sparse_var(E[base_ix, :]))
            Z = sparse_rowwise_multiply((E - zmean).T, 1 / zstd).T
        else:
            Z = E
        pca = PCA(n_components=numpc, random_state=random_state)

    pca.fit(Z[base_ix, :])
    return pca.transform(Z)


########## GENE FILTERING


def runningquantile(x, y, p, nBins):
    """calculate the quantile of y in bins of x"""

    ind = np.argsort(x)
    x = x[ind]
    y = y[ind]

    dx = (x[-1] - x[0]) / nBins
    xOut = np.linspace(x[0] + dx / 2, x[-1] - dx / 2, nBins)

    yOut = np.zeros(xOut.shape)

    for i in range(len(xOut)):
        ind = np.nonzero((x >= xOut[i] - dx / 2) & (x < xOut[i] + dx / 2))[0]
        if len(ind) > 0:
            yOut[i] = np.percentile(y[ind], p)
        else:
            if i > 0:
                yOut[i] = yOut[i - 1]
            else:
                yOut[i] = np.nan

    return xOut, yOut


def get_vscores(E, min_mean=0, nBins=50, fit_percentile=0.1, error_wt=1):
    """
    Calculate v-score (above-Poisson noise statistic) for genes in the input sparse counts matrix
    Return v-scores and other stats
    """

    ncell = E.shape[0]

    mu_gene = E.mean(axis=0).A.squeeze()
    gene_ix = np.nonzero(mu_gene > min_mean)[0]
    mu_gene = mu_gene[gene_ix]

    tmp = E[:, gene_ix]
    tmp.data **= 2
    var_gene = tmp.mean(axis=0).A.squeeze() - mu_gene**2
    del tmp
    FF_gene = var_gene / mu_gene

    data_x = np.log(mu_gene)
    data_y = np.log(FF_gene / mu_gene)

    x, y = runningquantile(data_x, data_y, fit_percentile, nBins)
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]

    gLog = lambda input: np.log(input[1] * np.exp(-input[0]) + input[2])
    h, b = np.histogram(np.log(FF_gene[mu_gene > 0]), bins=200)
    b = b[:-1] + np.diff(b) / 2
    max_ix = np.argmax(h)
    c = np.max((np.exp(b[max_ix]), 1))
    errFun = lambda b2: np.sum(abs(gLog([x, c, b2]) - y) ** error_wt)
    b0 = 0.1
    b = fmin(func=errFun, x0=[b0], disp=False)
    a = c / (1 + b) - 1

    v_scores = FF_gene / ((1 + a) * (1 + b) + b * mu_gene)
    CV_eff = np.sqrt((1 + a) * (1 + b) - 1)
    CV_input = np.sqrt(b)

    return v_scores, CV_eff, CV_input, gene_ix, mu_gene, FF_gene, a, b


def filter_genes(
    E,
    base_ix=[],
    min_vscore_pctl=85,
    min_counts=3,
    min_cells=3,
    show_vscore_plot=False,
    sample_name="",
):
    """
    Filter genes by expression level and variability

    Parameters
    ----------
    E: `sp.spmatrix`
        sparse count matrix
    base_ix: `np.array`
        List of column id's to sub-sample the matrix
    min_counts: int, optional (default: 3)
        Minimum number of UMIs per cell to be considered for selecting highly variable genes.
    min_cells: int, optional (default: 3)
        The minimum number of cells per gene to be considered for selecting highly variable genes.
    min_vscore_pctl: int, optional (default: 85)
        Genes wht a variability percentile higher than this threshold are marked as
        highly variable genes for dimension reduction. Range: [0,100]
    show_vscore_plot: `bool`, optional (default: False)
        If true, show the vscore plot for all genes
    sample_name: `str`, optional (default: '')
        Name of the plot title.

    Returns
    -------
    List of filtered gene indices (id's)
    """

    if len(base_ix) == 0:
        base_ix = np.arange(E.shape[0])

    Vscores, CV_eff, CV_input, gene_ix, mu_gene, FF_gene, a, b = get_vscores(
        E[base_ix, :]
    )
    ix2 = Vscores > 0
    Vscores = Vscores[ix2]
    gene_ix = gene_ix[ix2]
    mu_gene = mu_gene[ix2]
    FF_gene = FF_gene[ix2]
    min_vscore = np.percentile(Vscores, min_vscore_pctl)
    ix = ((E[:, gene_ix] >= min_counts).sum(0).A.squeeze() >= min_cells) & (
        Vscores >= min_vscore
    )

    x_min = 0.5 * np.min(mu_gene)
    x_max = 2 * np.max(mu_gene)
    Mean_value = x_min * np.exp(np.log(x_max / x_min) * np.linspace(0, 1, 100))
    FanoFactor = (1 + a) * (1 + b) + b * Mean_value

    if show_vscore_plot:
        plt.figure(figsize=(4, 3))
        plt.scatter(
            np.log10(mu_gene), np.log10(FF_gene), c=[[0.8, 0.8, 0.8]], alpha=0.3, s=3
        )
        plt.scatter(
            np.log10(mu_gene)[ix], np.log10(FF_gene)[ix], c=[[0, 0, 0]], alpha=0.3, s=3
        )
        plt.plot(np.log10(Mean_value), np.log10(FanoFactor))
        plt.title(sample_name)
        plt.xlabel("log10(mean)")
        plt.ylabel("log10(Fano factor)")
        plt.show()

    if FanoFactor[-1] < FanoFactor[0]:
        logg.warn(
            f"The estimated Fano factor is NOT in expected form, which would affect the results.\n"
            "Please make sure that the count matrix adata.X is NOT log-transformed."
        )
    return gene_ix[ix]


# We found that this does not work
def remove_corr_genes(
    E, gene_list, exclude_corr_genes_list, test_gene_idx, min_corr=0.1
):
    """
    Remove signature-correlated genes from a list of test genes

    Parameters
    ----------
    E: ssp.csc_matrix, shape (n_cells, n_genes)
        full counts matrix
    gene_list: numpy array, shape (n_genes,)
        full gene list
    exclude_corr_genes_list: list of list(s)
        Each sublist is used to build a signature. Test genes correlated
        with this signature will be removed
    test_gene_idx: 1-D numpy array
        indices of genes to test for correlation with the
        gene signatures from exclude_corr_genes_list
    min_corr: float (default=0.1)
        Test genes with a Pearson correlation of min_corr or higher
        with any of the gene sets from exclude_corr_genes_list will
        be excluded

    Returns
    -------
    Numpy array of gene indices (a subset of test_gene_idx) that are not correlated with any of the gene signatures
    """

    seed_ix_list = []
    for l in exclude_corr_genes_list:
        seed_ix_list.append(
            np.array([i for i in range(len(gene_list)) if gene_list[i] in l], dtype=int)
        )

    exclude_ix = []
    for iSet in range(len(seed_ix_list)):
        seed_ix = seed_ix_list[iSet][
            E[:, seed_ix_list[iSet]].sum(axis=0).A.squeeze() > 0
        ]
        if type(seed_ix) is int:
            seed_ix = np.array([seed_ix], dtype=int)
        elif type(seed_ix[0]) is not int:
            seed_ix = seed_ix[0]
        indat = E[:, seed_ix]
        tmp = sparse_zscore(indat)
        tmp = tmp.sum(1).A.squeeze()

        c = np.zeros(len(test_gene_idx))
        for iG in range(len(c)):
            c[iG], _ = scipy.stats.pearsonr(tmp, E[:, test_gene_idx[iG]].A.squeeze())

        exclude_ix.extend(
            [test_gene_idx[i] for i in range(len(test_gene_idx)) if (c[i]) >= min_corr]
        )
    exclude_ix = np.array(exclude_ix)

    return np.array([g for g in test_gene_idx if g not in exclude_ix], dtype=int)


#################################################################

# check if a given id is in the list L2 (day 24 or 46), or L4 (day26)
# a conversion algorithm
def converting_id_from_fullSpace_to_subSpace(
    query_id_array_fullSpace, subSpace_id_array_inFull
):
    """
    Convert indices in the full space to those in the subspace.

    Parameters
    ----------
    query_id_array_fullSpace: `np.array` or `list`
        Indices in the full space
    subSpace_id_array_inFull: `np.array` or `list`
        Indices of a targeted sub population in the full space

    Returns
    -------
    np.array(query_id_inSub): `np.array`
        A converted np.array of indices in the subspace
    query_success: `np.array`
        A bool array of conversion success
    """

    id_sub = np.array(subSpace_id_array_inFull)
    query_id_inSub = []
    query_success = np.zeros(len(query_id_array_fullSpace), dtype=bool)
    # check one by one
    for j, id_full in enumerate(query_id_array_fullSpace):
        temp = np.nonzero(id_sub == id_full)[0]
        if len(temp) > 0:
            query_success[j] = True
            query_id_inSub.append(temp[0])

    return np.array(query_id_inSub), query_success


def converting_id_from_subSpace_to_fullSpace(
    query_id_array_subSpace, subSpace_id_array_inFull
):
    """
    Parameters
    ----------
    query_id_array_subSpace: `np.array` or `list`
        Indices in the sub space
    subSpace_id_array_inFull: `np.array` or `list`
        Indices of a targeted sub population in the full space

    Returns
    -------
    A converted np.array of indices in the full space
    """

    return np.array(subSpace_id_array_inFull)[query_id_array_subSpace]


def analyze_selected_fates(state_info, selected_fates):
    """
    Analyze selected fates.

    We return only valid fate clusters.

    Parameters
    ----------
    selected_fates: `list`
        List of selected fate clusters.
    state_info: `list`
        The state_info vector.

    Returns
    -------
    mega_cluster_list: `list`, shape (n_fate)
        The list of names for the mega cluster. This is relevant when
        `selected_fates` has a nested structure.
    valid_fate_list: `list`, shape (n_fate)
        It is the same as selected_fates, could contain a nested list
        of fate clusters. It screens for valid fates, though.
    fate_array_flat: `list` shape (>n_fate)
        List of all selected fate clusters. It flattens the selected_fates if it contains
        nested structure, and allows only valid clusters.
    sel_index_list: `list`, shape (n_fate)
        List of selected cell indexes for each merged cluster.
    """

    state_info = np.array(state_info)
    valid_state_annot = list(set(state_info))
    if type(selected_fates) is str:
        selected_fates = [selected_fates]
    if selected_fates is None:
        selected_fates = valid_state_annot

    fate_array_flat = []  # a flatten list of cluster names
    valid_fate_list = (
        []
    )  # a list of cluster lists, each cluster list is a macro cluster
    mega_cluster_list = []  # a list of string description for the macro cluster
    sel_index_list = []
    for xx in selected_fates:
        if type(xx) is list:
            valid_fate_list.append(xx)
            des_temp = ""
            temp_idx = np.zeros(len(state_info), dtype=bool)
            for zz in xx:
                if zz in valid_state_annot:
                    fate_array_flat.append(zz)
                    if des_temp != "":
                        des_temp = des_temp + "_"

                    des_temp = des_temp + str(zz)
                    temp_idx = temp_idx | (state_info == zz)
                else:
                    raise ValueError(
                        f"{zz} is not a valid cluster name. Please select from: {valid_state_annot}"
                    )
            mega_cluster_list.append(des_temp)
            sel_index_list.append(temp_idx)
        else:
            if xx in valid_state_annot:
                valid_fate_list.append([xx])

                fate_array_flat.append(xx)
                mega_cluster_list.append(str(xx))
            else:
                raise ValueError(
                    f"{xx} is not a valid cluster name. Please select from: {valid_state_annot}"
                )
                mega_cluster_list.append("")

            temp_idx = state_info == xx
            sel_index_list.append(temp_idx)

    # exclude invalid clusters
    mega_cluster_list = np.array(mega_cluster_list)
    fate_array_flat = np.array(fate_array_flat)
    sel_index_list = np.array(sel_index_list)
    valid_idx = mega_cluster_list != ""

    return (
        mega_cluster_list[valid_idx],
        valid_fate_list,
        fate_array_flat,
        sel_index_list[valid_idx],
    )


# v1, the new methods, more options.
def compute_shortest_path_distance(
    adata,
    num_neighbors_target=5,
    mode="distances",
    limit=np.inf,
    method="umap",
    normalize=True,
    use_existing_KNN_graph=False,
):
    """
    Compute shortest path distance from raw data.

    The distance matrix has two modes: 'connectivity' or 'distance'.
    We found that the 'connectivity' version is sensitive to local cell
    density heterogeneity, and the 'distance' version is more robust.
    This discrepancy might be due to that the KNN graph construction does not
    directly take into account local density heterogeneity.

    The default is the UMAP method, which takes into account local
    density heterogeneity into account when constructing the KNN graph.

    Parameters
    ----------
    adata: :class:`~anndata.AnnaData` object
    num_neighbors_target: `int`, optional (default: 5)
        Used to construct the KNN graph.
    mode: `str`, optional (default: 'distance')
        Options: {'distance','connectivity')
    limit: `float`, optional (default: np.inf)
        If the distance is about this, stop computation, and set
        the distance beyong this limist by `limit`. This can speed up computation.
    method: `str`, optional (default: 'umap')
        The method to construct the KNN graph. Options: {'umap','gauss','others'}.
        The frist two methods are based on sc.pp.neighbors, while the last is from
        kneighbors_graph.
    normalize: `bool`, optional (default: True)
        Normalize the distance matrix by its maximum value.
    use_existing_KNN_graph: `bool`, optional (default: False)
        If true and adata.obsp['connectivities'], use the existing knn graph.
        This overrides all other parameters.

    Returns
    -------
    The normalized distance matrix is returned.
    """

    from scanpy.neighbors import neighbors

    if (not use_existing_KNN_graph) or ("connectivities" not in adata.obsp.keys()):

        if mode != "connectivities":
            mode = "distances"

        logg.hint(f"Chosen mode is {mode}")
        if method == "umap":
            neighbors(adata, n_neighbors=num_neighbors_target, method="umap")
            adj_matrix = adata.obsp[mode]

        elif method == "gauss":
            neighbors(adata, n_neighbors=num_neighbors_target, method="gauss")
            adj_matrix = adata.obsp[mode]

        else:
            if mode == "distances":
                mode = "distance"
            else:
                mode = "connectivity"
            data_matrix = adata.obsm["X_pca"]
            adj_matrix = kneighbors_graph(
                data_matrix, num_neighbors_target, mode=mode, include_self=True
            )

    else:
        logg.info(
            f"Use existing KNN graph at adata.obsp[{mode}] for generating the smooth matrix"
        )
        adj_matrix = adata.obsp[mode]

    ShortPath_dis = ssp.csgraph.dijkstra(
        csgraph=ssp.csr_matrix(adj_matrix), directed=False, return_predecessors=False
    )
    ShortPath_dis_max = np.nanmax(ShortPath_dis[ShortPath_dis != np.inf])
    ShortPath_dis[
        ShortPath_dis > ShortPath_dis_max
    ] = ShortPath_dis_max  # set threshold for shortest paths

    # Set normalized cost matrices based on shortest paths matrices at target and source spaces
    if normalize:
        ShortPath_dis_final = ShortPath_dis / ShortPath_dis.max()
    else:
        ShortPath_dis_final = ShortPath_dis

    return ShortPath_dis_final


def add_neighboring_cells_to_a_map(initial_idx, adata, neighbor_N=5):
    """
    Add neighboring cells to an initially selected population

    Parameters
    ----------
    initial_idx: `np.array`, shape (n_cell,)
        A boolean array for state selection.
    adata: :class:`~anndata.AnnData` object, shape (n_cell, n_genes)
    neighbor_N: `int`, optional (default: 5)
        Number of neighbors for KNN graph construction.

    Returns
    -------
    post_idx: `np.array`, shape (n_cell,)
        A boolean array of selected cell states post expansion.
    """

    initial_idx = initial_idx > 0
    # print(f"Initial: {np.sum(initial_idx)}")
    #     if (np.sum(initial_idx)<size_thresh) & (np.sum(initial_idx)>0):
    #         #n0=np.round(size_thresh/np.sum(initial_idx))
    #         #sc.pp.neighbors(adata, n_neighbors=int(n0)) #,method='gauss')
    #         output_idx=adata.uns['neighbors']['connectivities'][initial_idx].sum(0).A.flatten()>0
    #         initial_idx=initial_idx | output_idx

    from scanpy.neighbors import neighbors

    neighbors(adata, n_neighbors=neighbor_N)  # ,method='gauss')
    output_idx = adata.obsp["connectivities"][initial_idx].sum(0).A.flatten() > 0
    post_idx = initial_idx | output_idx
    # print(f"Final: {np.sum(post_idx)}")

    return post_idx


def get_hierch_order(hm, dist_metric="euclidean", linkage_method="ward"):
    """
    This is used to order the barcode in generating the barcode heatmap.
    """
    np.random.seed(0)
    D = pdist(hm, dist_metric)
    Z = linkage(D, linkage_method)
    n = len(Z) + 1
    cache = dict()
    for k in range(len(Z)):
        c1, c2 = int(Z[k][0]), int(Z[k][1])
        c1 = [c1] if c1 < n else cache.pop(c1)
        c2 = [c2] if c2 < n else cache.pop(c2)
        cache[n + k] = c1 + c2
    o = np.array(cache[2 * len(Z)])
    return o


def above_the_line(x_array, x1, x2):
    """
    Return states above a specified line defined by (x1, x2).

    We assume that a state has only two coordinates.

    Parameters
    ----------
    x_array: `np.array`
        A 2-d matrix. Usually, an embedding for data points.
    x1: `np.array`
        A list or array of two entries.
    x2: `np.array`
        A list or array of two entries.

    Returns
    -------
    A boolean array.
    """

    return (x_array[:, 1] - x1[1]) > ((x2[1] - x1[1]) / (x2[0] - x1[0])) * (
        x_array[:, 0] - x1[0]
    )


def save_map(adata):
    """
    Save the adata and print the filename prefix.

    The filename prefix `data_des` will be printed, and
    the saved file can be accessed again using this prefix.
    """

    data_des = adata.uns["data_des"][-1]
    # data_path=adata.uns['data_path'][0]
    data_path = settings.data_path

    # need to remove these, otherwise, it does not work
    for xx in [
        "fate_trajectory",
        "multiTime_cell_id_t1",
        "multiTime_cell_id_t2",
        "fate_map",
        "fate_bias",
        "fate_potency",
    ]:
        if xx in adata.uns.keys():
            adata.uns.pop(xx)

    file_name = os.path.join(data_path, f"{data_des}_adata_with_transition_map.h5ad")
    adata.write_h5ad(file_name, compression="gzip")
    print(f"Saved file: {file_name}")


def smooth_a_vector(
    adata,
    vector,
    round_of_smooth=5,
    use_full_Smatrix=True,
    trunca_threshold=0.001,
    compute_new=False,
    neighbor_N=20,
    show_groups=True,
    save_subset=True,
):
    """ """

    vector = np.array(vector)
    if len(vector) != adata.shape[0]:
        logg.error("The length of the vector does not match that of adata.shape[0]")
        return None

    data_des = adata.uns["data_des"][0]
    data_des_1 = adata.uns["data_des"][-1]
    data_path = settings.data_path
    if "sp_idx" in adata.uns.keys():
        sp_idx = adata.uns["sp_idx"]
    else:
        sp_idx = np.ones(adata.shape[0]).astype(bool)

    # trunca_threshold=0.001 # this value is only for reducing the computed matrix size for saving
    temp_str = "0" + str(trunca_threshold)[2:]

    if use_full_Smatrix:
        similarity_file_name = os.path.join(
            data_path,
            f"{data_des}_Similarity_matrix_with_all_cell_states_kNN{neighbor_N}_Truncate{temp_str}",
        )

        if not os.path.exists(similarity_file_name + f"_SM{round_of_smooth}.npz"):
            logg.error(
                f"Similarity matrix at given parameters have not been computed before! Fiale name: {similarity_file_name}"
            )
            logg.error(
                f"Please use other Tmap inference function to build the full similarity matrix at the corresponding smooth rounds, using adata_orig."
            )
            return None

    else:
        similarity_file_name = os.path.join(
            data_path,
            f"{data_des_1}_Similarity_matrix_with_selected_states_kNN{neighbor_N}_Truncate{temp_str}",
        )

    # we cannot force it to compute new at this time. Otherwise, if we use_full_Smatrix, the resulting similarity is actually from adata, thus not full similarity.

    re_compute = (not use_full_Smatrix) and (
        compute_new
    )  # re-compute only when not using full similarity
    similarity_matrix_full = tmap.generate_similarity_matrix(
        adata,
        similarity_file_name,
        round_of_smooth=round_of_smooth,
        neighbor_N=neighbor_N,
        truncation_threshold=trunca_threshold,
        save_subset=save_subset,
        compute_new_Smatrix=re_compute,
    )

    if use_full_Smatrix:
        # pdb.set_trace()
        similarity_matrix_full_sp = similarity_matrix_full[sp_idx][:, sp_idx]

    else:
        similarity_matrix_full_sp = similarity_matrix_full

    smooth_vector = similarity_matrix_full_sp * vector
    if show_groups:
        x_emb = adata.obsm["X_emb"][:, 0]
        y_emb = adata.obsm["X_emb"][:, 1]
        fig_width = settings.fig_width
        fig_height = settings.fig_height
        fig = plt.figure(figsize=(2 * fig_width, fig_height))
        ax = plt.subplot(1, 2, 1)
        pl.customized_embedding(x_emb, y_emb, vector, ax=ax)
        ax = plt.subplot(1, 2, 2)
        pl.customized_embedding(x_emb, y_emb, smooth_vector, ax=ax)

    return smooth_vector


def update_time_ordering(adata, updated_ordering=None, mode="force"):
    """
    Update the ordering of time points at adata.uns['time_ordering']

    Parameters
    ----------
    updated_ordering: `list`, optional (default: none)
        A list of distinct time points in ascending order.
        If not provided, sort the time variable directly.
        However, these time variables are string. Their sorting
        may not be correct.
    mode: `str`
        Options: {'force','auto'}. In the 'auto' mode, the algorithm only
        update the ordering if 'time_ordering' has not been computed before.
        The default method 'force' will always update the ordering.
    """

    time_info = list(set(adata.obs["time_info"]))
    if updated_ordering is not None:
        N_match = np.sum(np.in1d(time_info, updated_ordering))
        if (len(updated_ordering) != N_match) or (
            len(updated_ordering) != len(time_info)
        ):
            logg.error(
                "The provided time points are not correct (wrong length, or invalid value)"
            )
            logg.info(
                f"Please provide an ordering of all time points in ascending order. Available time points are: {time_info}"
            )
            return None

    else:
        updated_ordering = np.sort(time_info)

    if mode == "auto":
        if "time_ordering" in adata.uns.keys():
            time_ordering_0 = adata.uns["time_ordering"]
            N_match = np.sum(np.in1d(time_info, time_ordering_0))
            if (len(time_ordering_0) != N_match) or (
                len(time_ordering_0) != len(time_info)
            ):
                logg.warn(
                    "Pre-computed time_ordering does not include the right time points. Re-compute it!"
                )
                adata.uns["time_ordering"] = np.array(updated_ordering)
                if updated_ordering is None:
                    logg.info(
                        f"Current time ordering from simple sorting: {updated_ordering}"
                    )
            else:
                # do not update
                return None
        else:
            adata.uns["time_ordering"] = np.array(updated_ordering)
            if updated_ordering is None:
                logg.info(
                    f"Current time ordering from simple sorting: {updated_ordering}"
                )
    else:
        # always update
        adata.uns["time_ordering"] = np.array(updated_ordering)
        if updated_ordering is None:
            logg.info(f"Current time ordering from simple sorting: {updated_ordering}")


def check_adata_structure(adata):
    """
    Check whether the adata has the right structure.
    """

    flag = True
    if not ("X_pca" in adata.obsm.keys()):
        logg.error("*X_pca* missing from adata.obsm")
        flag = False

    if not ("X_emb" in adata.obsm.keys()):
        logg.error("*X_emb* missing from adata.obsm")
        flag = False

    if not ("X_clone" in adata.obsm.keys()):
        logg.error("*X_clone* missing from adata.obsm")
        flag = False

    if not ("time_info" in adata.obs.keys()):
        logg.error("*time_info* missing from adata.obs")
        flag = False

    if not ("state_info" in adata.obs.keys()):
        logg.error("*state_info* missing from adata.obs")
        flag = False

    if flag:
        print("The adata structure looks fine!")


def save_preprocessed_adata(adata, data_des=None):
    """
    Save preprocessed adata.

    It will remove unneeded entries, and use the default
    prefix (data_des) to save the results if a new data_des is not provided.
    """

    if data_des is None:
        data_des = adata.uns["data_des"][0]

    data_path = settings.data_path

    for xx in [
        "fate_trajectory",
        "multiTime_cell_id_t1",
        "multiTime_cell_id_t2",
        "fate_map",
        "fate_bias",
        "fate_potency",
        "transition_map",
        "intraclone_transition_map",
        "clonal_transition_map",
        "OT_transition_map",
        "HighVar_transition_map",
        "Tmap_cell_id_t1",
        "Tmap_cell_id_t2",
        "available_map",
        "clonal_cell_id_t1",
        "clonal_cell_id_t2",
    ]:
        if xx in adata.uns.keys():
            adata.uns.pop(xx)
        if xx in adata.obs.keys():
            adata.obs.pop(xx)
        if xx in adata.obsm.keys():
            adata.obsm.pop(xx)

    # check_list=list(adata.obsm.keys())
    # for xx in  check_list:
    #     if xx not in ['X_clone', 'X_emb', 'X_pca']:
    #         adata.obsm.pop(xx)

    # check_list=list(adata.obs.keys())
    # for xx in  check_list:
    #     if xx not in ['state_info', 'time_info']:
    #         adata.obs.pop(xx)

    file_name = os.path.join(data_path, f"{data_des}_adata_preprocessed.h5ad")
    adata.write_h5ad(file_name, compression="gzip")
    print(f"Saved file: {file_name}")


def load_saved_adata_with_key(data_des):
    """
    Load pre-saved adata based on the prefix 'data_des'
    """

    data_path = settings.data_path
    # print(f"Load data: data_des='{data_des}'")
    file_name = os.path.join(data_path, f"{data_des}_adata_with_transition_map.h5ad")
    if os.path.exists(file_name):
        adata = read(file_name)
        return adata
    else:
        logg.error(f"The file does not existed yet")


def check_available_map(adata):
    """
    Check available transition map.

    Update adata.uns['available_map'].
    """

    available_map = []
    for xx in adata.uns.keys():
        if "transition_map" in xx:
            available_map.append(xx)
    adata.uns["available_map"] = available_map


def switch_adata_representation(adata, to_new=True):
    if to_new:
        adata.obsm["X_clone"] = adata.obsm["cell_by_clone_matrix"]
        adata.obs["state_info"] = adata.obs["state_annotation"]
        # adata.uns['data_des']=['paper_OneTimeClone_t*pos_17*pos_21*D27']
        adata.obsm.pop("cell_by_clone_matrix")
        adata.obs.pop("state_annotation")
    else:
        adata.obsm["cell_by_clone_matrix"] = adata.obsm["X_clone"]
        adata.obs["state_annotation"] = adata.obs["state_info"]
        # adata.uns['data_des']=['paper_OneTimeClone_t*pos_17*pos_21*D27']
        adata.obsm.pop("X_clone")
        adata.obs.pop("state_info")


def selecting_cells_by_time_points(time_info, selected_time_points):
    """
    Check validity of selected time points, and return the selected index.

    selected_time_points can be either a string or a list.

    If selected_time_points=[], we select all cell states.
    """

    time_info = np.array(time_info)
    valid_time_points = set(time_info)
    if selected_time_points is not None:
        if type(selected_time_points) is str:
            selected_times = [selected_time_points]
        else:
            selected_times = list(selected_time_points)

        sp_idx = np.zeros(len(time_info), dtype=bool)
        for xx in selected_times:
            if xx not in valid_time_points:
                logg.error(f"{xx} is not a valid time point.")
            sp_id_temp = np.nonzero(time_info == xx)[0]
            sp_idx[sp_id_temp] = True
    else:
        sp_idx = np.ones(len(time_info), dtype=bool)

    return sp_idx


def check_available_clonal_info(adata):

    X_clone = adata.obsm["X_clone"]
    time_info = adata.obs["time_info"]

    update_time_ordering(adata, mode="auto")

    # record time points with clonal information
    if ssp.issparse(X_clone):
        clone_N_per_cell = X_clone.sum(1).A.flatten()
    else:
        clone_N_per_cell = X_clone.sum(1)

    clonal_time_points = []
    for xx in list(set(time_info)):
        idx = np.array(time_info) == xx
        if np.sum(clone_N_per_cell[idx]) > 0:
            clonal_time_points.append(xx)

    time_ordering = adata.uns["time_ordering"]
    sel_idx_temp = np.in1d(time_ordering, clonal_time_points)
    clonal_time_points = time_ordering[sel_idx_temp]
    adata.uns["clonal_time_points"] = clonal_time_points


def check_available_choices(adata):
    """
    Check available parameter choices.

    Also update adata.uns['available_map'] and adata.uns['clonal_time_points'].
    """

    check_available_map(adata)
    available_map = adata.uns["available_map"]

    check_available_clonal_info(adata)
    clonal_time_points = adata.uns["clonal_time_points"]

    print("Available transition maps:", available_map)

    if "state_info" in adata.obs.keys():
        print("Available clusters:", list(set(adata.obs["state_info"])))
    else:
        print("No state_info yet")

    if "time_ordering" in adata.uns.keys():
        print("Available time points:", adata.uns["time_ordering"])
    else:
        print("No time_ordering yet")
    print("Clonal time points:", clonal_time_points)


def compute_pca(m1, m2, n_components):
    matrices = list()
    matrices.append(m1 if not ssp.isspmatrix(m1) else m1.toarray())
    matrices.append(m2 if not ssp.isspmatrix(m2) else m2.toarray())
    x = np.vstack(matrices)
    mean_shift = x.mean(axis=0)
    x = x - mean_shift
    n_components = min(n_components, x.shape[0])  # n_components must be <= ncells
    pca = PCA(n_components=n_components, random_state=58951)
    pca.fit(x.T)
    comp = pca.components_.T
    m1_len = m1.shape[0]
    m2_len = m2.shape[0]
    pca_1 = comp[0:m1_len]
    pca_2 = comp[m1_len : (m1_len + m2_len)]
    return pca_1, pca_2, pca, mean_shift


def compute_default_cost_matrix(a, b, eigenvals=None):

    if eigenvals is not None:
        a = a.dot(eigenvals)
        b = b.dot(eigenvals)

    cost_matrix = pairwise.pairwise_distances(
        a.toarray() if ssp.isspmatrix(a) else a,
        b.toarray() if ssp.isspmatrix(b) else b,
        metric="sqeuclidean",
        n_jobs=-1,
    )
    cost_matrix = cost_matrix / np.median(cost_matrix)
    return cost_matrix


def compute_gene_exp_distance(adata, p0_indices, p1_indices, pc_n=30):
    """
    Compute the gene expression distance between t0 and t1 states.

    p0_indices could be either a boolean array or index array
    """
    p0 = adata[p0_indices, :]
    p1 = adata[p1_indices, :]
    p0_x, p1_x, pca, mean = compute_pca(p0.X, p1.X, pc_n)
    eigenvals = np.diag(pca.singular_values_)
    # gene_exp_dis_t0 = compute_default_cost_matrix(p0_x, p0_x, eigenvals)
    # gene_exp_dis_t1 = compute_default_cost_matrix(p1_x, p1_x, eigenvals)
    gene_exp_dis_t0t1 = compute_default_cost_matrix(p0_x, p1_x, eigenvals)
    return gene_exp_dis_t0t1


def update_data_description(adata, append_info=None, data_des=None):
    """
    Update data_des, a string to distinguish different datasets
    """

    if "data_des" not in adata.uns.keys():
        if data_des is None:
            logg.warn("data_des not set. Set to be [cospar]")
            adata.uns["data_des"] = ["cospar"]
        else:
            adata.uns["data_des"] = [data_des]
    else:
        if data_des is not None:
            adata.uns["data_des"][-1] = data_des

    if append_info is not None:
        data_des_0 = adata.uns["data_des"][-1]
        data_des_1 = f"{data_des_0}_{append_info}"
        adata.uns["data_des"][-1] = data_des_1


def set_up_folders(data_path_new=None, figure_path_new=None):
    from pathlib import Path, PurePath

    if data_path_new is not None:
        settings.data_path = data_path_new
    if figure_path_new is not None:
        settings.figure_path = figure_path_new

    if not Path(settings.data_path).is_dir():
        logg.info(f"creating directory {settings.data_path} for saving data")
        Path(settings.data_path).mkdir(parents=True)

    if not Path(settings.figure_path).is_dir():
        logg.info(f"creating directory {settings.figure_path} for saving figures")
        Path(settings.figure_path).mkdir(parents=True)


def get_X_clone_with_reference_ordering(
    clone_data_cell_id,
    clone_data_barcode_id,
    reference_cell_id=None,
    reference_clone_id=None,
    drop_duplicates=True,
    count_value=None,
):
    """
    Build the X_clone matrix from data.

    Convert the raw clonal data table (long format): [clone_data_cell_id,clone_data_barcode_id]
    to X_clone (wide format) based on the reference_cell_id.

    Parameters
    ----------
    clone_data_cell_id: `list`
        The list of cell id for each corresponding sequenced barcode.
    clone_data_barcode_id: `list`
        The list of barcode id from sequencing. It has the same shape as clone_data_cell_id.
    reference_cell_id: `list`
        A list of uniuqe cell id. X_clone will be generated based on its cell id ordering.
        This has to be provided to match the cell ordering in the adata object.
    reference_clone_id: `list`, optional (default: None)
        A list of uniuqe clone id. If provided, X_clone will be generated based on its barcode ordering.
    count_value: `list`, optional (default: None)
        A list of count values corresponding to clone_data_cell_id. If not set, it is default to 1.

    Returns
    -------
    X_clone: `ssp.sparse`
        The clonal data matrix, with the row in cell id, and column in barcode id.
    reference_clone_id: `list`
    """

    if reference_clone_id is None:
        reference_clone_id = list(set(clone_data_barcode_id))
    if reference_cell_id is None:
        reference_cell_id = list(set(clone_data_cell_id))
    if count_value is None:
        count_value = np.ones(len(clone_data_cell_id))

    reference_clone_id = np.array(reference_clone_id)
    clone_map = {x: j for j, x in enumerate(reference_clone_id)}
    reference_cell_id = np.array(reference_cell_id)
    cell_map = {x: j for j, x in enumerate(reference_cell_id)}
    count_value = np.array(count_value)

    df = pd.DataFrame(
        {"cell_id": clone_data_cell_id, "clone_id": clone_data_barcode_id}
    )
    if reference_cell_id is not None:
        df = df[(df["cell_id"].isin(reference_cell_id))]

    if reference_clone_id is not None:
        df = df[(df["clone_id"].isin(reference_clone_id))]

    if drop_duplicates:
        df = df.drop_duplicates()

    clone_data_cell_id = df["cell_id"]
    clone_data_barcode_id = df["clone_id"]
    clone_data_cell_id = np.array(clone_data_cell_id)
    clone_data_barcode_id = np.array(clone_data_barcode_id)

    X_clone_row = []
    X_clone_col = []
    X_clone_val = []
    for j in tqdm(range(len(clone_data_cell_id))):
        X_clone_row.append(cell_map[clone_data_cell_id[j]])
        X_clone_col.append(clone_map[clone_data_barcode_id[j]])
        X_clone_val.append(count_value[j])

    X_clone = ssp.coo_matrix(
        (X_clone_val, (X_clone_row, X_clone_col)),
        shape=(len(reference_cell_id), len(reference_clone_id)),
    )
    X_clone = ssp.csr_matrix(X_clone)

    sp_idx = X_clone.sum(0).A.flatten() > 0
    return X_clone[:, sp_idx], reference_clone_id[sp_idx], reference_cell_id


def parse_output_choices(adata, key_word, where="obs", interrupt=True):
    if where == "obs":
        raw_choices = [xx for xx in adata.obs.keys() if xx.startswith(f"{key_word}")]
    else:
        raw_choices = [xx for xx in adata.uns.keys() if xx.startswith(f"{key_word}")]

    if (interrupt) and (len(raw_choices) == 0):
        raise ValueError(
            f"{key_word} has not been computed yet. Please run the counterpart function at cs.tl.XXX using the appropriate source name."
        )

    available_choices = []
    for xx in raw_choices:
        y = xx.split(f"{key_word}")[1]
        if y.startswith("_"):
            y = y[1:]
        available_choices.append(y)

    return available_choices


def rename_list(old_names, new_names):
    if new_names is None:
        new_names = old_names

    if len(old_names) != len(new_names):
        logg.warn(
            "The new name list does not have the same length as old name list. Rename operation aborted"
        )
        new_names = old_names
    return new_names


def check_input_parameters(adata, **kwargs):
    keys = kwargs.keys()
    check_available_clonal_info(adata)
    clonal_time_points_0 = np.array(adata.uns["clonal_time_points"])
    time_ordering = np.array(adata.uns["time_ordering"])
    if len(time_ordering) == 1:
        raise ValueError(
            "There is only one time point. Tmap inference requires at least 2 time points. Inference aborted."
        )

    if ("save_subset" in keys) and ("smooth_array" in keys):
        save_subset = kwargs["save_subset"]
        smooth_array = kwargs["smooth_array"]
        if save_subset:
            if not (
                np.all(np.diff(smooth_array) <= 0)
                and np.all(np.array(smooth_array) % 5 == 0)
            ):
                raise ValueError(
                    "The smooth_array contains numbers not multiples of 5 or not in descending order.\n"
                    "The correct form is like [20,15,10], or [10,10,10,5]."
                    "You can also set save_subset=False to explore arbitrary smooth_array structure."
                )

    if "data_des" not in adata.uns.keys():
        logg.warn("data_des not defined at adata.uns['data_des']. Set to be 'cospar'")
        adata.uns["data_des"] = ["cospar"]

    if "later_time_point" in keys:
        later_time_point = kwargs["later_time_point"]
        if (later_time_point is not None) and (
            later_time_point not in clonal_time_points_0
        ):
            raise ValueError(
                f"later_time_point is not all among {clonal_time_points_0}. Computation aborted!"
            )

    if "initial_time_points" in keys:
        initial_time_points = kwargs["initial_time_points"]
        N_valid_time = np.sum(np.in1d(time_ordering, initial_time_points))
        if (N_valid_time != len(initial_time_points)) or (N_valid_time < 1):
            raise ValueError(
                f"The 'initial_time_points' are not all valid. Please select from {time_ordering}"
            )

    if "clonal_time_points" in keys:
        clonal_time_points = kwargs["clonal_time_points"]
        N_valid_time = np.sum(np.in1d(clonal_time_points_0, clonal_time_points))
        if (N_valid_time != len(clonal_time_points)) or (N_valid_time < 2):
            raise ValueError(
                f"Selected time points are not all among {clonal_time_points_0}, or less than 2 time points are selected. Computation aborted!"
            )
