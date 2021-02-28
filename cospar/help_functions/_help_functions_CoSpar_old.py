import numpy as np
import scipy
import scipy.stats
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise
#import time
#import os
#import json
#from datetime import datetime
#import matplotlib.pyplot as plt
import ot.bregman as otb
import scipy.sparse as ssp
import scanpy as sc
import pandas as pd
import statsmodels.sandbox.stats.multicomp
from scipy.spatial.distance import pdist
from fastcluster import linkage
from .. import settings
from .. import logging as logg

#import scipy.stats

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

    
    gene_mask = ((ad.X[mask1,:]>0).sum(0).A.squeeze()/mask1.sum() > min_frac_expr) | ((ad.X[mask2,:]>0).sum(0).A.squeeze()/mask2.sum() > min_frac_expr)
    #print(gene_mask.sum())
    E1 = ad.X[mask1,:][:,gene_mask].toarray()
    E2 = ad.X[mask2,:][:,gene_mask].toarray()
    
    m1 = E1.mean(0) + pseudocount
    m2 = E2.mean(0) + pseudocount
    r = np.log2(m1 / m2)
    
    pv = np.zeros(gene_mask.sum())
    for ii,iG in enumerate(np.nonzero(gene_mask)[0]):
        pv[ii] = scipy.stats.ranksums(E1[:,ii], E2[:,ii])[1]
    pv = statsmodels.sandbox.stats.multicomp.multipletests(pv, alpha=0.05, method='fdr_bh',)[1]
    sort_idx=np.argsort(pv)
    
    df = pd.DataFrame({
        'gene': ad.var_names.values.astype(str)[gene_mask][sort_idx],
        'pv': pv[sort_idx],
        'mean_1': m1[sort_idx] - pseudocount, 
        'mean_2': m2[sort_idx] - pseudocount, 
        'ratio': r[sort_idx]
    })
    
    return df


########## USEFUL SPARSE FUNCTIONS

def sparse_var(E, axis=0):
    """ calculate variance across the specified axis of a sparse matrix"""

    mean_gene = E.mean(axis=axis).A.squeeze()
    tmp = E.copy()
    tmp.data **= 2
    return tmp.mean(axis=axis).A.squeeze() - mean_gene ** 2

def mean_center(E, column_means=None):
    """ mean-center columns of a sparse matrix """

    if column_means is None:
        column_means = E.mean(axis=0)
    return E - column_means

def normalize_variance(E, column_stdevs=None):
    """ variance-normalize columns of a sparse matrix """

    if column_stdevs is None:
        column_stdevs = np.sqrt(sparse_var(E, axis=0))
    return sparse_rowwise_multiply(E.T, 1 / column_stdevs).T

def sparse_zscore(E, gene_mean=None, gene_stdev=None):
    """ z-score normalize each column of a sparse matrix """
    if gene_mean is None:
        gene_mean = E.mean(0)
    if gene_stdev is None:
        gene_stdev = np.sqrt(sparse_var(E))
    return sparse_rowwise_multiply((E - gene_mean).T, 1/gene_stdev).T

def sparse_rowwise_multiply(E, a):
    """ 
    multiply each row of sparse matrix by a scalar 

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
    if nrow!=a.shape[0]:
        logg.error("Dimension mismatch, multiplication failed")
        return E
    else:
        w = ssp.lil_matrix((nrow, nrow))
        w.setdiag(a)
        return w * E


def sparse_column_multiply(E, a):
    """ 
    multiply each columns of sparse matrix by a scalar 

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
    if ncol!=a.shape[0]:
        logg.error("Dimension mismatch, multiplication failed")
        return E
    else:
        w = ssp.lil_matrix((ncol, ncol))
        w.setdiag(a)
        return (ssp.csr_matrix(E)*w)


def matrix_row_or_column_thresholding(input_matrix,threshold=0.1,row_threshold=True):
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

    if ssp.issparse(input_matrix): input_matrix=input_matrix.A

    output_matrix=input_matrix.copy()
    max_vector=np.max(input_matrix,int(row_threshold))
    for j in range(len(max_vector)):
        #if j%2000==0: logg.hint(j)
        if row_threshold:
            idx=input_matrix[j,:]<threshold*max_vector[j]
            output_matrix[j,idx]=0
        else:
            idx=input_matrix[:,j]<threshold*max_vector[j]
            output_matrix[idx,j]=0

    return output_matrix


def get_pca(E, base_ix=[], numpc=50, keep_sparse=False, normalize=True, random_state=0):
    """
    Run PCA on the counts matrix E, gene-level normalizing if desired.

    By default, it performs z-score transformation for each gene across all cells, i.e., 
    a gene normalization, before computing PCA. (There is currently no concensus on doing 
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
        Number of principle components to keep
    keep_sparse: `bool`, optional (default: False)
        If true, do not substract the mean, but just divide by 
        standard deviation, before running PCA. 
        If false, substract the mean and then divide by standard deviation, 
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
        if normalize: # normalize variance
            zstd = np.sqrt(sparse_var(E[base_ix,:]))
            Z = sparse_rowwise_multiply(E.T, 1 / zstd).T
        else:
            Z = E
        pca = TruncatedSVD(n_components=numpc, random_state=random_state)

    else:
        if normalize:
            zmean = E[base_ix,:].mean(0)
            zstd = np.sqrt(sparse_var(E[base_ix,:]))
            Z = sparse_rowwise_multiply((E - zmean).T, 1/zstd).T
        else:
            Z = E
        pca = PCA(n_components=numpc, random_state=random_state)

    pca.fit(Z[base_ix,:])
    return pca.transform(Z)



########## GENE FILTERING

def runningquantile(x, y, p, nBins):
    """ calculate the quantile of y in bins of x """

    ind = np.argsort(x)
    x = x[ind]
    y = y[ind]

    dx = (x[-1] - x[0]) / nBins
    xOut = np.linspace(x[0]+dx/2, x[-1]-dx/2, nBins)

    yOut = np.zeros(xOut.shape)

    for i in range(len(xOut)):
        ind = np.nonzero((x >= xOut[i]-dx/2) & (x < xOut[i]+dx/2))[0]
        if len(ind) > 0:
            yOut[i] = np.percentile(y[ind], p)
        else:
            if i > 0:
                yOut[i] = yOut[i-1]
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

    tmp = E[:,gene_ix]
    tmp.data **= 2
    var_gene = tmp.mean(axis=0).A.squeeze() - mu_gene ** 2
    del tmp
    FF_gene = var_gene / mu_gene

    data_x = np.log(mu_gene)
    data_y = np.log(FF_gene / mu_gene)

    x, y = runningquantile(data_x, data_y, fit_percentile, nBins)
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]

    gLog = lambda input: np.log(input[1] * np.exp(-input[0]) + input[2])
    h,b = np.histogram(np.log(FF_gene[mu_gene>0]), bins=200)
    b = b[:-1] + np.diff(b)/2
    max_ix = np.argmax(h)
    c = np.max((np.exp(b[max_ix]), 1))
    errFun = lambda b2: np.sum(abs(gLog([x,c,b2])-y) ** error_wt)
    b0 = 0.1
    b = scipy.optimize.fmin(func = errFun, x0=[b0], disp=False)
    a = c / (1+b) - 1


    v_scores = FF_gene / ((1+a)*(1+b) + b * mu_gene);
    CV_eff = np.sqrt((1+a)*(1+b) - 1);
    CV_input = np.sqrt(b);

    return v_scores, CV_eff, CV_input, gene_ix, mu_gene, FF_gene, a, b

def filter_genes(E, base_ix = [], min_vscore_pctl = 85, min_counts = 3, min_cells = 3, show_vscore_plot = False, sample_name = ''):
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
        Minimum number of cells per gene to be considered for selecting highly variable genes. 
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

    Vscores, CV_eff, CV_input, gene_ix, mu_gene, FF_gene, a, b = get_vscores(E[base_ix, :])
    ix2 = Vscores>0
    Vscores = Vscores[ix2]
    gene_ix = gene_ix[ix2]
    mu_gene = mu_gene[ix2]
    FF_gene = FF_gene[ix2]
    min_vscore = np.percentile(Vscores, min_vscore_pctl)
    ix = (((E[:,gene_ix] >= min_counts).sum(0).A.squeeze() >= min_cells) & (Vscores >= min_vscore))
    
    if show_vscore_plot:
        import matplotlib.pyplot as plt
        x_min = 0.5*np.min(mu_gene)
        x_max = 2*np.max(mu_gene)
        xTh = x_min * np.exp(np.log(x_max/x_min)*np.linspace(0,1,100))
        yTh = (1 + a)*(1+b) + b * xTh
        plt.figure(figsize=(4, 3));
        plt.scatter(np.log10(mu_gene), np.log10(FF_gene), c = [[.8,.8,.8]], alpha = 0.3, s = 3);
        plt.scatter(np.log10(mu_gene)[ix], np.log10(FF_gene)[ix], c = [[0,0,0]], alpha = 0.3,  s= 3);
        plt.plot(np.log10(xTh),np.log10(yTh));
        plt.title(sample_name)
        plt.xlabel('log10(mean)');
        plt.ylabel('log10(Fano factor)');
        plt.show()

    return gene_ix[ix]

def remove_corr_genes(E, gene_list, exclude_corr_genes_list, test_gene_idx, min_corr = 0.1):
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
    numpy array of gene indices (subset of test_gene_idx) that are not correlated with any of the gene signatures
    """

    seed_ix_list = []
    for l in exclude_corr_genes_list:
        seed_ix_list.append(np.array([i for i in range(len(gene_list)) if gene_list[i] in l], dtype=int))

    exclude_ix = []
    for iSet in range(len(seed_ix_list)):
        seed_ix = seed_ix_list[iSet][E[:,seed_ix_list[iSet]].sum(axis=0).A.squeeze() > 0]
        if type(seed_ix) is int:
            seed_ix = np.array([seed_ix], dtype=int)
        elif type(seed_ix[0]) is not int:
            seed_ix = seed_ix[0]
        indat = E[:, seed_ix]
        tmp = sparse_zscore(indat)
        tmp = tmp.sum(1).A.squeeze()

        c = np.zeros(len(test_gene_idx))
        for iG in range(len(c)):
            c[iG],_ = scipy.stats.pearsonr(tmp, E[:,test_gene_idx[iG]].A.squeeze())

        exclude_ix.extend([test_gene_idx[i] for i in range(len(test_gene_idx)) if (c[i]) >= min_corr])
    exclude_ix = np.array(exclude_ix)

    return np.array([g for g in test_gene_idx if g not in exclude_ix], dtype=int)




#################################################################

# check if a given id is in the list L2 (day 24 or 46), or L4 (day26)
# a conversion algorithm 
def converting_id_from_fullSpace_to_subSpace(query_id_array_fullSpace,subSpace_id_array_inFull):
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

    id_sub=np.array(subSpace_id_array_inFull);
    query_id_inSub=[]
    query_success=np.zeros(len(query_id_array_fullSpace),dtype=bool)
    # check one by one
    for j,id_full in enumerate(query_id_array_fullSpace):
        temp=np.nonzero(id_sub==id_full)[0]
        if len(temp)>0:
            query_success[j]=True
            query_id_inSub.append(temp[0])
            
    return np.array(query_id_inSub), query_success
        


def converting_id_from_subSpace_to_fullSpace(query_id_array_subSpace,subSpace_id_array_inFull):
    """
    Convert indices in the subspace to those in the full space.

    Parameters
    ----------
    query_id_array_subSpace: `np.array` or `list`
        Indices in the sub space
    subSpace_id_array_inFull: `np.array` or `list`
        Indices of a targeted sub population in the full space
    
    Returns
    -------
    A converted np.array of indices in the sfull space
    """

    return np.array(subSpace_id_array_inFull)[query_id_array_subSpace]




def compute_state_potential(transition_map,state_annote,fate_array,
    fate_count=False,map_backwards=True):
    """
    Compute state probability towards/from given clusters

    Before any calculation, we row-normalize the transition map.  
    If map_backwards=True, compute the fate map towards given 
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
    map_backwards: `bool`, optional (default: True)
        If `map_backwards=True`, compute for initial cell states (rows of Tmap, at t1);
        else, for later cell states (columns of Tmap, at t2)
    
    Returns
    -------
    fate_map: `np.array`, shape (n_cells, n_fates)
        A matrix of fate potential for each state
    fate_entropy: `np.array`, shape (n_fates,)
        A vector of fate entropy for each state
    """
    
    if not ssp.issparse(transition_map): transition_map=ssp.csr_matrix(transition_map).copy()
    resol=10**(-10)
    transition_map=sparse_rowwise_multiply(transition_map,1/(resol+np.sum(transition_map,1).A.flatten()))
    fate_N=len(fate_array)
    N1,N2=transition_map.shape

    if map_backwards:
        idx_array=np.zeros((N2,fate_N),dtype=bool)
        for k in range(fate_N):
            idx_array[:,k]=(state_annote==fate_array[k])

        fate_map=np.zeros((N1,fate_N))
        fate_entropy=np.zeros(N1)

        for k in range(fate_N):
            fate_map[:,k]=np.sum(transition_map[:,idx_array[:,k]],1).A.flatten()

        for j in range(N1):
                ### compute the "fate-entropy" for each state
            if fate_count:
                p0=fate_map[j,:]
                fate_entropy[j]=np.sum(p0>0)
            else:
                p0=fate_map[j,:]
                p0=p0/(resol+np.sum(p0))+resol
                for k in range(fate_N):
                    fate_entropy[j]=fate_entropy[j]-p0[k]*np.log(p0[k])

    ### forward map
    else:
        idx_array=np.zeros((N1,fate_N),dtype=bool)
        for k in range(fate_N):
            idx_array[:,k]=(state_annote==fate_array[k])

        fate_map=np.zeros((N2,fate_N))
        fate_entropy=np.zeros(N2)

        for k in range(fate_N):
            fate_map[:,k]=np.sum(transition_map[idx_array[:,k],:],0).A.flatten()


        for j in range(N1):
                
                ### compute the "fate-entropy" for each state
            if fate_count:
                p0=fate_map[j,:]
                fate_entropy[j]=np.sum(p0>0)
            else:
                p0=fate_map[j,:]
                p0=p0/(resol+np.sum(p0))+resol
                for k in range(fate_N):
                    fate_entropy[j]=fate_entropy[j]-p0[k]*np.log(p0[k])

    return fate_map, fate_entropy



def compute_fate_probability_map(adata,fate_array=[],used_map_name='transition_map',map_backwards=True):
    """
    Compute fate map from the adata object

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    fate_array: `list`, optional (default: all)
        List of targeted clusters, consistent with adata.obs['state_info'].
        If set to be [], use all fate clusters in adata.obs['state_info'].
    used_map_name: `str`
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map','weinreb_transition_map','naive_transition_map',
        'OT_transition_map','HighVar_transition_map'}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    map_backwards: `bool`, optional (default: True)
        If `map_backwards=True`, compute for initial cell states (rows of Tmap, at t1);
        else, compute for later cell states (columns of Tmap, at t2)

    Returns
    -------
    Update `fate_array`, `fate_map`, `fate_entropy` as a dictionary in adata.uns['fate_map']. 
    """
    
    #transition_map=adata.uns['transition_map']
    #demultiplexed_map=adata.uns['demultiplexed_map']
    state_annote_0=adata.obs['state_info']
    if map_backwards:
        cell_id_t1=adata.uns['Tmap_cell_id_t1']
        cell_id_t2=adata.uns['Tmap_cell_id_t2']

    else:
        cell_id_t2=adata.uns['Tmap_cell_id_t1']
        cell_id_t1=adata.uns['Tmap_cell_id_t2']

    x_emb=adata.obsm['X_umap'][:,0]
    y_emb=adata.obsm['X_umap'][:,1]
    data_des=adata.uns['data_des'][0]
    
    if len(fate_array)==0: fate_array=list(set(state_annote_0))
    

    state_annote_BW=state_annote_0[cell_id_t2]
    
    if used_map_name in adata.uns.keys():
        used_map=adata.uns[used_map_name]

        potential_vector, fate_entropy=compute_state_potential(used_map,state_annote_BW,fate_array,fate_count=True,map_backwards=map_backwards)

        adata.uns['fate_map']={'fate_array':fate_array,'fate_map':potential_vector,'fate_entropy':fate_entropy}

    else:
        logg.error(f"used_map_name should be among adata.uns.keys(), with _transition_map as suffix")

        
def compute_fate_map_and_bias(adata,selected_fates=[],used_map_name='transition_map',map_backwards=True):
    """
    Compute fate map and the relative bias compared to expectation.
    
    `selected_fates` could contain a nested list of clusters. If so, we combine each sub list 
    into a mega-fate cluster and combine the fate map correspondingly. 

    The relative bias is obtained by comparing the fate_prob with the 
    expected_prob from targeted cluster size. It ranges from [0,1], 
    with 0.5 being the point that the fate_prob agrees with expected_prob. 
    1 is extremely biased. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_fates: `list`, optional (default: all)
        List of targeted clusters, consistent with adata.obs['state_info'].
        If set to be [], use all fate clusters in adata.obs['state_info'].
    used_map_name: `str`
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map','weinreb_transition_map','naive_transition_map',
        'OT_transition_map','HighVar_transition_map'}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    map_backwards: `bool`, optional (default: True)
        If `map_backwards=True`, compute for initial cell states (rows of Tmap, at t1);
        else, compute for later cell states (columns of Tmap, at t2)

    Returns
    -------
    Store `fate_array`, `fate_map`, `fate_entropy` in adata.uns['fate_map']. 

    fate_map: `np.array`, shape (n_cell, n_fate)
        n_fate is the number of mega cluster, equals len(selected_fates).
    mega_cluster_list: `list`, shape (n_fate)
        The list of names for the mega cluster. This is relevant when 
        `selected_fates` is a list of list.
    relative_bias: `np.array`, shape (n_cell, n_fate)
    expected_prob: `np.array`, shape (n_fate,)
    valid_fate_list: `list`, shape (n_fate)
        It is basically the same as selected_fates, could contain a nested list
        of fate clusters. It screens for valid fates, though. 
    """

    state_annote=adata.obs['state_info']
    valid_state_annot=list(set(np.array(state_annote)))
    if map_backwards:
        cell_id_t2=adata.uns['Tmap_cell_id_t2']
    else:
        cell_id_t2=adata.uns['Tmap_cell_id_t1']

    if len(selected_fates)==0: selected_fates=list(set(state_annote))

    fate_array_flat=[] # a flatten list of cluster names
    valid_fate_list=[] # a list of cluster lists, each cluster list is a macro cluster
    mega_cluster_list=[] # a list of string description for the macro cluster
    for xx in selected_fates:
        if type(xx) is list:
            valid_fate_list.append(xx)
            des_temp=''
            for zz in xx:
                if zz in valid_state_annot:
                    fate_array_flat.append(zz)
                    des_temp=des_temp+str(zz)+'_'
                else:
                    logg.error(f'{zz} is not a valid cluster name. Please select from: {valid_state_annot}')
            mega_cluster_list.append(des_temp)
        else:
            if xx in valid_state_annot:
                valid_fate_list.append([xx])

                fate_array_flat.append(xx)
                mega_cluster_list.append(str(xx))
            else:
                logg.error(f'{xx} is not a valid cluster name. Please select from: {valid_state_annot}')
                mega_cluster_list.append('')

    compute_fate_probability_map(adata,fate_array=fate_array_flat,used_map_name=used_map_name,map_backwards=map_backwards)
    fate_map_0=adata.uns['fate_map']['fate_map']

    N_macro=len(valid_fate_list)
    fate_map=np.zeros((fate_map_0.shape[0],N_macro))
    relative_bias=np.zeros((fate_map_0.shape[0],N_macro))
    expected_prob=np.zeros(N_macro)
    for jj in range(N_macro):
        idx=np.in1d(fate_array_flat,valid_fate_list[jj])
        fate_map[:,jj]=fate_map_0[:,idx].sum(1)

        for yy in valid_fate_list[jj]:
            expected_prob[jj]=expected_prob[jj]+np.sum(state_annote[cell_id_t2]==yy)/len(cell_id_t2)

        # transformation
        temp_idx=fate_map[:,jj]<expected_prob[jj]
        temp_diff=fate_map[:,jj]-expected_prob[jj]
        relative_bias[temp_idx,jj]=temp_diff[temp_idx]/expected_prob[jj]
        relative_bias[~temp_idx,jj]=temp_diff[~temp_idx]/(1-expected_prob[jj])

        relative_bias[:,jj]=(relative_bias[:,jj]+1)/2 # rescale to the range [0,1]

    return fate_map,mega_cluster_list,relative_bias,expected_prob,valid_fate_list
    

    

def mapout_trajectories(transition_map,state_prob_t2,threshold=0.1,cell_id_t1=[],cell_id_t2=[]):
    """
    map out the ancestor probability for a given later state distribution.

    We assume that transition_map is a normalized probablistic map from 
    t1-state to t2-states. Given a distribution of states at t2, we find 
    and return the initial state distribution. 

    Although it is designed to map trajectories backwards, one can simply 
    tanspose the Tmap, and swap everything related to t1 and t2, to map forward. 

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

    if ssp.issparse(transition_map): transition_map=transition_map.A

    N1,N2=transition_map.shape
    if len(cell_id_t1)==0 and N1==N2: # cell_id_t1 and cell_id_t2 live in the same state space
        state_prob_t1=transition_map.dot(state_prob_t2)
        state_prob_t1_idx=state_prob_t1>threshold*np.max(state_prob_t1)
        state_prob_t1_id=np.nonzero(state_prob_t1_idx)[0]

        state_prob_t1_truc=np.zeros(len(state_prob_t1))
        state_prob_t1_truc[state_prob_t1_id]=state_prob_t1[state_prob_t1_id]
    else:
        # both cell_id_t1 and cell_id_t2 are id's in the full space
        # selected_cell_id is also in the full space
        cell_id_t1=np.array(cell_id_t1)
        cell_id_t2=np.array(cell_id_t2)
        state_prob_t2_subspace=state_prob_t2[cell_id_t2]

        state_prob_t1=transition_map.dot(state_prob_t2_subspace)
        state_prob_t1_idx=state_prob_t1>threshold*np.max(state_prob_t1)
        state_prob_t1_id=np.nonzero(state_prob_t1_idx)[0] # id in t1 subspace
        #state_prob_t1_truc=state_prob_t1[state_prob_t1_id]
        state_prob_t1_truc=np.zeros(len(state_prob_t1))
        state_prob_t1_truc[state_prob_t1_id]=state_prob_t1[state_prob_t1_id]

    return state_prob_t1_truc



# v1, the new methods, more options.
def compute_shortest_path_distance(adata,num_neighbors_target=5,mode='distances',limit=np.inf, method='umap',normalize=True):
    """
    Compute shortest path distance from raw data.

    The distance matrix has two mode: 'connectivity' or 'distance'.
    We found that the 'connectivity' version is sensitive to local cell 
    density heterogeneity, and the 'distance' version is more robust.
    This discrepancy might be due to that the KNN graph construction does not
    direclty take into account of local density heterogeneity. 

    The default is the UMAP method, which takes into account of local 
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

    Returns
    -------
    The normaized distance matrix is returned.
    """
    
    if mode!='connectivities':
        mode='distances'

    logg.hint(f"Chosen mode is {mode}")
    if method=='umap':
        sc.pp.neighbors(adata, n_neighbors=num_neighbors_target,method='umap')
        adj_matrix=adata.obsp[mode].A.copy()

    elif method=='gauss':
        sc.pp.neighbors(adata, n_neighbors=num_neighbors_target,method='gauss')
        adj_matrix=adata.obsp[mode].A.copy()            

    else:
        if mode=='distances': 
            mode='distance'
        else:
            mode='connectivity'
        data_matrix=adata.obsm['X_pca']
        adj_matrix = kneighbors_graph(data_matrix, num_neighbors_target, mode=mode, include_self=True)


    ShortPath_dis = dijkstra(csgraph = ssp.csr_matrix(adj_matrix), directed = False,return_predecessors = False)
    ShortPath_dis_max = np.nanmax(ShortPath_dis[ShortPath_dis != np.inf])
    ShortPath_dis[ShortPath_dis > ShortPath_dis_max] = ShortPath_dis_max #set threshold for shortest paths

    # Set normalized cost matrices based on shortest paths matrices at target and source spaces
    if normalize:
        ShortPath_dis_final=ShortPath_dis / ShortPath_dis.max()
    else:
        ShortPath_dis_final=ShortPath_dis

    return  ShortPath_dis_final


# v0, the new methods, more options.
def compute_shortest_path_distance_from_raw_matrix(data_matrix,num_neighbors_target=5,mode='distance',limit=np.inf):
    """
    Compute shortest path distance from raw data.

    The distance matrix has two mode: 'connectivity' or 'distance'.
    We found that the 'connectivity' version is sensitive to local cell 
    density heterogeneity, and the 'distance' version is more robust.
    This discrepancy might be due to that the KNN graph construction does not
    direclty take into account of local density heterogeneity. 

    Parameters
    ----------
    data_matrix: `np.array`
    num_neighbors_target: `int`, optional (default: 5)
        Used to construct the KNN graph.
    mode: `str`, optional (default: 'distance')
        Options: {'distance','connectivity')
    limit: `float`, optional (default: np.inf)
        If the distance is about this, stop computation, and set 
        the distance beyong this limist by `limit`. This can speed up computation.

    Returns
    -------
    The normaized distance matrix is returned.
    """

    adj_matrix = kneighbors_graph(data_matrix, num_neighbors_target, mode=mode, include_self=True)
    ShortPath_dis = dijkstra(csgraph = ssp.csr_matrix(adj_matrix), directed = False,return_predecessors = False,limit=limit)
    ShortPath_dis_max = np.nanmax(ShortPath_dis[ShortPath_dis != np.inf])
    ShortPath_dis[ShortPath_dis > ShortPath_dis_max] = ShortPath_dis_max #set threshold for shortest paths

    # Set normalized cost matrices based on shortest paths matrices at target and source spaces
    return ShortPath_dis / ShortPath_dis.max()


def add_neighboring_cells_to_a_map(initial_idx,adata,neighbor_N=5):
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

    initial_idx=initial_idx>0
    #print(f"Initial: {np.sum(initial_idx)}")
#     if (np.sum(initial_idx)<size_thresh) & (np.sum(initial_idx)>0):
#         #n0=np.round(size_thresh/np.sum(initial_idx))
#         #sc.pp.neighbors(adata, n_neighbors=int(n0)) #,method='gauss')
#         output_idx=adata.uns['neighbors']['connectivities'][initial_idx].sum(0).A.flatten()>0
#         initial_idx=initial_idx | output_idx

    sc.pp.neighbors(adata, n_neighbors=neighbor_N) #,method='gauss')
    output_idx=adata.uns['neighbors']['connectivities'][initial_idx].sum(0).A.flatten()>0
    post_idx=initial_idx | output_idx
    #print(f"Final: {np.sum(post_idx)}")

    return post_idx


# This one is not necessary in the OT package. 
def compute_symmetric_Wasserstein_distance(sp_id_target,sp_id_ref,full_cost_matrix,
    target_value=[], ref_value=[],OT_epsilon=0.05,OT_stopThr=10**(-8),OT_max_iter=1000):
    """
    Compute symmetric Wasserstein distance between two distributions.

    Parameters
    ----------
    sp_id_target: `np.array`, (n_1_sp,)
        List of cell id's among the targeted population. 
    sp_id_ref: `np.array`, (n_2_sp,)
        List of cell id's for the reference sub population.
    full_cost_matrix: `np.array`, shape (n_1, n_2)
        A cost matrix to map all 'target' to all 'ref'. This is a full matrix.
    target_value: `np.array`, (n_1_sp,)
        The probability for each selected target state.
    ref_value: `np.array`, (n_2_sp,)
        The probability for each selected reference state.
    OT_epsilon: `float`, optional (default: 0.05)
        Entropic regularization parameter to compute the optional 
        transport map from target to ref. 
    OT_stopThr: `float`, optional (default: 10**(-8))
        The stop thresholding for computing the transport map. 
    OT_max_iter: `float`, optional (default: 1000)
        The maximum number of iteration for computing the transport map. 

    Return 
    A vector for [forward_distance, backward_distance, the average]
    """


    import ot.bregman as otb
    # normalized distribution
    if len(target_value)==0:
        target_value=np.ones(len(sp_id_target))
    if len(ref_value)==0:
        ref_value=np.ones(len(sp_id_ref))
    
    input_mu=target_value/np.sum(target_value);
    input_nu=ref_value/np.sum(ref_value);

    full_cost_matrix_1=full_cost_matrix[sp_id_target][:,sp_id_ref]
    OT_transition_map_1=otb.sinkhorn_stabilized(input_mu,input_nu,full_cost_matrix_1,OT_epsilon,numItermax=OT_max_iter,stopThr=OT_stopThr)

    # ot_config = {'C':full_cost_matrix_1,'G':target_value, 'epsilon': OT_epsilon, 'lambda1': 1, 'lambda2': 50,
    #               'epsilon0': 1, 'tau': 10000, 'tolerance': 1e-08,
    #               'max_iter': 1e7, 'batch_size': 5}
    # OT_transition_map_1=optimal_transport_duality_gap(**ot_config)

    full_cost_matrix_2=full_cost_matrix[sp_id_ref][:,sp_id_target]
    OT_transition_map_2=otb.sinkhorn_stabilized(input_nu,input_mu,full_cost_matrix_2,OT_epsilon,numItermax=OT_max_iter,stopThr=OT_stopThr)

    for_Wass_dis=np.sum(OT_transition_map_1*full_cost_matrix_1)
    back_Wass_dis=np.sum(OT_transition_map_2*full_cost_matrix_2)
    return [for_Wass_dis, back_Wass_dis, (for_Wass_dis+back_Wass_dis)/2]


def get_hierch_order(hm, dist_metric='euclidean', linkage_method='ward'):
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
        cache[n+k] = c1 + c2
    o = np.array(cache[2*len(Z)])
    return o


def get_normalized_covariance(data,method='Weinreb'):
    """
    Compute the normalized correlation of the data matrix.

    This is used to compute the fate coupling. 
    Two methods are provided, `Weinreb` method and 'SW'.
    The `Weinreb` method perform normalization against the mean observation
    for each fate; while the `SW` method normalizes against the square root of 
    the self coupling, brining the self coupling to 1 after normalization.

    Parameters
    ----------
    data: `np.array`, shape (n_obs, n_fates)
        A observation matrix for the fate distribution. The observable
        could be the number of barcodes in each fate, or the probability
        of a cell to enter a fate. 
    method: `str`, optional (default: 'Weinreb')
        Method for computing the normalized covariance. Choice: {'Weinreb','SW'}

    Returns
    -------
    Normalized covariance matrix.
    """

    if method=='Weinreb':
        cc = np.cov(data.T)
        mm = np.mean(data,axis=0) + .0001
        X,Y = np.meshgrid(mm,mm)
        cc = cc / X / Y
        return cc#/np.max(cc)
    else:
        resol=10**(-10)
        
        # No normalization performs better.  Not all cell states contribute equally to lineage coupling
        # Some cell states are in the progenitor regime, most ambiguous. They have a larger probability to remain in the progenitor regime, rather than differentiate.
        # Normalization would force these cells to make early choices, which could add noise to the result. 
        # data=core.sparse_rowwise_multiply(data,1/(resol+np.sum(data,1)))
        
        X=data.T.dot(data)
        diag_temp=np.sqrt(np.diag(X))
        for j in range(len(diag_temp)):
            for k in range(len(diag_temp)):
                X[j,k]=X[j,k]/(diag_temp[j]*diag_temp[k])
        return X#/np.max(X)
    


def above_the_line(x_array,x1,x2):
    """
    Return states above a specified line defined by (x1, x2).

    We assume that a state has only two coordiates. 

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
    return (x_array[:,1]-x1[1])>((x2[1]-x1[1])/(x2[0]-x1[0]))*(x_array[:,0]-x1[0])


def save_map(adata):
    """
    Save the adata and print file name prefix. 

    The file name prefix `data_des` will be printed, and 
    the saved file can be accessed again using this prefix.
    """

    data_des=adata.uns['data_des'][0]
    #data_path=adata.uns['data_path'][0]
    data_path=settings.data_path


    # need to remove these, otherwise, it does not work
    for xx in  ['fate_trajectory', 'multiTime_cell_id_t1', 'multiTime_cell_id_t2', 'fate_map']:
        if xx in adata.uns.keys():
            adata.uns.pop(xx)

    file_name=f'{data_path}/{data_des}_adata_with_transition_map.h5ad'
    adata.write_h5ad(file_name, compression='gzip')
    print(f"Saved file: data_des='{data_des}'")

def save_preprocessed_adata(adata,data_des=''):
    if len(data_des)==0:
        data_des=adata.uns['data_des'][0]
    data_path=settings.data_path

    for xx in  ['fate_trajectory', 'multiTime_cell_id_t1', 'multiTime_cell_id_t2', 'fate_map']:
        if xx in adata.uns.keys():
            adata.uns.pop(xx)

    adata.write_h5ad(f'{data_path}/{data_des}_adata_preprocessed.h5ad', compression='gzip')
    print(f"Saved file: data_des='{data_des}'")

def load_saved_adata(data_des):
    """
    Load pre-saved adata based on the prefix 'data_des'
    """

    data_path=settings.data_path
    #print(f"Load data: data_des='{data_des}'")
    adata=sc.read(f'{data_path}/{data_des}_adata_with_transition_map.h5ad')
    return adata

def check_available_map(adata):
    """
    Check available transition map. 

    Update adata.uns['available_map'].
    """

    available_map=[]
    for xx in adata.uns.keys():
        if 'transition_map' in xx:
            available_map.append(xx)
    adata.uns['available_map']=available_map

def switch_adata_representation(adata,to_new=True):
    if to_new:
        adata.obsm['X_clone']=adata.obsm['cell_by_clone_matrix']
        adata.obs['state_info']=adata.obs['state_annotation']
        #adata.uns['data_des']=['paper_OneTimeClone_t*pos_17*pos_21*D27']
        adata.obsm.pop('cell_by_clone_matrix')
        adata.obs.pop('state_annotation')
    else:
        adata.obsm['cell_by_clone_matrix']=adata.obsm['X_clone']
        adata.obs['state_annotation']=adata.obs['state_info']
        #adata.uns['data_des']=['paper_OneTimeClone_t*pos_17*pos_21*D27']
        adata.obsm.pop('X_clone')
        adata.obs.pop('state_info')

def check_available_choices(adata):
    """
    Check available parameter choices.

    Also update adata.uns['available_map'] and adata.uns['clonal_time_points'].
    """

    check_available_map(adata)
    available_map=adata.uns['available_map']

    X_clone=adata.obsm['X_clone']
    time_info=adata.obs['time_info']

    # record time points with clonal information
    if ssp.issparse(X_clone):
        clone_N_per_cell=X_clone.sum(1).A.flatten()
    else:
        clone_N_per_cell=X_clone.sum(1)

    clonal_time_points=[]
    for xx in list(set(time_info)):
        idx=np.array(time_info)==xx
        if np.sum(clone_N_per_cell[idx])>0:
            clonal_time_points.append(xx)
    adata.uns['clonal_time_points']=clonal_time_points

    print("Available transition maps:",available_map)
    print("Availabel clusters:", list(set(adata.obs['state_info'])))
    print("Availabel time points:", list(set(adata.obs['time_info'])))
    print("Clonal time points:",clonal_time_points)

def compute_pca(m1, m2, n_components):
    matrices = list()
    matrices.append(m1 if not scipy.sparse.isspmatrix(m1) else m1.toarray())
    matrices.append(m2 if not scipy.sparse.isspmatrix(m2) else m2.toarray())
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
    pca_2 = comp[m1_len:(m1_len + m2_len)]
    return pca_1, pca_2, pca, mean_shift

def compute_default_cost_matrix(a, b, eigenvals=None):

    if eigenvals is not None:
        a = a.dot(eigenvals)
        b = b.dot(eigenvals)

    cost_matrix = pairwise.pairwise_distances(a.toarray() if scipy.sparse.isspmatrix(a) else a,
                                                              b.toarray() if scipy.sparse.isspmatrix(b) else b,
                                                              metric='sqeuclidean', n_jobs=-1)
    cost_matrix = cost_matrix / np.median(cost_matrix)
    return cost_matrix

def compute_gene_exp_distance(adata,p0_indices,p1_indices,pc_n=30):
    '''
    Compute the gene expression distance between t0 and t1 states.

    p0_indices could be either a boolean array or index array
    '''
    p0=adata[p0_indices,:]
    p1=adata[p1_indices,:]
    p0_x, p1_x, pca, mean = compute_pca(p0.X, p1.X, pc_n)
    eigenvals = np.diag(pca.singular_values_)
    #gene_exp_dis_t0 = compute_default_cost_matrix(p0_x, p0_x, eigenvals)
    #gene_exp_dis_t1 = compute_default_cost_matrix(p1_x, p1_x, eigenvals)
    gene_exp_dis_t0t1 = compute_default_cost_matrix(p0_x, p1_x, eigenvals)
    return gene_exp_dis_t0t1

