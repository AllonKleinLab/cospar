import numpy as np
import scipy
import os
import scipy.stats
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise
import scipy.sparse as ssp
import scanpy as sc
import pandas as pd
from scanpy import read
import statsmodels.sandbox.stats.multicomp
from scipy.spatial.distance import pdist
from fastcluster import linkage
from .. import settings
from .. import pl
from .. import tmap
from .. import logging as logg
import time
from matplotlib import pyplot as plt

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
        'Qvalue': pv[sort_idx],
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

# this is not working well
def sparse_zscore(E, gene_mean=None, gene_stdev=None):
    """ z-score normalize each column of a sparse matrix """
    if gene_mean is None:
        gene_mean = E.mean(0)
    if gene_stdev is None:
        gene_stdev = np.sqrt(sparse_var(E))
    return sparse_rowwise_multiply((E - gene_mean).T, 1/gene_stdev).T


def corr2_coeff(A,B):
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

    resol=10**(-15)
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);

    # Finally get corr coeff
    return np.dot(A_mA,B_mB.T)/(np.sqrt(np.dot(ssA[:,None],ssB[None]))+resol)


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
    if nrow!=a.shape[0]:
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
    if ncol!=a.shape[0]:
        logg.error("Dimension mismatch, multiplication failed")
        return E
    else:
        w = ssp.lil_matrix((ncol, ncol))
        w.setdiag(a)
        return (ssp.csr_matrix(E)*w)


# This is faster than v0
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

    #print("V1")
    #t1=time.time()
    if ssp.issparse(input_matrix): 
        input_matrix=input_matrix.A
        #print("Turn the sparse matrix into numpy array")
        #print(f"Time-1: {time.time()-t1}")
    
    max_vector=np.max(input_matrix,int(row_threshold))    
    for j in range(len(max_vector)):
        if row_threshold:
            idx=input_matrix[j,:]<threshold*max_vector[j]
            input_matrix[j,idx]=0
        else:
            idx=input_matrix[:,j]<threshold*max_vector[j]
            input_matrix[idx,j]=0
    #print(f"matrix_row_or_column_thresholding time:{time.time()-t1}")
    return input_matrix



# This is slower due to a step of copying
def matrix_row_or_column_thresholding_v0(input_matrix,threshold=0.1,row_threshold=True):
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

    Vscores, CV_eff, CV_input, gene_ix, mu_gene, FF_gene, a, b = get_vscores(E[base_ix, :])
    ix2 = Vscores>0
    Vscores = Vscores[ix2]
    gene_ix = gene_ix[ix2]
    mu_gene = mu_gene[ix2]
    FF_gene = FF_gene[ix2]
    min_vscore = np.percentile(Vscores, min_vscore_pctl)
    ix = (((E[:,gene_ix] >= min_counts).sum(0).A.squeeze() >= min_cells) & (Vscores >= min_vscore))
    

    x_min = 0.5*np.min(mu_gene)
    x_max = 2*np.max(mu_gene)
    Mean_value = x_min * np.exp(np.log(x_max/x_min)*np.linspace(0,1,100))
    FanoFactor = (1 + a)*(1+b) + b * Mean_value


    if show_vscore_plot:
        plt.figure(figsize=(4, 3));
        plt.scatter(np.log10(mu_gene), np.log10(FF_gene), c = [[.8,.8,.8]], alpha = 0.3, s = 3);
        plt.scatter(np.log10(mu_gene)[ix], np.log10(FF_gene)[ix], c = [[0,0,0]], alpha = 0.3,  s= 3);
        plt.plot(np.log10(Mean_value),np.log10(FanoFactor));
        plt.title(sample_name)
        plt.xlabel('log10(mean)');
        plt.ylabel('log10(Fano factor)');
        plt.show()

    if FanoFactor[-1]<FanoFactor[0]:
        logg.error(f'The estimated Fano factor is NOT in expected form, which would affect the results.\n'
            'Please make sure that the count matrix adata.X is NOT log-transformed.')
        return None
    else:
        return gene_ix[ix]

# We found that this does not work
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
    Numpy array of gene indices (a subset of test_gene_idx) that are not correlated with any of the gene signatures
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


## based on mean
def compute_state_potential(transition_map,state_annote,fate_array,
    fate_count=False,map_backward=True,method='sum'):
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
    
    if not ssp.issparse(transition_map): transition_map=ssp.csr_matrix(transition_map).copy()
    resol=10**(-10)
    transition_map=sparse_rowwise_multiply(transition_map,1/(resol+np.sum(transition_map,1).A.flatten()))
    fate_N=len(fate_array)
    N1,N2=transition_map.shape

    #logg.info(f"Use the method={method} to compute differentiation bias")

    if map_backward:
        idx_array=np.zeros((N2,fate_N),dtype=bool)
        for k in range(fate_N):
            idx_array[:,k]=(state_annote==fate_array[k])

        fate_map=np.zeros((N1,fate_N))
        fate_entropy=np.zeros(N1)

        for k in range(fate_N):
            if method=='max':
                fate_map[:,k]=np.max(transition_map[:,idx_array[:,k]],1).A.flatten()
            elif method=='mean':
                fate_map[:,k]=np.mean(transition_map[:,idx_array[:,k]],1).A.flatten()
            else: # just perform summation
                fate_map[:,k]=np.sum(transition_map[:,idx_array[:,k]],1).A.flatten()


        # rescale. After this, the fate map value spreads between [0,1]. Otherwise, they can be tiny.
        if (method!='sum') and (method!='norm-sum'):
            fate_map=fate_map/np.max(fate_map)
        elif (method=='norm-sum'):
            # perform normalization of the fate map. This works only if there are more than two fates
            if fate_N>1:
                #logg.info('conditional method: perform column normalization')
                fate_map=sparse_column_multiply(fate_map,1/(resol+np.sum(fate_map,0).flatten())).A
                fate_map=fate_map/np.max(fate_map)


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
            if method=='max':
                fate_map[:,k]=np.max(transition_map[idx_array[:,k],:],0).A.flatten()
            elif method=='mean':
                fate_map[:,k]=np.mean(transition_map[idx_array[:,k],:],0).A.flatten()
            else:
                fate_map[:,k]=np.sum(transition_map[idx_array[:,k],:],0).A.flatten()

        # rescale. After this, the fate map value spreads between [0,1]. Otherwise, they can be tiny.
        if (method!='sum') and (method!='norm-sum'):
            fate_map=fate_map/np.max(fate_map)
        elif (method=='norm-sum'):
            # perform normalization of the fate map. This works only if there are more than two fates
            if fate_N>1:
                #logg.info('conditional method: perform column normalization')
                fate_map=sparse_column_multiply(fate_map,1/(resol+np.sum(fate_map,0).flatten())).A

         
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




        

def analyze_selected_fates(selected_fates,state_info):
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

    state_info=np.array(state_info)
    valid_state_annot=list(set(state_info))
    if selected_fates is None:
        selected_fates=valid_state_annot

    fate_array_flat=[] # a flatten list of cluster names
    valid_fate_list=[] # a list of cluster lists, each cluster list is a macro cluster
    mega_cluster_list=[] # a list of string description for the macro cluster
    sel_index_list=[]
    for xx in selected_fates:
        if type(xx) is list:
            valid_fate_list.append(xx)
            des_temp=''
            temp_idx=np.zeros(len(state_info),dtype=bool)
            for zz in xx:
                if zz in valid_state_annot:
                    fate_array_flat.append(zz)
                    if des_temp!='':
                        des_temp=des_temp+'_'

                    des_temp=des_temp+str(zz)
                    temp_idx=temp_idx | (state_info==zz)
                else:
                    logg.error(f'{zz} is not a valid cluster name. Please select from: {valid_state_annot}')
            mega_cluster_list.append(des_temp)
            sel_index_list.append(temp_idx)
        else:
            if xx in valid_state_annot:
                valid_fate_list.append([xx])

                fate_array_flat.append(xx)
                mega_cluster_list.append(str(xx))
            else:
                logg.error(f'{xx} is not a valid cluster name. Please select from: {valid_state_annot}')
                mega_cluster_list.append('')

            temp_idx=(state_info==xx)
            sel_index_list.append(temp_idx)

    # exclude invalid clusters
    mega_cluster_list=np.array(mega_cluster_list)
    fate_array_flat=np.array(fate_array_flat)
    sel_index_list=np.array(sel_index_list)
    valid_idx=mega_cluster_list!=''

    return mega_cluster_list[valid_idx],valid_fate_list,fate_array_flat,sel_index_list[valid_idx]


def compute_fate_probability_map(adata,selected_fates=None,used_Tmap='transition_map',map_backward=True,method='norm-sum',fate_count=True):
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

    if method not in ['max','sum','mean','norm-sum']:
        logg.warn("method not in {'max','sum','mean','norm-sum'}; use the 'norm-sum' method")
        method='norm-sum'


    if map_backward:
        cell_id_t2=adata.uns['Tmap_cell_id_t2']
    else:
        cell_id_t2=adata.uns['Tmap_cell_id_t1']

    state_annote=adata.obs['state_info']
    if selected_fates is None: selected_fates=list(set(state_annote))
    mega_cluster_list,valid_fate_list,fate_array_flat,sel_index_list=analyze_selected_fates(selected_fates,state_annote)



    state_annote_0=np.array(adata.obs['state_info'])
    if map_backward:
        cell_id_t1=adata.uns['Tmap_cell_id_t1']
        cell_id_t2=adata.uns['Tmap_cell_id_t2']

    else:
        cell_id_t2=adata.uns['Tmap_cell_id_t1']
        cell_id_t1=adata.uns['Tmap_cell_id_t2']

    x_emb=adata.obsm['X_emb'][:,0]
    y_emb=adata.obsm['X_emb'][:,1]
    data_des=adata.uns['data_des'][-1]

    state_annote_1=state_annote_0.copy()
    for j1,new_cluster_id in enumerate(mega_cluster_list):
        idx=np.in1d(state_annote_0,valid_fate_list[j1])
        state_annote_1[idx]=new_cluster_id

    state_annote_BW=state_annote_1[cell_id_t2]
    
    if used_Tmap in adata.uns.keys():
        used_map=adata.uns[used_Tmap]

        fate_map, fate_entropy=compute_state_potential(used_map,state_annote_BW,mega_cluster_list,fate_count=fate_count,map_backward=map_backward,method=method)

        adata.uns['fate_map']={'fate_array':mega_cluster_list,'fate_map':fate_map,'fate_entropy':fate_entropy}

    else:
        logg.error(f"used_Tmap should be among adata.uns.keys(), with _transition_map as suffix")
    #### finish


    N_macro=len(valid_fate_list)
#    fate_map=np.zeros((fate_map_0.shape[0],N_macro))
    relative_bias=np.zeros((fate_map.shape[0],N_macro))
    expected_prob=np.zeros(N_macro)
    for jj in range(N_macro):
        # idx=np.in1d(fate_array_flat,valid_fate_list[jj])
        # if method=='max':
        #     fate_map[:,jj]=fate_map_0[:,idx].max(1)
        # elif method=='mean':
        #     fate_map[:,jj]=fate_map_0[:,idx].mean(1)
        # else: # use the sum method
        #     fate_map[:,jj]=fate_map_0[:,idx].sum(1)


        for yy in valid_fate_list[jj]:
            expected_prob[jj]=expected_prob[jj]+np.sum(state_annote[cell_id_t2]==yy)/len(cell_id_t2)

        # transformation, this is useful only when the method =='sum'
        temp_idx=fate_map[:,jj]<expected_prob[jj]
        temp_diff=fate_map[:,jj]-expected_prob[jj]
        relative_bias[temp_idx,jj]=temp_diff[temp_idx]/expected_prob[jj]
        relative_bias[~temp_idx,jj]=temp_diff[~temp_idx]/(1-expected_prob[jj])

        relative_bias[:,jj]=(relative_bias[:,jj]+1)/2 # rescale to the range [0,1]

    return fate_map,mega_cluster_list,relative_bias,expected_prob,valid_fate_list,sel_index_list,fate_entropy
    


    

def mapout_trajectories(transition_map,state_prob_t2,threshold=0.1,cell_id_t1=[],cell_id_t2=[]):
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
def compute_shortest_path_distance(adata,num_neighbors_target=5,mode='distances',limit=np.inf, method='umap',normalize=True,use_existing_KNN_graph=False):
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
    
    if (not use_existing_KNN_graph) or ('connectivities' not in adata.obsp.keys()):

        if mode!='connectivities':
            mode='distances'

        logg.hint(f"Chosen mode is {mode}")
        if method=='umap':
            sc.pp.neighbors(adata, n_neighbors=num_neighbors_target,method='umap')
            adj_matrix=adata.obsp[mode]

        elif method=='gauss':
            sc.pp.neighbors(adata, n_neighbors=num_neighbors_target,method='gauss')
            adj_matrix=adata.obsp[mode]           

        else:
            if mode=='distances': 
                mode='distance'
            else:
                mode='connectivity'
            data_matrix=adata.obsm['X_pca']
            adj_matrix = kneighbors_graph(data_matrix, num_neighbors_target, mode=mode, include_self=True)

    else:
        logg.info("Use existing KNN graph at adata.obsp['connectivities'] for generating the smooth matrix")
        adj_matrix=adata.obsp['connectivities'];

    ShortPath_dis = dijkstra(csgraph = ssp.csr_matrix(adj_matrix), directed = False,return_predecessors = False)
    ShortPath_dis_max = np.nanmax(ShortPath_dis[ShortPath_dis != np.inf])
    ShortPath_dis[ShortPath_dis > ShortPath_dis_max] = ShortPath_dis_max #set threshold for shortest paths

    # Set normalized cost matrices based on shortest paths matrices at target and source spaces
    if normalize:
        ShortPath_dis_final=ShortPath_dis / ShortPath_dis.max()
    else:
        ShortPath_dis_final=ShortPath_dis

    return  ShortPath_dis_final


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
    output_idx=adata.obsp['connectivities'][initial_idx].sum(0).A.flatten()>0
    post_idx=initial_idx | output_idx
    #print(f"Final: {np.sum(post_idx)}")

    return post_idx


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

    if method not in ['Weinreb','SW']:
        logg.warn('method not among [Weinreb, SW]; set method=SW')
        method='SW'

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

    return (x_array[:,1]-x1[1])>((x2[1]-x1[1])/(x2[0]-x1[0]))*(x_array[:,0]-x1[0])


def save_map(adata):
    """
    Save the adata and print the filename prefix. 

    The filename prefix `data_des` will be printed, and 
    the saved file can be accessed again using this prefix.
    """

    data_des=adata.uns['data_des'][-1]
    #data_path=adata.uns['data_path'][0]
    data_path=settings.data_path


    # need to remove these, otherwise, it does not work
    for xx in  ['fate_trajectory', 'multiTime_cell_id_t1', 
    'multiTime_cell_id_t2', 'fate_map','fate_bias','fate_potency']:
        if xx in adata.uns.keys():
            adata.uns.pop(xx)

    file_name=f'{data_path}/{data_des}_adata_with_transition_map.h5ad'
    adata.write_h5ad(file_name, compression='gzip')
    print(f"Saved file: {file_name}")



def smooth_a_vector(adata,vector,round_of_smooth=5,use_full_Smatrix=True,trunca_threshold=0.001,compute_new=False,neighbor_N=20,show_groups=True,save_subset=True):
    """
    

    """

    vector=np.array(vector)
    if len(vector) != adata.shape[0]:
        logg.error("The length of the vector does not match that of adata.shape[0]")
        return None

    data_des=adata.uns['data_des'][0]
    data_des_1=adata.uns['data_des'][-1]
    data_path=settings.data_path
    if 'sp_idx' in adata.uns.keys():
        sp_idx=adata.uns['sp_idx']
    else:
        sp_idx=np.ones(adata.shape[0]).astype(bool)

    #trunca_threshold=0.001 # this value is only for reducing the computed matrix size for saving
    temp_str='0'+str(trunca_threshold)[2:]

    if use_full_Smatrix:
        similarity_file_name=f'{data_path}/{data_des}_Similarity_matrix_with_all_cell_states_kNN{neighbor_N}_Truncate{temp_str}'

        if not os.path.exists(similarity_file_name+f'_SM{round_of_smooth}.npz'):
            logg.error(f"Similarity matrix at given parameters have not been computed before! Fiale name: {similarity_file_name}")
            logg.error(f'Please use other Tmap inference function to build the full similarity matrix at the corresponding smooth rounds, using adata_orig.')     
            return  None

    else:
        similarity_file_name=f'{data_path}/{data_des_1}_Similarity_matrix_with_selected_states_kNN{neighbor_N}_Truncate{temp_str}'


    # we cannot force it to compute new at this time. Otherwise, if we use_full_Smatrix, the resulting similarity is actually from adata, thus not full similarity. 

    re_compute=(not use_full_Smatrix) and (compute_new) # re-compute only when not using full similarity 
    similarity_matrix_full=tmap.generate_similarity_matrix(adata,similarity_file_name,round_of_smooth=round_of_smooth,
                neighbor_N=neighbor_N,truncation_threshold=trunca_threshold,save_subset=save_subset,compute_new_Smatrix=re_compute)

    if use_full_Smatrix:
        #pdb.set_trace()
        similarity_matrix_full_sp=similarity_matrix_full[sp_idx][:,sp_idx]

    else:
        similarity_matrix_full_sp=similarity_matrix_full

    smooth_vector=similarity_matrix_full_sp*vector
    if show_groups:
        x_emb=adata.obsm['X_emb'][:,0]
        y_emb=adata.obsm['X_emb'][:,1]
        fig_width=settings.fig_width; fig_height=settings.fig_height;
        fig=plt.figure(figsize=(2*fig_width,fig_height))
        ax=plt.subplot(1,2,1)
        pl.customized_embedding(x_emb,y_emb,vector,ax=ax)
        ax=plt.subplot(1,2,2)
        pl.customized_embedding(x_emb,y_emb,smooth_vector,ax=ax)
        
    return smooth_vector

def update_time_ordering(adata,updated_ordering=None):
    """
    Update the ordering of time points at adata.uns['time_ordering']

    Parameters
    ----------
    updated_ordering: `list`, optional (default: none) 
        A list of distinct time points in ascending order.
        If not provided, sort the time variable directly.
        However, these time variables are string. Their sorting
        may not be correct.   
    """

    if updated_ordering is not None:
        time_info=list(set(adata.obs['time_info']))
        N_match=np.sum(np.in1d(time_info,updated_ordering))
        if (len(updated_ordering)!=N_match) or (len(updated_ordering)!=len(time_info)):
            logg.error("The provided time points are not correct (wrong length, or invalid value)")
            logg.info(f"Please provide an ordering of all time points in ascending order. Available time points are: {time_info}")
        else:
            adata.uns['time_ordering']=np.array(updated_ordering)
    else:
        time_ordering=np.sort(list(set(adata.obs['time_info'])))
        logg.info(f"Current time ordering from simple sorting: {time_ordering}")
        adata.uns['time_ordering']=np.array(time_ordering)

def check_adata_structure(adata):
    """
    Check whether the adata has the right structure. 
    """

    flag=True
    if not ('X_pca' in adata.obsm.keys()):
        logg.error('*X_pca* missing from adata.obsm')
        flag=False

    if not ('X_emb' in adata.obsm.keys()):
        logg.error('*X_emb* missing from adata.obsm')
        flag=False

    if not ('X_clone' in adata.obsm.keys()):
        logg.error('*X_clone* missing from adata.obsm')
        flag=False

    if not ('time_info' in adata.obs.keys()):
        logg.error('*time_info* missing from adata.obs')
        flag=False

    if not ('state_info' in adata.obs.keys()):
        logg.error('*state_info* missing from adata.obs')
        flag=False

    if flag:
        print("The adata structure looks fine!")

def save_preprocessed_adata(adata,data_des=None):
    """
    Save preprocessed adata.

    It will remove unneeded entries, and use the default
    prefix (data_des) to save the results if a new data_des is not provided. 
    """

    if data_des is None:
        data_des=adata.uns['data_des'][0]
        
    data_path=settings.data_path

    for xx in  ['fate_trajectory', 'multiTime_cell_id_t1', 
                'multiTime_cell_id_t2', 'fate_map','fate_bias', 'fate_potency',
                'transition_map','intraclone_transition_map','clonal_transition_map',
               'OT_transition_map','HighVar_transition_map',
               'Tmap_cell_id_t1', 'Tmap_cell_id_t2', 'available_map', 
                'clonal_cell_id_t1', 'clonal_cell_id_t2']:
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

    file_name=f'{data_path}/{data_des}_adata_preprocessed.h5ad'
    adata.write_h5ad(file_name, compression='gzip')
    print(f"Saved file: {file_name}")

def load_saved_adata_with_key(data_des):
    """
    Load pre-saved adata based on the prefix 'data_des'
    """

    data_path=settings.data_path
    #print(f"Load data: data_des='{data_des}'")
    file_name=f'{data_path}/{data_des}_adata_with_transition_map.h5ad'
    if os.path.exists(file_name):
        adata=sc.read(file_name)
        return adata
    else:
        logg.error(f"The file does not existed yet")

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

def selecting_cells_by_time_points(time_info,selected_time_points):
    """
    Check validity of selected time points, and return the selected index.

    If selected_time_points=[], we select all cell states. 
    """

    time_info=np.array(time_info)
    valid_time_points=set(time_info)
    if selected_time_points is not None:
        sp_idx=np.zeros(len(time_info),dtype=bool)
        for xx in selected_time_points:
            if xx not in valid_time_points:
                logg.error(f"{xx} is not a valid time point.")
            sp_id_temp=np.nonzero(time_info==xx)[0]
            sp_idx[sp_id_temp]=True
    else:
        sp_idx=np.ones(len(time_info),dtype=bool)

    return sp_idx


def check_available_clonal_info(adata):

    X_clone=adata.obsm['X_clone']
    time_info=adata.obs['time_info']

    if 'time_ordering' not in adata.uns.keys():
        update_time_ordering(adata)

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

    time_ordering=adata.uns['time_ordering']
    sel_idx_temp=np.in1d(time_ordering,clonal_time_points)
    clonal_time_points=time_ordering[sel_idx_temp]
    adata.uns['clonal_time_points']=clonal_time_points


def check_available_choices(adata):
    """
    Check available parameter choices.

    Also update adata.uns['available_map'] and adata.uns['clonal_time_points'].
    """

    check_available_map(adata)
    available_map=adata.uns['available_map']

    check_available_clonal_info(adata)
    clonal_time_points=adata.uns['clonal_time_points']

    print("Available transition maps:",available_map)
    print("Available clusters:", list(set(adata.obs['state_info'])))
    print("Available time points:", adata.uns['time_ordering'])
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
    """
    Compute the gene expression distance between t0 and t1 states.

    p0_indices could be either a boolean array or index array
    """
    p0=adata[p0_indices,:]
    p1=adata[p1_indices,:]
    p0_x, p1_x, pca, mean = compute_pca(p0.X, p1.X, pc_n)
    eigenvals = np.diag(pca.singular_values_)
    #gene_exp_dis_t0 = compute_default_cost_matrix(p0_x, p0_x, eigenvals)
    #gene_exp_dis_t1 = compute_default_cost_matrix(p1_x, p1_x, eigenvals)
    gene_exp_dis_t0t1 = compute_default_cost_matrix(p0_x, p1_x, eigenvals)
    return gene_exp_dis_t0t1






def get_X_clone_with_reference_ordering(clone_data_cell_id,clone_data_barcode_id,reference_cell_id,reference_clone_id=None):
    '''
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

    Returns
    -------
    X_clone: `np.array`     
        The clonal data matrix, with the row in cell id, and column in barcode id.
    reference_clone_id: `list`
    '''
    
    if reference_clone_id is None:
        reference_clone_id=list(set(clone_data_barcode_id))
        
    reference_clone_id=np.array(reference_clone_id)
    ## generate X_clone where the cell idx have been sorted
    X_clone=np.zeros((len(reference_cell_id),len(reference_clone_id)))
    logg.info(f"Total number of barcode entries: {len(clone_data_cell_id)}")
    for j in range(len(clone_data_cell_id)):
        if j%100000==0:
            logg.info(f"Current barcode entry: {j}")
        cell_id_1=np.nonzero(reference_cell_id==clone_data_cell_id[j])[0]
        clone_id_1=np.nonzero(reference_clone_id==clone_data_barcode_id[j])[0]
        #X_clone[cell_id_1,clone_id_1] += 1
        X_clone[cell_id_1,clone_id_1] = 1
        
    sp_idx=X_clone.sum(0)>0
    return X_clone[:,sp_idx],reference_clone_id[sp_idx]

