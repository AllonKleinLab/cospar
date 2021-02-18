# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd
import time
import scanpy as sc
import scipy.sparse as ssp 
from pathlib import Path, PurePath
from matplotlib import pyplot as plt

from .. import help_functions as hf
from .. import plotting as CSpl
from .. import settings
from .. import logging as logg


def initialize_adata_object(cell_by_gene_matrix,gene_names,time_info,
    X_clone=[],X_pca=[],X_emb=[],state_info=[],data_des='cospar'):
    """
    Initialized the :class:`~anndata.AnnData` object.

    The principal components (`X_pca`), 2-d embedding (`X_emb`), and 
    `state_info` can be provided upfront, or generated in the next step.
    The clonal information (`X_clone`) is also optional.

    Parameters
    ---------- 
    cell_by_gene_matrix: `np.ndarray` or `sp.spmatrix`
        The count matrix for state information. 
        Rows correspond to cells and columns to genes. 
    gene_names: `np.ndarray`
        An array of gene names.
    time_info: `np.ndarray`
        Time annotation for each cell in `str`,like 'Day27' or 'D27'.
        However, it can also contain other sample_info, 
        like 'GFP+_day27', and 'GFP-_day27'.
    X_clone: `sp.spmatrix` or `np.ndarray`, optional (default: None)        
        The clonal data matrix, with the row in cell id, and column in barcode id.
        For evolvable barcoding, a cell may carry several different barcode id. 
        Currently, we assume each entry is 0 or 1. 
    X_pca: `np.ndarray`, optional (default: None)
        A matrix of the shape n_cell*n_pct. Used for computing similarity matrices.
    X_emb: `np.ndarray`, optional (default: None)
        Two-dimensional matrix for embedding.  
    state_info: `np.ndarray`, optional (default: None)
        The classification and annotation for each cell state. 
        This will be used only after the map is created. Can be adjusted later.
    data_des: `str`, optional (default:'cospar')
        This is just a name to label/distinguish this data. 
        Will be used for saving the results. It should be a unique 
        name for a new dataset stored in the same folder to avoid conflicts.
            
    Returns
    -------
    Generate an :class:`~anndata.AnnData` object with the following entries 
    obs: 'time_info', 'state_info'
    uns: 'data_des', 'clonal_time_points'
    obsm: 'X_clone', 'X_pca', 'X_emb'
    """
   

    data_path=settings.data_path
    figure_path=settings.figure_path


    ### making folders
    if not Path(data_path).is_dir():
        logg.info(f'creating directory {data_path}/ for saving data')
        Path(data_path).mkdir(parents=True)

    if not Path(figure_path).is_dir():
        logg.info(f'creating directory {figure_path}/ for saving figures')
        Path(figure_path).mkdir(parents=True)

    
    time_info=np.array(time_info)
    time_info=time_info.astype(str)

    #!mkdir -p $data_path
    adata=sc.AnnData(ssp.csr_matrix(cell_by_gene_matrix))
    adata.var_names=list(gene_names)

    if X_clone.shape[0]==0:
        X_clone=np.zeros((adata.shape[0],1))


    # we do not remove zero-sized clones here as in some case, we want 
    # to use this package even if we do not have clonal data.
    # Removing zero-sized clones will be handled downstream when we have to 
    adata.obsm['X_clone']=ssp.csr_matrix(X_clone)
    adata.obs['time_info']=pd.Categorical(time_info)
    adata.uns['data_des']=[data_des]
    # adata.uns['data_path']=[data_path]
    # adata.uns['figure_path']=[figure_path]

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


    if len(X_pca)==adata.shape[0]:
        adata.obsm['X_pca']=np.array(X_pca)

    if len(state_info)==adata.shape[0]:
        adata.obs['state_info']=pd.Categorical(state_info)

    if len(X_emb)==adata.shape[0]:
        adata.obsm['X_emb']=X_emb

    logg.info(f"All time points: {set(adata.obs['time_info'])}")
    logg.info(f"Time points with clonal info: {set(adata.uns['clonal_time_points'])}")
            


    time_ordering=np.sort(list(set(adata.obs['time_info'])))
    logg.warn(f"Default ascending ordering of time points are: {time_ordering}. If not correct, run cs.hf.update_time_ordering for correction.")
    adata.uns['time_ordering']=np.array(time_ordering)

    return adata


def get_highly_variable_genes(adata,normalized_counts_per_cell=10000,min_counts=3, 
    min_cells=3, min_gene_vscore_pctl=85):
    """
    Get highly variable genes.

    We assume that data preprocessing are already done, like removing low quality cells. 
    It first perform count normalization, then variable gene selection. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    normalized_counts_per_cell: `int`, optional (default: 1000)
        count matrix normalization 
    min_counts: `int`, optional (default: 3)  
        Minimum number of UMIs per cell to be considered for selecting highly variable genes. 
    min_cells: `int`, optional (default: 3)
        Minimum number of cells per gene to be considered for selecting highly variable genes. 
    min_gene_vscore_pctl: `int`, optional (default: 85)
        Gene expression variability threshold, in the unit of percentile,  
        for selecting highly variable genes. Range: [0,100], with a higher 
        number selecting more variable genes. 

    Returns
    -------
    None. Modify adata.var['highly_variable'].
    """

    sc.pp.normalize_per_cell(adata, counts_per_cell_after=normalized_counts_per_cell)

    verbose=logg._settings_verbosity_greater_or_equal_than(2) # the highest level is 3

    logg.info('Finding highly variable genes...')
    gene_list=adata.var_names
    highvar_genes = gene_list[hf.filter_genes(
        adata.X, 
        min_counts=min_counts, 
        min_cells=min_cells, 
        min_vscore_pctl=min_gene_vscore_pctl, 
        show_vscore_plot=verbose)]

    adata.var['highly_variable'] = False
    adata.var.loc[highvar_genes, 'highly_variable'] = True
    logg.info(f'Keeping {len(highvar_genes)} genes')
    


def remove_cell_cycle_correlated_genes(adata,cycling_gene_list=['Ube2c','Hmgb2', 'Hmgn2', 'Tuba1b', 'Ccnb1', 'Tubb5', 'Top2a','Tubb4b'],corr_threshold=0.1,confirm_change=False):
    """
    Remove cell-cycle correlated genes.

    Take pre-selected highly variable genes, and compute their correlation with 
    the set of cell cycle genes. If confirm_change=True, remove those having absolute correlation 
    score are above given correlation threshold. It is a prerequisite to run 
    :func:`get_highly_variable_genes` first. 

    Warning: the default cell cycle gene sets are from mouse genes. Please convert them
    to upper case if you want to apply it to human data. Also, consider using your own
    gene sets. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    cycling_gene_list: `list`, optional
        A list of cell cycle correlated genes to compute correlation with. 
    corr_threshold: `float`, optional (default: 0.1)
        Highly variable genes with absolute correlation score about this 
        threshold will be removed from the highly variable gene list. 
    confirm_change: `bool`, optional (default: False)
        If set True, adata.var['highly_variable'] will be updated to exclude
        cell cycle correlated genes. 
    """

    if 'highly_variable' not in adata.var.keys():
        logg.error("Did not find highly variable genes index in adata.var['highly_variable']\n"
                   "Please run get_highly_variable_genes first!")

    else:
        gene_list=np.array(adata.var_names)

        cycling_gene_idx=np.in1d(gene_list,cycling_gene_list)
        if np.sum(cycling_gene_idx)!=len(cycling_gene_list):
            logg.error(f"Provided cyclcing genes: {cycling_gene_list}\n"
              f"They are for mouse genes. Only {np.sum(cycling_gene_idx)} found in the reference gene list.")
        else:
            E=adata.X
            cycling_expression=E[:,cycling_gene_idx].A.T

            highvar_genes_idx=np.array(adata.var['highly_variable'])
            highvar_genes=gene_list[highvar_genes_idx]
            test_expression=E[:,highvar_genes_idx].A.T

            cell_cycle_corr=hf.corr2_coeff(test_expression,cycling_expression)

            #adata.uns['cycling_correlation']

            if (not confirm_change):
                max_corr=np.max(abs(cell_cycle_corr),1)
                fig = plt.figure(figsize = (4, 3.5))
                ax0 = plt.subplot(1,1,1)
                ax0.hist(max_corr,100)
                ax0.set_xlabel('Max. corr. with cycling genes')
                ax0.set_ylabel('Histogram')
                    
                logg.info("adata.var['highly_variable'] not updated.\n"
                           "Please choose corr_threshold properly, and set confirm_change=True")
            else:
                max_corr=np.max(abs(cell_cycle_corr),1)
                noCycle_idx=max_corr<corr_threshold

                highvar_genes_noCycle=highvar_genes[noCycle_idx]
                logg.info(f"Number of selected non-cycling highly variable genes: {len(highvar_genes_noCycle)}\n"
                    f"Remove {np.sum(~noCycle_idx)} cell cycle correlated genes.")

                highvar_genes_noCycle_idx=np.in1d(gene_list,highvar_genes_noCycle)
                adata.var['highly_variable']=highvar_genes_noCycle_idx
                logg.info("adata.var['highly_variable'] updated")


def get_X_pca(adata,n_pca_comp=40):
    """
    Get X_pca.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    n_pca_comp: int, optional (default: 40)
        Number of top principle components to keep

    Returns
    -------
    None. Modify adata.obsm['X_pca']. 
    """

    if 'highly_variable' not in adata.var.keys():
        logg.error("Did not find highly variable genes index in adata.var['highly_variable']\n"
                   "Please run get_highly_variable_genes first!")
    else:

        highvar_genes_idx=np.array(adata.var['highly_variable'])
        gene_list=np.array(adata.var_names)
        highvar_genes=gene_list[highvar_genes_idx]

        adata.obsm['X_pca'] = hf.get_pca(adata[:, highvar_genes].X, numpc=n_pca_comp,keep_sparse=False,normalize=True,random_state=0)


def get_X_emb(adata,n_neighbors=20,umap_min_dist=0.3):
    """
    Get X_emb using :func:`scanpy.tl.umap`

    We assume that X_pca is computed. It first runs KNN graph construction. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    n_neighbors: `int`, optional (default: 20)
        Neighber number for constructing the KNN graph, using the UMAP method. 
    umap_min_dist: `float`, optional (default: 0.3)
        The effective minimum distance between embedded points. 

    Returns
    -------
    None. Modify adata.obsm['X_emb'].
    """

    if not ('X_pca' in adata.obsm.keys()):
        logg.error('*X_pca* missing from adata.obsm... abort the operation')
    else:
        # Number of neighbors for KNN graph construction
        sc.pp.neighbors(adata, n_neighbors=n_neighbors)
        sc.tl.umap(adata, min_dist=umap_min_dist)
        adata.obsm['X_emb']=adata.obsm['X_umap']


def get_state_info(adata,n_neighbors=20,resolution=0.5):
    """
    Update `state_info` using :func:`scanpy.tl.leiden`

    We assume that `adata.obsm['X_pca']` exists. It first runs KNN graph construction. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    n_neighbors: `int`, optional (default: 20)
        Neighber number for constructing the KNN graph, using the UMAP method. 
    resolution: `float`, optional (default: 0.5)
        A parameter value controlling the coarseness of the clustering. 
        Higher values lead to more clusters.

    Returns
    -------
    None. Modify adata.obs['state_info']. 
    """
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    if not ('X_pca' in adata.obsm.keys()):
        logg.error('*X_pca* missing from adata.obsm... abort the operation')
    else:
        # Number of neighbors for KNN graph construction
        sc.tl.leiden(adata,resolution=resolution)
        adata.obs['state_info']=adata.obs['leiden']




############# refine clusters for state_info

def refine_state_info_by_leiden_clustering(adata,selected_time_points=[],
    resolution=0.5,n_neighbors=20,confirm_change=False,cluster_name_prefix='S'):
    """
    Refine state info by clustering states at given time points.

    Select states at desired time points to improve the clustering. When
    first run, set confirm_change=False. Only when you are happy with the 
    result, set confirm_change=True to update adata.obs['state_info'].
    The original state_info will be stored at adata.obs['old_state_info'].  

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    selected_time_points: `list`, optional (default: include all)
        A list of selected time points for clustering. Should be 
        among adata.obs['time_info']. 
    adata: :class:`~anndata.AnnData` object
    n_neighbors: `int`, optional (default: 20)
        Neighber number for constructing the KNN graph, using the UMAP method. 
    resolution: `float`, optional (default: 0.5)
        A parameter value controlling the coarseness of the clustering. 
        Higher values lead to more clusters.
    confirm_change: `bool`, optional (default: False)
        If True, update adata.obs['state_info']
    cluster_name_prefix: `str`, optional (default: 'S')
        prefix for the new cluster name to distinguish it from 
        existing cluster names.

    Returns
    -------
    Update adata.obs['state_info'] if confirm_change=True.
    """

    time_info=adata.obs['time_info']
    available_time_points=list(set(time_info))
    
    if len(selected_time_points)==0:
        selected_time_points=available_time_points

    if np.sum(np.in1d(selected_time_points,available_time_points))!=len(selected_time_points):
        logg.error(f"Selected time points not available. Please select from {available_time_points}")

    else:
        sp_idx=np.zeros(adata.shape[0],dtype=bool)
        for xx in selected_time_points:
            idx=time_info==xx
            sp_idx[idx]=True
            
        adata_sp=sc.AnnData(adata.X[sp_idx]);
        #adata_sp.var_names=adata.var_names
        adata_sp.obsm['X_pca']=adata.obsm['X_pca'][sp_idx]
        adata_sp.obsm['X_emb']=adata.obsm['X_emb'][sp_idx]
        
        sc.pp.neighbors(adata_sp, n_neighbors=n_neighbors)
        sc.tl.leiden(adata_sp,resolution=resolution)

        CSpl.embedding(adata_sp,color='leiden')
        
        if confirm_change:
            logg.info("Change state annotation at adata.obs['state_info']")
            adata.obs['old_state_info']=adata.obs['state_info']
            orig_state_annot=np.array(adata.obs['state_info'])
            temp_array=np.array(adata_sp.obs['leiden'])
            for j in range(len(temp_array)):
                temp_array[j]=cluster_name_prefix+temp_array[j]
            
            orig_state_annot[sp_idx]=temp_array
            adata.obs['state_info']=pd.Categorical(orig_state_annot)
            CSpl.embedding(adata,color='state_info')
        


def refine_state_info_by_marker_genes(adata,marker_genes,express_threshold=0.1,
    selected_time_points=[],new_cluster_name='new_cluster',confirm_change=False,add_neighbor_N=5):
    """
    Refine state info according to marker gene expression.

    In this method, a state is selected if it expresses all genes in the list 
    of marker_genes, and the expression are above the relative `express_threshold`. 
    You can also specify which time point you want to focus on. In addition, we also include 
    cell states neighboring to these valid states to smooth the selection 
    (controlled by add_neighbor_N).
    
    When you run it the first time, set confirm_change=False. Only when you are happy with 
    the result, set confirm_change=True to update the adata.obs['state_info'].
    The original state_info will be stored at adata.obs['old_state_info'].  

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    marker_genes: `list` or 'str'
        List of marker genes to be used for defining cell states.
    express_threshold: `float`, optional (default: 0.1)
        Relative threshold of marker gene expression, in the range [0,1].
        A state must have an expression above this threshold for all genes
        to be included.
    selected_time_points: `list`, optional (default: all)
        A list of selected time points for performing clustering,
        among adata.obs['time_info']. 
    new_cluster_name: `str`, optional (default: 'new_cluster')
    confirm_change: `bool`, optional (default: False)
        If True, update adata.obs['state_info'].
    add_neighbor_N: `int`, optional (default: 5)
        Add to the new cluster neighboring cells of a qualified 
        high-expressing state according to the KNN graph 
        with K=add_neighbor_N.

    Returns
    -------
    Update the adata.obs['state_info'] if confirm_change=True.
    """
    
    time_info=adata.obs['time_info']
    x_emb=adata.obsm['X_emb'][:,0]
    y_emb=adata.obsm['X_emb'][:,1]
    available_time_points=list(set(time_info))
    
    if len(selected_time_points)==0:
        selected_time_points=available_time_points
        
    sp_idx=np.zeros(adata.shape[0],dtype=bool)
    for xx in selected_time_points:
        idx=time_info==xx
        sp_idx[idx]=True
        
        
    # add gene constraints 
    selected_states_idx=np.ones(adata.shape[0],dtype=bool)
    gene_list=list(adata.var_names)
    tot_name=''

    if type(marker_genes)==str:
        marker_genes=[marker_genes]

    for marker_gene_temp in marker_genes:
        if marker_gene_temp in gene_list:
            expression=adata.obs_vector(marker_gene_temp)
            thresh=express_threshold*np.max(expression)
            idx=expression>thresh
            selected_states_idx=selected_states_idx & idx
            
            tot_name=tot_name+marker_gene_temp
            
    # add temporal constraint
    selected_states_idx[~sp_idx]=0    
    
    if np.sum(selected_states_idx)>0:
        # add neighboring cells to smooth selected cells (in case the expression is sparse)
        selected_states_idx=hf.add_neighboring_cells_to_a_map(selected_states_idx,adata,neighbor_N=add_neighbor_N)

        fig_width=settings.fig_width; fig_height=settings.fig_height;
        fig=plt.figure(figsize=(fig_width,fig_height));ax=plt.subplot(1,1,1)
        CSpl.customized_embedding(x_emb,y_emb,selected_states_idx,ax=ax)
        ax.set_title(f"{tot_name}; Cell #: {np.sum(selected_states_idx)}")
        #print(f"Selected cell state number: {np.sum(selected_states_idx)}")


        if confirm_change:
            adata.obs['old_state_info']=adata.obs['state_info']
            logg.info("Change state annotation at adata.obs['state_info']")
            if new_cluster_name=='':
                new_cluster_name=marker_genes[0]

            orig_state_annot=np.array(adata.obs['state_info'])
            orig_state_annot[selected_states_idx]=np.array([new_cluster_name for j in range(np.sum(selected_states_idx))])
            adata.obs['state_info']=pd.Categorical(orig_state_annot)
            CSpl.embedding(adata,color='state_info')
    else:
        logg.error("Either the gene names or the time point names are not right.")



