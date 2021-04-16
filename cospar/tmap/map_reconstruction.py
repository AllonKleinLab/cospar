# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd
import time
import scanpy as sc
import scipy.sparse as ssp 

from .. import help_functions as hf
from .. import plotting as pl
from .optimal_transport import *
from .. import settings
from .. import logging as logg


####################

# Constructing the similarity matrix (similarity matrix)

####################


def generate_similarity_matrix(adata,file_name,round_of_smooth=10,neighbor_N=20,beta=0.1,truncation_threshold=0.001,save_subset=True,use_existing_KNN_graph=False,compute_new_Smatrix=False):
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

    if os.path.exists(file_name+f'_SM{round_of_smooth}.npz') and (not compute_new_Smatrix):
        
        logg.hint("Compute similarity matrix: load existing data")
        similarity_matrix=ssp.load_npz(file_name+f'_SM{round_of_smooth}.npz')
    else: # compute now
        
        logg.hint(f"Compute similarity matrix: computing new; beta={beta}")

        # add a step to compute PCA in case this is not computed 

        if (not use_existing_KNN_graph) or ('connectivities' not in adata.obsp.keys()):
            # here, we assume that adata already has pre-computed PCA
            sc.pp.neighbors(adata, n_neighbors=neighbor_N)
        else:
            logg.hint("Use existing KNN graph at adata.obsp['connectivities'] for generating the smooth matrix")
        adjacency_matrix=adata.obsp['connectivities'];

        ############## The new method
        adjacency_matrix=(adjacency_matrix+adjacency_matrix.T)/2
        ############## 

        adjacency_matrix = hf.sparse_rowwise_multiply(adjacency_matrix, 1 / adjacency_matrix.sum(1).A.squeeze())
        nrow = adata.shape[0]
        similarity_matrix = ssp.lil_matrix((nrow, nrow))
        similarity_matrix.setdiag(np.ones(nrow))
        transpose_A=adjacency_matrix.T
        for iRound in range(round_of_smooth):
            SM=iRound+1
            
            logg.info("Smooth round:",SM)
            t=time.time()
            similarity_matrix =beta*similarity_matrix+(1-beta)*transpose_A*similarity_matrix
            #similarity_matrix =beta*similarity_matrix+(1-beta)*similarity_matrix*adjacency_matrix
            #similarity_matrix_array.append(similarity_matrix)
            
            logg.hint("Time elapsed:",time.time()-t)

            t=time.time()
            sparsity_frac=(similarity_matrix>0).sum()/(similarity_matrix.shape[0]*similarity_matrix.shape[1])
            if sparsity_frac>=0.1:
                #similarity_matrix_truncate=similarity_matrix
                #similarity_matrix_truncate_array.append(similarity_matrix_truncate)
                
                logg.hint(f"Orignal sparsity={sparsity_frac}, Thresholding")
                similarity_matrix=hf.matrix_row_or_column_thresholding(similarity_matrix,truncation_threshold)
                sparsity_frac_2=(similarity_matrix>0).sum()/(similarity_matrix.shape[0]*similarity_matrix.shape[1])
                #similarity_matrix_truncate_array.append(similarity_matrix_truncate)
                
                logg.hint(f"Final sparsity={sparsity_frac_2}")
            
                logg.info(f"similarity matrix truncated (Smooth round={SM}): ", time.time()-t)

            #logg.info("Save the matrix")
            #file_name=f'data/20200221_truncated_similarity_matrix_SM{round_of_smooth}_kNN{neighbor_N}_Truncate{str(truncation_threshold)[2:]}.npz'
            similarity_matrix=ssp.csr_matrix(similarity_matrix)


            ############## The new method
            #similarity_matrix=similarity_matrix.T.copy() 
            ##############


            if save_subset: 
                if SM%5==0: # save when SM=5,10,15,20,...
                    
                    logg.hint("Save the matrix at every 5 rounds")
                    ssp.save_npz(file_name+f'_SM{SM}.npz',similarity_matrix)
            else: # save all
                
                logg.hint("Save the matrix at every round")
                ssp.save_npz(file_name+f'_SM{SM}.npz',similarity_matrix)
        

    return similarity_matrix




def generate_initial_similarity(similarity_matrix,initial_index_0,initial_index_1):
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
    
    t=time.time()
    initial_similarity=similarity_matrix[initial_index_0][:,initial_index_1];
    #initial_similarity=hf.sparse_column_multiply(initial_similarity,1/(resol+initial_similarity.sum(0)))
    if ssp.issparse(initial_similarity): initial_similarity=initial_similarity.A
    
    logg.hint("Time elapsed: ", time.time()-t)
    return initial_similarity 


def generate_final_similarity(similarity_matrix,final_index_0,final_index_1):
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
    
    t=time.time()
    final_similarity=similarity_matrix.T[final_index_0][:,final_index_1];
    if ssp.issparse(final_similarity):final_similarity=final_similarity.A
    #final_similarity=hf.sparse_rowwise_multiply(final_similarity,1/(resol+final_similarity.sum(1)))
    
    logg.hint("Time elapsed: ", time.time()-t)
    return final_similarity



def select_time_points(adata_orig,time_point=['day_1','day_2'],extend_Tmap_space=False):
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
        If true, the initial and later state space for Tmap will be expanded to all cells,
        whether they have clonal barcodes or not. Otherwise, the initial and later state
        space of the Tmap will be restricted to cells with multi-time clonal information
        alone. The latter case speeds up the computation, which is recommended.

    Returns
    -------
    Subsampled :class:`~anndata.AnnData` object
    """
    
    #x_emb_orig=adata_orig.obsm['X_emb'][:,0]
    #y_emb_orig=adata_orig.obsm['X_emb'][:,1]
    time_info_orig=np.array(adata_orig.obs['time_info'])
    clone_annot_orig=adata_orig.obsm['X_clone']
    if len(time_point)==0: # use all clonally labelled cell states 
        time_point=np.sort(list(set(time_info_orig))) # this automatic ordering may not work

    if (len(time_point)<2):
        logg.error("Must select more than 1 time point!")
    else:

        At=[]
        for j, time_0 in enumerate(time_point):
            At.append(ssp.csr_matrix(clone_annot_orig[time_info_orig==time_0]))

        ### Day t - t+1
        Clonal_cell_ID_FOR_t=[]
        for j in range(len(time_point)-1):
            idx_t=np.array((At[j]*At[j+1].T).sum(1)>0).flatten()
            time_index_t=time_info_orig==time_point[j]
            temp=np.nonzero(time_index_t)[0][idx_t]
            Clonal_cell_ID_FOR_t.append(temp) # this index is in the original space, without sampling etc
            
            logg.hint(f"Clonal cell fraction (day {time_point[j]}-{time_point[j+1]}):",len(temp)/np.sum(time_index_t))

        ### Day t+1 - t
        Clonal_cell_ID_BACK_t=[]
        for j in range(len(time_point)-1):
            idx_t=np.array((At[j+1]*At[j].T).sum(1)>0).flatten()
            time_index_t=time_info_orig==time_point[j+1]
            temp=np.nonzero(time_index_t)[0][idx_t]
            Clonal_cell_ID_BACK_t.append(temp) # this index is in the original space, without sampling etc
            
            logg.hint(f"Clonal cell fraction (day {time_point[j+1]}-{time_point[j]}):",len(temp)/np.sum(time_index_t))

        
        for j in range(len(time_point)-1):    
            logg.hint(f"Numer of cells that are clonally related -- day {time_point[j]}: {len(Clonal_cell_ID_FOR_t[j])}  and day {time_point[j+1]}: {len(Clonal_cell_ID_BACK_t[j])}")

        proportion=np.ones(len(time_point))
        # flatten the list
        flatten_clonal_cell_ID_FOR=np.array([sub_item for item in Clonal_cell_ID_FOR_t for sub_item in item])
        flatten_clonal_cell_ID_BACK=np.array([sub_item for item in Clonal_cell_ID_BACK_t for sub_item in item])
        valid_clone_N_FOR=np.sum(clone_annot_orig[flatten_clonal_cell_ID_FOR].A.sum(0)>0)
        valid_clone_N_BACK=np.sum(clone_annot_orig[flatten_clonal_cell_ID_BACK].A.sum(0)>0)

        
        logg.info(f"Number of multi-time clones post selection: {valid_clone_N_FOR}")
        #logg.info("Valid clone number 'BACK' post selection",valid_clone_N_BACK)


        ###################### select initial and later cell states

        if extend_Tmap_space:
            old_Tmap_cell_id_t1=[]
            for t_temp in time_point[:-1]:
                old_Tmap_cell_id_t1=old_Tmap_cell_id_t1+list(np.nonzero(time_info_orig==t_temp)[0])
            old_Tmap_cell_id_t1=np.array(old_Tmap_cell_id_t1)

            ########
            old_Tmap_cell_id_t2=[]
            for t_temp in time_point[1:]:
                old_Tmap_cell_id_t2=old_Tmap_cell_id_t2+list(np.nonzero(time_info_orig==t_temp)[0])
            old_Tmap_cell_id_t2=np.array(old_Tmap_cell_id_t2)

        else:
            old_Tmap_cell_id_t1=flatten_clonal_cell_ID_FOR
            old_Tmap_cell_id_t2=flatten_clonal_cell_ID_BACK


        old_clonal_cell_id_t1=flatten_clonal_cell_ID_FOR
        old_clonal_cell_id_t2=flatten_clonal_cell_ID_BACK
        ########################

        sp_id=np.sort(list(set(list(old_Tmap_cell_id_t1)+list(old_Tmap_cell_id_t2))))
        sp_idx=np.zeros(clone_annot_orig.shape[0],dtype=bool)
        sp_idx[sp_id]=True

        Tmap_cell_id_t1=hf.converting_id_from_fullSpace_to_subSpace(old_Tmap_cell_id_t1,sp_id)[0]
        clonal_cell_id_t1=hf.converting_id_from_fullSpace_to_subSpace(old_clonal_cell_id_t1,sp_id)[0]
        clonal_cell_id_t2=hf.converting_id_from_fullSpace_to_subSpace(old_clonal_cell_id_t2,sp_id)[0]
        Tmap_cell_id_t2=hf.converting_id_from_fullSpace_to_subSpace(old_Tmap_cell_id_t2,sp_id)[0]

        Clonal_cell_ID_FOR_t_new=[]
        for temp_id_list in Clonal_cell_ID_FOR_t:
            convert_list=hf.converting_id_from_fullSpace_to_subSpace(temp_id_list,sp_id)[0]
            Clonal_cell_ID_FOR_t_new.append(convert_list)

        Clonal_cell_ID_BACK_t_new=[]
        for temp_id_list in Clonal_cell_ID_BACK_t:
            convert_list=hf.converting_id_from_fullSpace_to_subSpace(temp_id_list,sp_id)[0]
            Clonal_cell_ID_BACK_t_new.append(convert_list)


        sp_id_0=np.sort(list(old_clonal_cell_id_t1)+list(old_clonal_cell_id_t2))
        sp_idx_0=np.zeros(clone_annot_orig.shape[0],dtype=bool)
        sp_idx_0[sp_id_0]=True

        barcode_id=np.nonzero(clone_annot_orig[sp_idx_0].A.sum(0).flatten()>0)[0]
        #sp_id=np.nonzero(sp_idx)[0]
        clone_annot=clone_annot_orig[sp_idx][:,barcode_id]


        adata=adata_orig[sp_idx]
        # adata=sc.AnnData(adata_orig.X[sp_idx]);
        # adata.var_names=adata_orig.var_names
        # adata.obsm['X_pca']=adata_orig.obsm['X_pca'][sp_idx]
        # adata.obsm['X_emb']=adata_orig.obsm['X_emb'][sp_idx]
        # adata.obs['state_info']=pd.Categorical(adata_orig.obs['state_info'][sp_idx])
        # adata.obs['time_info']=pd.Categorical(adata_orig.obs['time_info'][sp_idx])
        
        adata.obsm['X_clone']=clone_annot
        adata.uns['clonal_cell_id_t1']=clonal_cell_id_t1
        adata.uns['clonal_cell_id_t2']=clonal_cell_id_t2
        adata.uns['Tmap_cell_id_t1']=Tmap_cell_id_t1
        adata.uns['Tmap_cell_id_t2']=Tmap_cell_id_t2
        adata.uns['multiTime_cell_id_t1']=Clonal_cell_ID_FOR_t_new
        adata.uns['multiTime_cell_id_t2']=Clonal_cell_ID_BACK_t_new
        adata.uns['proportion']=np.ones(len(time_point)-1)
        adata.uns['sp_idx']=sp_idx

        data_des_orig=adata_orig.uns['data_des'][0]
        data_des_0=adata_orig.uns['data_des'][-1]
        time_label='t'
        for x in time_point:
            time_label=time_label+f'*{x}'

        data_des=data_des_0+f'_MultiTimeClone_FullSpace{int(extend_Tmap_space)}_{time_label}'
        adata.uns['data_des']=[data_des_orig,data_des]

        if logg._settings_verbosity_greater_or_equal_than(3):
            N_cell,N_clone=clone_annot.shape;
            logg.info(f"Cell number={N_cell}, Clone number={N_clone}")
            x_emb=adata.obsm['X_emb'][:,0]
            y_emb=adata.obsm['X_emb'][:,1]
            pl.customized_embedding(x_emb,y_emb,-x_emb)

        return adata        



####################

# CoSpar: two-time points

####################


# v0, sparsify in the beginning; so the output is the smoothed matrix with 
# full entries (should be very big). We can sparsify it after the map converges
def refine_Tmap_through_cospar(MultiTime_cell_id_array_t1,MultiTime_cell_id_array_t2,
    proportion,transition_map,X_clone,initial_similarity,final_similarity,
    sparsity_threshold=0.1,normalization_mode=1):
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

    #logg.warn('CoSpar-v0; another normalization')
    resol=10**(-10)

    transition_map=hf.matrix_row_or_column_thresholding(transition_map,sparsity_threshold,row_threshold=True)

    
    if normalization_mode==0: logg.hint("Single-cell normalization")
    if normalization_mode==1: logg.hint("Clone normalization")

    if ssp.issparse(X_clone):
        X_clone=ssp.csr_matrix(X_clone)

    cell_N,clone_N=X_clone.shape
    N1,N2=transition_map.shape
    new_coupling_matrix=ssp.lil_matrix((N1,N2))

    # cell id order in the similarity matrix is obtained by concatenating the cell id 
    # list in MultiTime_cell_id_array_t1. So, we need to offset the id if we move to the next list
    offset_N1=0; 
    offset_N2=0;
    for j in range(len(MultiTime_cell_id_array_t1)):
        
        logg.hint("Relative time point pair index:",j)
        cell_id_array_t1=MultiTime_cell_id_array_t1[j]
        cell_id_array_t2=MultiTime_cell_id_array_t2[j]


        for clone_id in range(clone_N):
            #pdb.set_trace()
            
            if clone_id%1000==0: logg.hint("Clone id:",clone_id)
            idx1=X_clone[cell_id_array_t1,clone_id].A.flatten()
            idx2=X_clone[cell_id_array_t2,clone_id].A.flatten()
            if idx1.sum()>0 and idx2.sum()>0:
                ## update the new_coupling matrix
                id_1=offset_N1+np.nonzero(idx1)[0]
                id_2=offset_N2+np.nonzero(idx2)[0]
                prob=transition_map[id_1][:,id_2]
                

                ## try row normalization
                if normalization_mode==0:
                    prob=hf.sparse_rowwise_multiply(prob,1/(resol+np.sum(prob,1))) # cell-level normalization
                else:
                    prob=prob/(resol+np.sum(prob)) # clone level normalization, account for proliferation

                weight_factor=np.sqrt(np.mean(idx1[idx1>0])*np.mean(idx2[idx2>0])) # the contribution of a particular clone can be tuned by its average entries
                if (weight_factor>1):
                    logg.hint("X_clone has entries not 0 or 1. Using weight modulation",weight_factor)

                #Use the add mode, add up contributions from each clone
                new_coupling_matrix[id_1[:,np.newaxis],id_2]=new_coupling_matrix[id_1[:,np.newaxis],id_2]+proportion[j]*prob*weight_factor 

        ## update offset
        offset_N1=offset_N1+len(cell_id_array_t1)
        offset_N2=offset_N2+len(cell_id_array_t2)
            

    ## rescale
    new_coupling_matrix=new_coupling_matrix/(new_coupling_matrix.A.max())

    ## convert to sparse matrix form
    new_coupling_matrix=new_coupling_matrix.tocsr()

    
    logg.hint("Start to smooth the refined clonal map")
    t=time.time()
    temp=new_coupling_matrix*final_similarity
    
    logg.hint("Phase I: time elapsed -- ", time.time()-t)
    smoothed_new_transition_map=initial_similarity.dot(temp)
    
    logg.hint("Phase II: time elapsed -- ", time.time()-t)

    # both return are numpy array
    un_SM_transition_map=new_coupling_matrix.A
    #smoothed_new_transition_map=hf.sparse_rowwise_multiply(smoothed_new_transition_map,1/(resol+np.sum(smoothed_new_transition_map,1)))
    return smoothed_new_transition_map, un_SM_transition_map




def refine_Tmap_through_cospar_noSmooth(MultiTime_cell_id_array_t1,
    MultiTime_cell_id_array_t2,proportion,transition_map,
    X_clone,sparsity_threshold=0.1,normalization_mode=1):
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

    if not isinstance(X_clone[0,0], bool):
        X_clone=X_clone.astype(bool)

    resol=10**(-10)
    
    if normalization_mode==0: logg.hint("Single-cell normalization")
    if normalization_mode==1: logg.hint("Clone normalization")

    transition_map=hf.matrix_row_or_column_thresholding(transition_map,sparsity_threshold,row_threshold=True)
    
    if not ssp.issparse(transition_map): transition_map=ssp.csr_matrix(transition_map)
    if not ssp.issparse(X_clone): X_clone=ssp.csr_matrix(X_clone)

    cell_N,clone_N=X_clone.shape
    N1,N2=transition_map.shape
    new_coupling_matrix=ssp.lil_matrix((N1,N2))

    # cell id order in the similarity matrix is obtained by concatenating the cell id 
    # list in MultiTime_cell_id_array_t1. So, we need to offset the id if we move to the next list
    offset_N1=0; 
    offset_N2=0;
    for j in range(len(MultiTime_cell_id_array_t1)):
        
        logg.hint("Relative time point pair index:",j)
        cell_id_array_t1=MultiTime_cell_id_array_t1[j]
        cell_id_array_t2=MultiTime_cell_id_array_t2[j]


        for clone_id in range(clone_N):
            
            if clone_id%1000==0: logg.hint("Clone id:",clone_id)
            idx1=X_clone[cell_id_array_t1,clone_id].A.flatten()
            idx2=X_clone[cell_id_array_t2,clone_id].A.flatten()
            if idx1.sum()>0 and idx2.sum()>0:
                ## update the new_coupling matrix
                id_1=offset_N1+np.nonzero(idx1)[0]
                id_2=offset_N2+np.nonzero(idx2)[0]
                prob=transition_map[id_1][:,id_2].A
                

                 ## try row normalization
                if normalization_mode==0:
                    prob=hf.sparse_rowwise_multiply(prob,1/(resol+np.sum(prob,1))) # cell-level normalization
                else:
                    prob=prob/(resol+np.sum(prob)) # clone level normalization, account for proliferation


                weight_factor=np.sqrt(np.mean(idx1[idx1>0])*np.mean(idx2[idx2>0])) # the contribution of a particular clone can be tuned by its average entries
                if (weight_factor>1):
                    logg.hint("marker gene weight",weight_factor)

                #Use the add mode, add up contributions from each clone
                new_coupling_matrix[id_1[:,np.newaxis],id_2]=new_coupling_matrix[id_1[:,np.newaxis],id_2]+proportion[j]*prob*weight_factor 

        ## update offset
        offset_N1=offset_N1+len(cell_id_array_t1)
        offset_N2=offset_N2+len(cell_id_array_t2)
            

    ## convert to sparse matrix form
    new_coupling_matrix=new_coupling_matrix.tocsr()
    #
    un_SM_transition_map=new_coupling_matrix
    return  un_SM_transition_map


###############

# v1 version, allows to set later time point
def infer_Tmap_from_multitime_clones(adata_orig,clonal_time_points=None,
    later_time_point=None,smooth_array=[15,10,5],CoSpar_KNN=20,sparsity_threshold=0.1,
    intraclone_threshold=0.05,normalization_mode=1,extend_Tmap_space=False,save_subset=True,
    use_full_Smatrix=True,trunca_threshold=[0.001,0.01],compute_new=False,max_iter_N=5,epsilon_converge=0.05):
    """
    Infer transition map for clonal data with multiple time points.

    It prepares adata object for cells of targeted time points by 
    :func:`.select_time_points`, generates the similarity matrix 
    via :func:`.generate_similarity_matrix`, and iteratively calls 
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
        If true, the initial and later state space for Tmap will be expanded to all cells,
        whether they have clonal barcodes or not. Otherwise, the initial and later state
        space of the Tmap will be restricted to cells with multi-time clonal information
        alone. The latter case usually speeds up the computation. 
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

    t0=time.time()
    if 'data_des' not in adata_orig.uns.keys():
        adata_orig.uns['data_des']=['cospar']
    hf.check_available_clonal_info(adata_orig)
    clonal_time_points_0=np.array(adata_orig.uns['clonal_time_points'])
    if type(later_time_point)==list: later_time_point=later_time_point[0]
    if len(clonal_time_points_0)<2:
        logg.error("There are no multi-time clones. Abort the inference.")
        return None

    else:
        if clonal_time_points is None:
            clonal_time_points=clonal_time_points_0

        if (later_time_point is not None) and (later_time_point not in clonal_time_points_0):
            logg.error(f"later_time_point is not all among {clonal_time_points_0}. Computation aborted!")
            return None 

        if (later_time_point is not None):
            clonal_time_points=list(clonal_time_points)+[later_time_point]
            clonal_time_points=list(set(clonal_time_points))

        N_valid_time=np.sum(np.in1d(clonal_time_points_0,clonal_time_points))
        if (N_valid_time!=len(clonal_time_points)) or (N_valid_time<2): 
            logg.error(f"Selected time points are not all among {clonal_time_points_0}, or less than 2 time points are selected. Computation aborted!")
            return None

    if save_subset:
        if not (np.all(np.diff(smooth_array)<=0) and np.all(np.array(smooth_array)%5==0)):
            logg.error("The smooth_array contains numbers not multiples of 5 or not in descending order.\n"
             "The correct form is like [20,15,10], or [10,10,10,5]. Its length determines the number of iteration.\n"
              "You can also set save_subset=False to explore arbitrary smooth_array structure.")
            return None

    time_ordering=adata_orig.uns['time_ordering']
    sel_idx_temp=np.in1d(time_ordering,clonal_time_points)
    clonal_time_points=time_ordering[sel_idx_temp]

    logg.info("------Compute the full Similarity matrix if necessary------")
    data_path=settings.data_path
    if use_full_Smatrix: # prepare the similarity matrix with all state info, all subsequent similarity will be down-sampled from this one.

        temp_str='0'+str(trunca_threshold[0])[2:]
        round_of_smooth=np.max(smooth_array)
        data_des=adata_orig.uns['data_des'][0]
        similarity_file_name=f'{data_path}/{data_des}_Similarity_matrix_with_all_cell_states_kNN{CoSpar_KNN}_Truncate{temp_str}'
        if not (os.path.exists(similarity_file_name+f'_SM{round_of_smooth}.npz') and (not compute_new)):
            similarity_matrix_full=generate_similarity_matrix(adata_orig,similarity_file_name,round_of_smooth=round_of_smooth,
                        neighbor_N=CoSpar_KNN,truncation_threshold=trunca_threshold[0],save_subset=save_subset,compute_new_Smatrix=compute_new)

    # compute transition map between neighboring time points
    if later_time_point is None:
        logg.info("----Infer transition map between neighboring time points-----")
        logg.info("Step 1: Select time points")
        adata=select_time_points(adata_orig,time_point=clonal_time_points,extend_Tmap_space=extend_Tmap_space)


        logg.info("Step 2: Optimize the transition map recursively")
        infer_Tmap_from_multitime_clones_private(adata,smooth_array=smooth_array,neighbor_N=CoSpar_KNN,sparsity_threshold=sparsity_threshold,intraclone_threshold=intraclone_threshold,normalization_mode=normalization_mode,
                save_subset=save_subset,use_full_Smatrix=use_full_Smatrix,
                trunca_threshold=trunca_threshold,compute_new_Smatrix=compute_new,
                max_iter_N=max_iter_N,epsilon_converge=epsilon_converge)    



        if 'Smatrix' in adata.uns.keys():
            adata.uns.pop('Smatrix')

        logg.info(f"-----------Total used time: {time.time()-t0} s ------------")
        return adata

    else:
        # compute transition map between initial time points and the later time point
        sel_id=np.nonzero(np.in1d(clonal_time_points,later_time_point))[0][0]
        initial_time_points=clonal_time_points[:sel_id]

        time_info_orig=np.array(adata_orig.obs['time_info'])
        sp_idx=np.zeros(adata_orig.shape[0],dtype=bool)
        all_time_points=list(initial_time_points)+[later_time_point]
        label='t'
        for xx in all_time_points:
            id_array=np.nonzero(time_info_orig==xx)[0]
            sp_idx[id_array]=True
            label=label+'*'+str(xx)


        adata=adata_orig[sp_idx]
        # adata=sc.AnnData(adata_orig.X[sp_idx]);
        # adata.var_names=adata_orig.var_names
        # adata.obsm['X_pca']=adata_orig.obsm['X_pca'][sp_idx]
        # adata.obsm['X_emb']=adata_orig.obsm['X_emb'][sp_idx]
        # adata.obs['state_info']=pd.Categorical(adata_orig.obs['state_info'][sp_idx])
        # adata.obs['time_info']=pd.Categorical(adata_orig.obs['time_info'][sp_idx])
        # clone_annot_orig=adata_orig.obsm['X_clone']        
        # clone_annot=clone_annot_orig[sp_idx]
        # adata.obsm['X_clone']=clone_annot


        data_des_orig=adata_orig.uns['data_des'][0]
        data_des_0=adata_orig.uns['data_des'][-1]
        data_des=data_des_0+f'_MultiTimeClone_Later_FullSpace{int(extend_Tmap_space)}_{label}'
        adata.uns['data_des']=[data_des_orig,data_des]


        time_info=np.array(adata.obs['time_info'])
        time_index_t2=time_info==later_time_point
        time_index_t1=~time_index_t2

        #### used for similarity matrix generation
        Tmap_cell_id_t1=np.nonzero(time_index_t1)[0]
        Tmap_cell_id_t2=np.nonzero(time_index_t2)[0]
        adata.uns['Tmap_cell_id_t1']=Tmap_cell_id_t1
        adata.uns['Tmap_cell_id_t2']=Tmap_cell_id_t2
        adata.uns['clonal_cell_id_t1']=Tmap_cell_id_t1
        adata.uns['clonal_cell_id_t2']=Tmap_cell_id_t2
        adata.uns['sp_idx']=sp_idx
        data_path=settings.data_path

        transition_map=np.zeros((len(Tmap_cell_id_t1),len(Tmap_cell_id_t2)))
        intraclone_transition_map=np.zeros((len(Tmap_cell_id_t1),len(Tmap_cell_id_t2)))


        logg.info("------Infer transition map between initial time points and the later time one------")
        for yy in initial_time_points:
            
            logg.info(f"--------Current initial time point: {yy}--------")

            logg.info("Step 1: Select time points")
            adata_temp=select_time_points(adata_orig,time_point=[yy,later_time_point],extend_Tmap_space=True) # for this to work, we need to set extend_Tmap_space=True, otherwise for different initial time points, the later Tmap_cell_id_t2 may be different


            logg.info("Step 2: Optimize the transition map recursively")
            infer_Tmap_from_multitime_clones_private(adata_temp,smooth_array=smooth_array,neighbor_N=CoSpar_KNN,sparsity_threshold=sparsity_threshold,intraclone_threshold=intraclone_threshold,normalization_mode=normalization_mode,
                    save_subset=save_subset,use_full_Smatrix=use_full_Smatrix,
                    trunca_threshold=trunca_threshold,compute_new_Smatrix=compute_new,
                    max_iter_N=max_iter_N,epsilon_converge=epsilon_converge)    

            if adata_temp is not None:
                temp_id_t1=np.nonzero(time_info==yy)[0]
                sp_id_t1=hf.converting_id_from_fullSpace_to_subSpace(temp_id_t1,Tmap_cell_id_t1)[0]
                

                transition_map[sp_id_t1,:]=adata_temp.uns['transition_map'].A
                intraclone_transition_map[sp_id_t1,:]=adata_temp.uns['intraclone_transition_map'].A

                if 'Smatrix' in adata_temp.uns.keys():
                    adata_temp.uns.pop('Smatrix')
            else:
                logg.error('Incorrect return')
                return None

        adata.uns['transition_map']=ssp.csr_matrix(transition_map)
        adata.uns['intraclone_transition_map']=ssp.csr_matrix(intraclone_transition_map)


        logg.info(f"-----------Total used time: {time.time()-t0} s ------------")
        return adata
  



# new scheme, with converence check
def infer_Tmap_from_multitime_clones_private(adata,smooth_array=[15,10,5],neighbor_N=20,
    sparsity_threshold=0.1,intraclone_threshold=0.05,normalization_mode=1,save_subset=True,
    use_full_Smatrix=True,trunca_threshold=[0.001,0.01],compute_new_Smatrix=False,max_iter_N=5,epsilon_converge=0.05):
    """
    Internal function for Tmap inference from multi-time clonal data.

    Same as :func:`.infer_Tmap_from_multitime_clones` except that it 
    assumes that the adata object has been prepared for targeted 
    time points. It generates the similarity matrix 
    via :func:`.generate_similarity_matrix`, and iteratively calls 
    the core function :func:`.refine_Tmap_through_cospar` to update 
    the transition map. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Should be prepared by :func:`.select_time_points`
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
    clone_annot=adata.obsm['X_clone']
    clonal_cell_id_t1=adata.uns['clonal_cell_id_t1']
    clonal_cell_id_t2=adata.uns['clonal_cell_id_t2']
    Tmap_cell_id_t1=adata.uns['Tmap_cell_id_t1']
    Tmap_cell_id_t2=adata.uns['Tmap_cell_id_t2']
    sp_idx=adata.uns['sp_idx']
    data_des=adata.uns['data_des'][0] # original label
    data_des_1=adata.uns['data_des'][-1] # current label, sensitive to selected time points
    multiTime_cell_id_t1=adata.uns['multiTime_cell_id_t1']
    multiTime_cell_id_t2=adata.uns['multiTime_cell_id_t2']
    proportion=adata.uns['proportion']
    data_path=settings.data_path

    ######### check whether we need to extend the map space 
    ratio_t1=np.sum(np.in1d(Tmap_cell_id_t1,clonal_cell_id_t1))/len(Tmap_cell_id_t1)
    ratio_t2=np.sum(np.in1d(Tmap_cell_id_t2,clonal_cell_id_t2))/len(Tmap_cell_id_t2)
    if (ratio_t1==1) and (ratio_t2==1): 
        extend_Tmap_space=False  # no need to extend the map space
    else:
        extend_Tmap_space=True

    
    ########################### Compute the transition map 
    
    #logg.info("---------Compute the transition map-----------")

    #trunca_threshold=[0.001,0.01] # this value is only for reducing the computed matrix size for saving
    temp_str='0'+str(trunca_threshold[0])[2:]

    if use_full_Smatrix:
        similarity_file_name=f'{data_path}/{data_des}_Similarity_matrix_with_all_cell_states_kNN{neighbor_N}_Truncate{temp_str}'
        for round_of_smooth in smooth_array:
            if not os.path.exists(similarity_file_name+f'_SM{round_of_smooth}.npz'):
                logg.error(f"Similarity matrix at given parameters have not been computed before! File name: {similarity_file_name}")
                logg.error(f'Please re-run the function with: compute_new=True. If you want to use smooth round not the multiples of 5, set save_subset=False')     
                return None  

    else:
        similarity_file_name=f'{data_path}/{data_des_1}_Similarity_matrix_with_selected_states_kNN{neighbor_N}_Truncate{temp_str}'

    initial_similarity_array=[]
    final_similarity_array=[]
    initial_similarity_array_ext=[]
    final_similarity_array_ext=[]

    logg.info("Load pre-computed similarity matrix")

    if 'Smatrix' not in adata.uns.keys(): 
        logg.hint("Load from hard disk--------")
        for round_of_smooth in smooth_array:
            # we cannot force it to compute new at this time. Otherwise, if we use_full_Smatrix, the resulting similarity is actually from adata, thus not full similarity. 

            re_compute=(not use_full_Smatrix) and (compute_new_Smatrix) # re-compute only when not using full similarity 
            similarity_matrix_full=generate_similarity_matrix(adata,similarity_file_name,round_of_smooth=round_of_smooth,
                        neighbor_N=neighbor_N,truncation_threshold=trunca_threshold[0],save_subset=save_subset,compute_new_Smatrix=re_compute)

            if use_full_Smatrix:
                #pdb.set_trace()
                similarity_matrix_full_sp=similarity_matrix_full[sp_idx][:,sp_idx]

                ### minimum similarity matrix that only involves the multi-time clones
                initial_similarity=generate_initial_similarity(similarity_matrix_full_sp,clonal_cell_id_t1,clonal_cell_id_t1)
                final_similarity=generate_final_similarity(similarity_matrix_full_sp,clonal_cell_id_t2,clonal_cell_id_t2)

                if extend_Tmap_space:
                    initial_similarity_ext=generate_initial_similarity(similarity_matrix_full_sp,Tmap_cell_id_t1,clonal_cell_id_t1)
                    final_similarity_ext=generate_final_similarity(similarity_matrix_full_sp,clonal_cell_id_t2,Tmap_cell_id_t2)
                    
            else:
                initial_similarity=generate_initial_similarity(similarity_matrix_full,clonal_cell_id_t1,clonal_cell_id_t1)
                final_similarity=generate_final_similarity(similarity_matrix_full,clonal_cell_id_t2,clonal_cell_id_t2)

                if extend_Tmap_space:
                    initial_similarity_ext=generate_initial_similarity(similarity_matrix_full,Tmap_cell_id_t1,clonal_cell_id_t1)
                    final_similarity_ext=generate_final_similarity(similarity_matrix_full,clonal_cell_id_t2,Tmap_cell_id_t2)

            initial_similarity_array.append(initial_similarity)
            final_similarity_array.append(final_similarity)
            if extend_Tmap_space:
                initial_similarity_array_ext.append(initial_similarity_ext)
                final_similarity_array_ext.append(final_similarity_ext)

        # loading the map is too costly. We attach it to adata, and remove that after Tmap inference
        # This is useful only for the joint optimization.
        adata.uns['Smatrix']={}
        adata.uns['Smatrix']['initial_similarity_array']=initial_similarity_array
        adata.uns['Smatrix']['final_similarity_array']=final_similarity_array
        adata.uns['Smatrix']['initial_similarity_array_ext']=initial_similarity_array_ext
        adata.uns['Smatrix']['final_similarity_array_ext']=final_similarity_array_ext

    else:
        logg.hint("Copy from adata (pre-loaded)--------")
        initial_similarity_array=adata.uns['Smatrix']['initial_similarity_array']
        final_similarity_array=adata.uns['Smatrix']['final_similarity_array']
        initial_similarity_array_ext=adata.uns['Smatrix']['initial_similarity_array_ext']
        final_similarity_array_ext=adata.uns['Smatrix']['final_similarity_array_ext']


    #### Compute the core of the transition map that involve multi-time clones, then extend to other cell states
    transition_map=np.ones((len(clonal_cell_id_t1),len(clonal_cell_id_t2)))
    #transition_map_array=[transition_map_v1]



    X_clone=clone_annot.copy()
    if not ssp.issparse(X_clone):
        X_clone=ssp.csr_matrix(X_clone)


    #smooth_iter_N=len(smooth_array)
    for j in range(max_iter_N):
        
        #transition_map=Tmap_temp
        if j<len(smooth_array):
            
            logg.info(f"Iteration {j+1}, Use smooth_round={smooth_array[j]}")
            used_initial_similarity=initial_similarity_array[j]
            used_final_similarity=final_similarity_array[j]
        else:
            
            logg.info(f"Iteration {j+1}, Use smooth_round={smooth_array[-1]}")
            used_initial_similarity=initial_similarity_array[-1]
            used_final_similarity=final_similarity_array[-1]

        # transition_map, unSM_sc_coupling=refine_transition_map_by_integrating_clonal_info(clonal_cell_id_t1,clonal_cell_id_t2,
        #        transition_map,X_clone,used_initial_similarity,used_final_similarity,sparsity_threshold,row_normalize=True,normalization_mode=normalization_mode)

        
        transition_map_new, unSM_sc_coupling=refine_Tmap_through_cospar(multiTime_cell_id_t1,multiTime_cell_id_t2,
            proportion,transition_map,X_clone,used_initial_similarity,used_final_similarity,sparsity_threshold=sparsity_threshold,normalization_mode=normalization_mode)


#       ########################### Convergency test
        # sample cell states to convergence test
        sample_N_x=50
        sample_N_y=100
        t0=time.time()
        cell_N_tot_x=transition_map.shape[0]
        if cell_N_tot_x<sample_N_x:
            sample_id_temp_x=np.arange(cell_N_tot_x)
        else:
            xx=np.arange(cell_N_tot_x)
            yy=list(np.nonzero(xx%3==0)[0])+list(np.nonzero(xx%3==1)[0])+list(np.nonzero(xx%3==2)[0])
            sample_id_temp_x=yy[:sample_N_x]

        cell_N_tot_y=transition_map.shape[1]
        if cell_N_tot_y<sample_N_y:
            sample_id_temp_y=np.arange(cell_N_tot_y)
        else:
            xx=np.arange(cell_N_tot_y)
            yy=list(np.nonzero(xx%3==0)[0])+list(np.nonzero(xx%3==1)[0])+list(np.nonzero(xx%3==2)[0])
            sample_id_temp_y=yy[:sample_N_y]


        # transition_map is changed by refine_Tmap_through_cospar (thresholding). So, we only use transition_map_new to update
        if j==0:
            X_map_0=transition_map[sample_id_temp_x,:][:,sample_id_temp_y]
        else:
            X_map_0=X_map_1.copy()

        X_map_1=transition_map_new[sample_id_temp_x,:][:,sample_id_temp_y].copy()
        transition_map=transition_map_new

        if (j>=2) and (j+1>=len(smooth_array)) : # only perform convergency test after at least 3 iterations
            verbose=logg._settings_verbosity_greater_or_equal_than(3)
            corr_X=np.diag(hf.corr2_coeff(X_map_0,X_map_1)).mean()
            if verbose:
                from matplotlib import pyplot as plt
                fig=plt.figure()
                ax=plt.subplot(1,1,1)
                ax.plot(X_map_0.flatten(),X_map_1.flatten(),'.r')
                ax.set_xlabel('$T_{ij}$: previous iteration')
                ax.set_ylabel('$T_{ij}$: current iteration')
                ax.set_title(f'CoSpar, iter_N={j+1}, R={int(100*corr_X)/100}')
                plt.show()
            else:
                logg.info(f"Convergence (CoSpar, iter_N={j+1}): corr(previous_T, current_T)={int(1000*corr_X)/1000}")
                #logg.info(f"Convergence (CoSpar, iter_N={j+1}): corr(previous_T, current_T)={corr_X}; cost time={time.time()-t0}")

            if (1-corr_X)<epsilon_converge: break
        #############################

        

    ### expand the map to other cell states
    if not extend_Tmap_space:
        
        logg.hint("No need for Final Smooth (i.e., clonally-labeled states are the final state space for Tmap)")
        
        transition_map=hf.matrix_row_or_column_thresholding(transition_map,threshold=trunca_threshold[1],row_threshold=True)
        adata.uns['transition_map']=ssp.csr_matrix(transition_map)
    else:
        
        logg.hint("Final round of Smooth (to expand the state space of Tmap to include non-clonal states)")

        if j<len(smooth_array):
            used_initial_similarity_ext=initial_similarity_array_ext[j]
            used_final_similarity_ext=final_similarity_array_ext[j]
        else:
            used_initial_similarity_ext=initial_similarity_array_ext[-1]
            used_final_similarity_ext=final_similarity_array_ext[-1]

        unSM_sc_coupling=ssp.csr_matrix(unSM_sc_coupling)
        t=time.time()
        temp=unSM_sc_coupling*used_final_similarity_ext
        
        logg.hint("Phase I: time elapsed -- ", time.time()-t)
        transition_map_1=used_initial_similarity_ext.dot(temp)
        
        logg.hint("Phase II: time elapsed -- ", time.time()-t)

        transition_map_1=hf.matrix_row_or_column_thresholding(transition_map_1,threshold=trunca_threshold[1],row_threshold=True)
        adata.uns['transition_map']=ssp.csr_matrix(transition_map_1)
        #adata.uns['transition_map_unExtended']=ssp.csr_matrix(transition_map)


    
    logg.hint("----Intraclone transition map----")

    #pdb.set_trace()
    demultiplexed_map_0=refine_Tmap_through_cospar_noSmooth(multiTime_cell_id_t1,multiTime_cell_id_t2,proportion,transition_map,
        X_clone,sparsity_threshold=intraclone_threshold,normalization_mode=normalization_mode)

    idx_t1=hf.converting_id_from_fullSpace_to_subSpace(clonal_cell_id_t1,Tmap_cell_id_t1)[0]
    idx_t2=hf.converting_id_from_fullSpace_to_subSpace(clonal_cell_id_t2,Tmap_cell_id_t2)[0]
    demultiplexed_map=np.zeros((len(Tmap_cell_id_t1),len(Tmap_cell_id_t2)))
    demultiplexed_map[idx_t1[:,np.newaxis],idx_t2]=demultiplexed_map_0.A
    adata.uns['intraclone_transition_map']=ssp.csr_matrix(demultiplexed_map)






def infer_intraclone_Tmap(adata,intraclone_threshold=0.05,normalization_mode=1):
    """
    Infer intra-clone transition map.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Should be prepared by :func:`.select_time_points`
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
    if 'transition_map' not in adata.uns.keys():
        logg.error("Please run ---- CS.tmap.infer_Tmap_from_multitime_clones ---- first")

    else:

        clone_annot=adata.obsm['X_clone']

        multiTime_cell_id_t1=[adata.uns['Tmap_cell_id_t1']]
        multiTime_cell_id_t2=[adata.uns['Tmap_cell_id_t2']]
        proportion=adata.uns['proportion']

        transition_map=adata.uns['transition_map']

        X_clone=clone_annot.copy()
        if not ssp.issparse(X_clone):
            X_clone=ssp.csr_matrix(X_clone)

        demultiplexed_map=refine_Tmap_through_cospar_noSmooth(multiTime_cell_id_t1,multiTime_cell_id_t2,proportion,transition_map,
            X_clone,sparsity_threshold=intraclone_threshold,normalization_mode=normalization_mode)

        adata.uns['intraclone_transition_map']=ssp.csr_matrix(demultiplexed_map)


# v0: avoid cells that are already selected. We tested, this is better than not avoiding...
def infer_Tmap_from_HighVar(adata,min_counts=3,min_cells=3,
    min_gene_vscore_pctl=85,smooth_array=[15,10,5],neighbor_N=20,
    sparsity_threshold=0.2,normalization_mode=1,use_full_Smatrix=True,
    trunca_threshold=[0.001,0.01],compute_new_Smatrix=True,max_iter_N=5,epsilon_converge=0.05,
    save_subset=True):
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

    #logg.info("HighVar-v0: avoid cells that have been selected")
    weight=1 # wehight of each gene. 

    cell_id_array_t1=adata.uns['Tmap_cell_id_t1']
    cell_id_array_t2=adata.uns['Tmap_cell_id_t2']
    real_clone_annot=adata.obsm['X_clone']

    time_info=np.array(adata.obs['time_info'])
    selected_time_points=[time_info[cell_id_array_t1][0],time_info[cell_id_array_t2][0]]


    
    #logg.info("----------------")
    logg.info('Step a: find the commonly shared highly variable genes------')
    adata_t1=sc.AnnData(adata.X[cell_id_array_t1]);
    adata_t2=sc.AnnData(adata.X[cell_id_array_t2]);

    ## use marker genes
    gene_list=adata.var_names

    verbose=logg._settings_verbosity_greater_or_equal_than(3)

    gene_idx_t1=hf.filter_genes(adata_t1.X, min_counts=min_counts, min_cells=min_cells, 
        min_vscore_pctl=min_gene_vscore_pctl, show_vscore_plot=verbose)
    if gene_idx_t1 is not None:
        highvar_genes_t1 = gene_list[gene_idx_t1]
    else:
        return None

    gene_idx_t2=hf.filter_genes(adata_t2.X, min_counts=min_counts, min_cells=min_cells, 
        min_vscore_pctl=min_gene_vscore_pctl, show_vscore_plot=verbose)
    if gene_idx_t2 is not None:
        highvar_genes_t2 = gene_list[gene_idx_t2]
    else:
        return None


    common_gene=list(set(highvar_genes_t1).intersection(highvar_genes_t2))
    
    logg.info(f"Highly varable gene number: {len(highvar_genes_t1)} (t1); {len(highvar_genes_t2)} (t2). Common set: {len(common_gene)}")

    #logg.info("----------------")
    logg.info('Step b: convert the shared highly variable genes into clonal info------')

    sel_marker_gene_list=common_gene.copy()
    clone_annot_gene=np.zeros((adata.shape[0],len(sel_marker_gene_list)))
    N_t1=len(cell_id_array_t1)
    N_t2=len(cell_id_array_t2)
    cumu_sel_idx_t1=np.zeros(N_t1,dtype=bool)
    cumu_sel_idx_t2=np.zeros(N_t2,dtype=bool)
    cell_fraction_per_gene=1/len(sel_marker_gene_list) # fraction of cells as clonally related by this gene
    for j,gene_id in enumerate(sel_marker_gene_list): 
        temp_t1=adata.obs_vector(gene_id)[cell_id_array_t1]
        temp_t1[cumu_sel_idx_t1]=0 # set selected cell id to have zero expression
        cutoff_t1=int(np.ceil(len(cell_id_array_t1)*cell_fraction_per_gene))
        sel_id_t1=np.argsort(temp_t1,kind='stable')[::-1][:cutoff_t1]
        clone_annot_gene[cell_id_array_t1[sel_id_t1],j]=weight
        cumu_sel_idx_t1[sel_id_t1]=True 
        #logg.info(f"Gene id {gene_id}, cell number at t1 is {sel_id_t1.shape[0]}, fraction at t1: {sel_id_t1.shape[0]/len(cell_id_array_t1)}")

        temp_t2=adata.obs_vector(gene_id)[cell_id_array_t2]
        temp_t2[cumu_sel_idx_t2]=0 # set selected cell id to have zero expression
        cutoff_t2=int(np.ceil(len(cell_id_array_t2)*cell_fraction_per_gene))
        sel_id_t2=np.argsort(temp_t2,kind='stable')[::-1][:cutoff_t2]
        clone_annot_gene[cell_id_array_t2[sel_id_t2],j]=weight
        cumu_sel_idx_t2[sel_id_t2]=True 
        #logg.info(f"Gene id {gene_id}, cell number at t2 is {sel_id_t2.shape[0]}, fraction at t2: {sel_id_t2.shape[0]/len(cell_id_array_t2)}")
        
        if (np.sum(~cumu_sel_idx_t1)==0) or (np.sum(~cumu_sel_idx_t2)==0):
            logg.info(f'Total used genes={j} (no cells left)')
            break

    #logg.info(f"Selected cell fraction: t1 -- {np.sum(cumu_sel_idx_t1)/len(cell_id_array_t1)}; t2 -- {np.sum(cumu_sel_idx_t2)/len(cell_id_array_t2)}")


    
    #logg.info("----------------")
    logg.info("Step c: compute the transition map based on clonal info from highly variable genes------")
    
    adata.obsm['X_clone']=ssp.csr_matrix(clone_annot_gene)
    adata.uns['multiTime_cell_id_t1']=[cell_id_array_t1]
    adata.uns['multiTime_cell_id_t2']=[cell_id_array_t2]
    adata.uns['proportion']=[1]
    # data_des_0=adata.uns['data_des'][-1]
    # data_des_orig=adata.uns['data_des'][0]
    # data_des_1=data_des_0+'_HighVar0' # to distinguish Similarity matrix for this step and the next step of CoSpar (use _HighVar0, instead of _HighVar1)
    # adata.uns['data_des']=[data_des_orig,data_des_1]

    infer_Tmap_from_multitime_clones_private(adata,smooth_array=smooth_array,neighbor_N=neighbor_N,sparsity_threshold=sparsity_threshold,
        normalization_mode=normalization_mode,save_subset=save_subset,use_full_Smatrix=use_full_Smatrix,
        trunca_threshold=trunca_threshold,compute_new_Smatrix=compute_new_Smatrix,max_iter_N=max_iter_N,epsilon_converge=epsilon_converge)

    adata.uns['HighVar_transition_map']=adata.uns['transition_map']
    adata.obsm['X_clone']=real_clone_annot # This entry has been changed previously. Note correct the clonal matrix
    #data_des_1=data_des_0+'_HighVar1' # to record which initialization is used
    #adata.uns['data_des']=[data_des_orig,data_des_1]

    if 'Smatrix' in adata.uns.keys():
        adata.uns.pop('Smatrix')



# this is the new version: v1, finally used
def infer_Tmap_from_optimal_transport(adata,OT_epsilon=0.02,OT_dis_KNN=5,
    OT_solver='duality_gap',OT_cost='SPD',compute_new=True,use_existing_KNN_graph=False):
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

    cell_id_array_t1=adata.uns['Tmap_cell_id_t1']
    cell_id_array_t2=adata.uns['Tmap_cell_id_t2']
    data_des=adata.uns['data_des'][-1]
    data_path=settings.data_path


    ######## apply optimal transport
    CustomOT_file_name=f'{data_path}/{data_des}_CustomOT_map_epsilon{OT_epsilon}_KNN{OT_dis_KNN}_OTcost{OT_cost}.npz'
    if os.path.exists(CustomOT_file_name) and (not compute_new):

        logg.info("Load pre-computed custom OT matrix")
        OT_transition_map=ssp.load_npz(CustomOT_file_name)

    else:

        ############ Compute shorted-path distance
        # use sklearn KNN graph construction method and select the connectivity option, not related to UMAP
        # use the mode 'distance' to obtain the shortest-path *distance*, rather than 'connectivity'
        if OT_cost=='SPD':
            SPD_file_name=f'{data_path}/{data_des}_ShortestPathDistanceMatrix_t0t1_KNN{OT_dis_KNN}.npy'
            if os.path.exists(SPD_file_name) and (not compute_new):

                logg.info("Load pre-computed shortest path distance matrix")
                OT_cost_matrix=np.load(SPD_file_name)

            else:

                logg.info("Compute new shortest path distance matrix")
                t=time.time()       
                #data_matrix=adata.obsm['X_pca']
                #ShortPath_dis=hf.compute_shortest_path_distance_from_raw_matrix(data_matrix,num_neighbors_target=OT_dis_KNN,mode='distance')
                ShortPath_dis=hf.compute_shortest_path_distance(adata,num_neighbors_target=OT_dis_KNN,mode='distances',method='umap',use_existing_KNN_graph=use_existing_KNN_graph)
                
                idx0=cell_id_array_t1
                idx1=cell_id_array_t2
                ShortPath_dis_t0t1=ShortPath_dis[idx0[:,np.newaxis],idx1]; 
                OT_cost_matrix=ShortPath_dis_t0t1/ShortPath_dis_t0t1.max()


                np.save(SPD_file_name,OT_cost_matrix) # This is not a sparse matrix at all. 


                logg.info(f"Finishing computing shortest-path distance, used time {time.time()-t}")
        else:
            t=time.time()
            pc_n=adata.obsm['X_pca'].shape[1]
            OT_cost_matrix=hf.compute_gene_exp_distance(adata,cell_id_array_t1,cell_id_array_t2,pc_n=pc_n)
            logg.info(f"Finishing computing gene expression distance, used time {time.time()-t}")  


        ##################        
        logg.info("Compute new custom OT matrix")

        t=time.time()
        mu1=np.ones(len(cell_id_array_t1));
        nu1=np.ones(len(cell_id_array_t2));
        input_mu=mu1 # initial distribution
        input_nu=nu1 # final distribution

        ######### We have tested that it is at least 3 times slower than WOT's built-in method, 
        #### although the results are the same
        # # This taks 170s for the subsampled hematopoietic data
#             logg.info("Use sinkhorn solver solver")
#             OT_transition_map=otb.sinkhorn_stabilized(input_mu,input_nu,ShortPath_dis_t0t1,OT_epsilon,numItermax=OT_max_iter,stopThr=OT_stopThr)

        #############
        OT_solver='duality_gap'
        logg.info(f"OT solver: {OT_solver}")
        if OT_solver == 'fixed_iters': # This takes 50s for the subsampled hematopoietic data. The result is the same.
            ot_config = {'C':OT_cost_matrix,'G':mu1, 'epsilon': OT_epsilon, 'lambda1': 1, 'lambda2': 50,
                          'epsilon0': 1, 'scaling_iter': 3000,'tau': 10000, 'inner_iter_max': 50, 'extra_iter': 1000}
            
            OT_transition_map=transport_stablev2(**ot_config)
            
        elif OT_solver == 'duality_gap': # This takes 30s for the subsampled hematopoietic data. The result is the same.
            ot_config = {'C':OT_cost_matrix,'G':mu1, 'epsilon': OT_epsilon, 'lambda1': 1, 'lambda2': 50,
                          'epsilon0': 1, 'tau': 10000, 'tolerance': 1e-08,
                          'max_iter': 1e7, 'batch_size': 5}
            
            OT_transition_map=optimal_transport_duality_gap(**ot_config)
            
        else:
            raise ValueError('Unknown solver')

        OT_transition_map=hf.matrix_row_or_column_thresholding(OT_transition_map,threshold=0.01)
        if not ssp.issparse(OT_transition_map): OT_transition_map=ssp.csr_matrix(OT_transition_map)
        ssp.save_npz(CustomOT_file_name,OT_transition_map)

        logg.info(f"Finishing computing optial transport map, used time {time.time()-t}")


    adata.uns['OT_transition_map']=OT_transition_map
    # data_des_0=adata.uns['data_des'][-1]
    # data_des_orig=adata.uns['data_des'][0]
    # data_des_1=data_des_0+'_OT' # to record which initialization is used
    # adata.uns['data_des']=[data_des_orig,data_des_1]


## This is just used for testing WOT
def infer_Tmap_from_optimal_transport_v0(adata,OT_epsilon=0.02,OT_dis_KNN=5,
    OT_solver='duality_gap',OT_cost='SPD',compute_new=True,use_existing_KNN_graph=False):
    """
    Test WOT

    Returns
    -------
    None. Results are stored at adata.uns['OT_transition_map'].
    """

    cell_id_array_t1=adata.uns['Tmap_cell_id_t1']
    cell_id_array_t2=adata.uns['Tmap_cell_id_t2']
    data_des=adata.uns['data_des'][0]
    data_path=settings.data_path

    logg.warn("-------------Using WOT----------------")
    logg.warn(f"epsilon={OT_epsilon}")
    import wot
    time_info=np.zeros(adata.shape[0])
    time_info[cell_id_array_t1]=1
    time_info[cell_id_array_t2]=2
    adata.obs['day']=time_info
    adata.obs['cell_growth_rate']=np.ones(len(time_info))
    ot_model = wot.ot.OTModel(adata,epsilon = OT_epsilon, lambda1 =1,lambda2 = 50)
    OT_transition_map = ot_model.compute_transport_map(1,2).X 

    adata.uns['OT_transition_map']=ssp.csr_matrix(OT_transition_map)



########### v1, with convergence test, 20210326
# We tested that, for clones of all different sizes, where np.argsort gives unique results, 
# this method reproduces the v01, v1 results, when use_fixed_clonesize_t1=True, and when change
# sort_clone=0,1,-1.
def refine_Tmap_through_joint_optimization(adata,initialized_map,
    smooth_array=[15,10,5],max_iter_N=[3,5],epsilon_converge=[0.05,0.05],
    CoSpar_KNN=20,normalization_mode=1,sparsity_threshold=0.2,
    use_full_Smatrix=True,trunca_threshold=[0.001,0.01],compute_new=True,
    use_fixed_clonesize_t1=False,sort_clone=1,save_subset=True):
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
    max_iter_N: `list`, optional (default: [3,5])
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

    #logg.info("Joint optimization that consider possibility of clonal overlap")

    cell_id_array_t1=adata.uns['Tmap_cell_id_t1']
    cell_id_array_t2=adata.uns['Tmap_cell_id_t2']
    data_des=adata.uns['data_des'][-1]
    data_path=settings.data_path
    X_clone=adata.obsm['X_clone']
    if not ssp.issparse(X_clone): X_clone=ssp.csr_matrix(X_clone) 

    time_info=np.array(adata.obs['time_info'])
    time_index_t1=time_info==(time_info[cell_id_array_t1[0]])
    time_index_t2=time_info==(time_info[cell_id_array_t2[0]])

    if not ssp.issparse(initialized_map):
        map_temp=ssp.csr_matrix(initialized_map)
    else:
        map_temp=initialized_map


    # a clone must has at least 2 cells, to be updated later. 
    valid_clone_id=np.nonzero(X_clone[cell_id_array_t2].sum(0).A.flatten()>0)[0]
    X_clone_temp=X_clone[:,valid_clone_id]
    clonal_cells_t2=np.sum(X_clone_temp[cell_id_array_t2].sum(1).flatten())

    logg.hint(f"original clone shape: {X_clone.shape}")
    logg.hint(f"After excluding zero-sized clones at t2: {X_clone_temp.shape}")


    flag=True # to check whether overlapping clones are found or not
    if use_fixed_clonesize_t1:
        logg.hint("Use fixed clone size at t1")

    ##### Partition cells into non-overlapping, combinatorial BC_id.  
    # ---------------------------------
    # find the combinatorial barcodes
    clone_idx=np.nonzero(X_clone_temp.A)
    dic=[[] for j in range(X_clone_temp.shape[0])] # a list of list
    for j in range(clone_idx[0].shape[0]):
        dic[clone_idx[0][j]].append(clone_idx[1][j])

    BC_id=[tuple(x) for x in dic] # a BC_id is a unique barcode combination, does not change the ordering of cells


    # --------------------
    # construct the new X_clone_temp matrix, and the clone_mapping
    unique_BC_id=list(set(BC_id))
    if () in unique_BC_id: # () is resulted from cells without any barcodes
        unique_BC_id.remove(())

    # construct a X_clone_newBC for the new BC_id
    # also record how the new BC_id is related to the old barcode

    X_clone_newBC=np.zeros((X_clone_temp.shape[0],len(unique_BC_id)))
    for i, BC_0 in enumerate(BC_id):
        for j, BC_1 in enumerate(unique_BC_id):
            if BC_1==BC_0:
                X_clone_newBC[i,j]=1 # does not change the ordering of cells

    clone_mapping=np.zeros((X_clone_temp.shape[1],X_clone_newBC.shape[1]))
    for j, BC_1 in enumerate(unique_BC_id):
        for k in BC_1:
            clone_mapping[k,j]=1

    X_clone_newBC=ssp.csr_matrix(X_clone_newBC)
    clone_mapping=ssp.csr_matrix(clone_mapping)
    # To recover the original X_clone_temp, use 'X_clone_newBC*(clone_mapping.T)'
    # howver, clone_mapping is not invertible. We cannot get from X_clone_temp to 
    # X_clone_newBC using matrix multiplification.



    ### select the early states using the grouped distribution of a clone
    ### clones are not overlapping, and all early states should be attached to clones at the end

    # we sort clones according to their sizes. The order of cells are not affected. So, it should not affect downstream analysis
    # small clones tend to be the ones that are barcoded/mutated later, while large clones tend to be early mutations...
    clone_size_t2_temp=X_clone_newBC[cell_id_array_t2].sum(0).A.flatten()


    if sort_clone==1:
        logg.hint("Sort clones by size (small to large)")

        sort_clone_id=np.argsort(clone_size_t2_temp,kind='stable')
        clone_size_t2=clone_size_t2_temp[sort_clone_id]
        X_clone_sort=X_clone_newBC[:,sort_clone_id]
        clone_mapping_sort=clone_mapping[:,sort_clone_id]

    elif sort_clone==-1:
        logg.hint("Sort clones by size (large to small)")

        sort_clone_id=np.argsort(clone_size_t2_temp,kind='stable')[::-1]
        clone_size_t2=clone_size_t2_temp[sort_clone_id]
        X_clone_sort=X_clone_newBC[:,sort_clone_id]
        clone_mapping_sort=clone_mapping[:,sort_clone_id]

    else:
        logg.hint("Do not order clones by size ")
        clone_size_t2=clone_size_t2_temp
        X_clone_sort=X_clone_newBC
        clone_mapping_sort=clone_mapping


    logg.hint("Infer the number of initial cells to extract for each clone in advance")
    clone_N1=X_clone_sort.shape[1]
    ave_clone_size_t1=int(np.ceil(len(cell_id_array_t1)/clone_N1));
    cum_cell_N=np.ceil(np.cumsum(clone_size_t2)*len(cell_id_array_t1)/clonal_cells_t2)
    cell_N_to_extract=np.zeros(len(cum_cell_N),dtype=int)
    if use_fixed_clonesize_t1:
        cell_N_to_extract += ave_clone_size_t1
    else:
        cell_N_to_extract[0]=cum_cell_N[0]
        cell_N_to_extract[1:]=np.diff(cum_cell_N)


    for x0 in range(max_iter_N[0]):
        logg.info(f"-----JointOpt Iteration {x0+1}: Infer initial clonal structure")

        # update initial state probability matrix based on the current map 
        initial_prob_matrix=(map_temp*X_clone_sort[cell_id_array_t2]).A # a initial probability matrix for t1 cells, shape (n_t1_cell,n_clone)
        

        ########## begin: update clones
        remaining_ids_t1=list(np.arange(len(cell_id_array_t1),dtype=int))

        X_clone_new=np.zeros(X_clone_sort.shape,dtype=bool)
        X_clone_new[cell_id_array_t2]=X_clone_sort[cell_id_array_t2].A.astype(bool) # update the whole t2 clones at once

        for j in range(clone_N1):
            if (j%100==0):
                #pdb.set_trace()
                logg.hint(f"Inferring early clonal states: current clone id {j}")



            # infer the earlier clonal states for each clone
            ### select the early states using the grouped distribution of a clone
            sorted_id_array=np.argsort(initial_prob_matrix[remaining_ids_t1,j],kind='stable')[::-1]

            sel_id_t1=sorted_id_array[:cell_N_to_extract[j]]
            temp_t1_idx=np.zeros(len(cell_id_array_t1),dtype=bool)
            temp_t1_idx[np.array(remaining_ids_t1)[sel_id_t1]]=True
            X_clone_new[cell_id_array_t1,j]=temp_t1_idx
            for kk in np.array(remaining_ids_t1)[sel_id_t1]:
                remaining_ids_t1.remove(kk)

            if (len(remaining_ids_t1)==0) and ((j+1)<clone_N1): 
                logg.hint(f'Early break; current clone id: {j+1}')
                break

        ########### end: update clones
        cell_id_array_t1_new=np.nonzero((X_clone_new.sum(1)>0) & (time_index_t1))[0]
        cell_id_array_t2_new=np.nonzero((X_clone_new.sum(1)>0) & (time_index_t2))[0]

        adata.obsm['X_clone']=ssp.csr_matrix(X_clone_new)*(clone_mapping_sort.T) # convert back to the original clone structure
        adata.uns['multiTime_cell_id_t1']=[cell_id_array_t1_new] # For CoSpar, clonally-related states
        adata.uns['multiTime_cell_id_t2']=[cell_id_array_t2_new]
        adata.uns['clonal_cell_id_t1']=cell_id_array_t1_new # for prepare the similarity matrix with same cell states
        adata.uns['clonal_cell_id_t2']=cell_id_array_t2_new
        adata.uns['proportion']=[1]

        logg.info(f"-----JointOpt Iteration {x0+1}: Update the transition map by CoSpar")
        infer_Tmap_from_multitime_clones_private(adata,smooth_array=smooth_array,neighbor_N=CoSpar_KNN,sparsity_threshold=sparsity_threshold,
            normalization_mode=normalization_mode,save_subset=save_subset,use_full_Smatrix=use_full_Smatrix,
            trunca_threshold=trunca_threshold,compute_new_Smatrix=compute_new,max_iter_N=max_iter_N[1],epsilon_converge=epsilon_converge[1])

        # update, for the next iteration
        if 'transition_map' in adata.uns.keys():
            
            # sample cell states to perform the accuracy test
            sample_N_x=50
            sample_N_y=100
            t0=time.time()
            cell_N_tot_x=map_temp.shape[0]
            if cell_N_tot_x<sample_N_x:
                sample_id_temp_x=np.arange(cell_N_tot_x)
            else:
                xx=np.arange(cell_N_tot_x)
                yy=list(np.nonzero(xx%3==0)[0])+list(np.nonzero(xx%3==1)[0])+list(np.nonzero(xx%3==2)[0])
                sample_id_temp_x=yy[:sample_N_x]

            cell_N_tot_y=map_temp.shape[1]
            if cell_N_tot_y<sample_N_y:
                sample_id_temp_y=np.arange(cell_N_tot_y)
            else:
                xx=np.arange(cell_N_tot_y)
                yy=list(np.nonzero(xx%3==0)[0])+list(np.nonzero(xx%3==1)[0])+list(np.nonzero(xx%3==2)[0])
                sample_id_temp_y=yy[:sample_N_y]


            if x0==0:
                X_map_0=map_temp[sample_id_temp_x,:][:,sample_id_temp_y].A
            else:
                X_map_0=X_map_1.copy()

            X_map_1=adata.uns['transition_map'][sample_id_temp_x,:][:,sample_id_temp_y].A

            verbose=logg._settings_verbosity_greater_or_equal_than(3)
            corr_X=np.diag(hf.corr2_coeff(X_map_0,X_map_1)).mean()
            if verbose:
                from matplotlib import pyplot as plt
                fig=plt.figure()
                ax=plt.subplot(1,1,1)
                ax.plot(X_map_0.flatten(),X_map_1.flatten(),'.r')
                ax.set_xlabel('$T_{ij}$: previous iteration')
                ax.set_ylabel('$T_{ij}$: current iteration')
                ax.set_title(f'Joint Opt., iter_N={x0+1}, R={int(100*corr_X)/100}')
                plt.show()
            else:
                #logg.info(f"Convergence (JointOpt, iter_N={x0+1}): corr(previous_T, current_T)={int(1000*corr_X)/1000}; cost time={time.time()-t0}")
                logg.info(f"Convergence (JointOpt, iter_N={x0+1}): corr(previous_T, current_T)={int(1000*corr_X)/1000}")

            if abs(1-corr_X)<epsilon_converge[0]: break

            map_temp=adata.uns['transition_map']
        else:
            logg.error("transition_map not updated in infer_Tmap_from_multitime_clones_private.")
            return None




def infer_Tmap_from_one_time_clones(adata_orig,initial_time_points=None,later_time_point=None,
    initialize_method='OT',OT_epsilon=0.02,OT_dis_KNN=5,OT_cost='SPD',
    HighVar_gene_pctl=85,padding_X_clone=False,normalization_mode=1,
    sparsity_threshold=0.2,CoSpar_KNN=20,use_full_Smatrix=True,smooth_array=[15,10,5],
    trunca_threshold=[0.001,0.01],compute_new=False,max_iter_N=[3,5],epsilon_converge=[0.05,0.05],
    use_fixed_clonesize_t1=False,sort_clone=1,save_subset=True,use_existing_KNN_graph=False):
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
    max_iter_N: `list`, optional (default: [3,5])
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

    t0=time.time()
    if 'data_des' not in adata_orig.uns.keys():
        adata_orig.uns['data_des']=['cospar']

    if type(later_time_point)==list: later_time_point=later_time_point[0]

    hf.check_available_clonal_info(adata_orig)
    clonal_time_points_0=adata_orig.uns['clonal_time_points']
    time_ordering=adata_orig.uns['time_ordering']

    if len(clonal_time_points_0)==0:
        logg.error('No clonal time points available for this dataset. Please run cs.tmap.infer_Tmap_from_state_info_alone.')
        return None

    # use the last clonal later time point
    if later_time_point is None:  
        sel_idx_temp=np.in1d(time_ordering,clonal_time_points_0)
        later_time_point=time_ordering[sel_idx_temp][-1]     
    else:
        if not (later_time_point in clonal_time_points_0):
            logg.warn(f"'later_time_point' do not contain clonal information. Please set later_time_point to be one of {adata_orig.uns['clonal_time_points']}")
            return None


    if initial_time_points is None:
        sel_id_temp=np.nonzero(np.in1d(time_ordering,[later_time_point]))[0][0]
        initial_time_points=time_ordering[:sel_id_temp]
    else:
        # re-order time points. This also gets rid of invalid time points
        sel_idx_temp=np.in1d(time_ordering,initial_time_points)
        if np.sum(sel_idx_temp)>0:
            initial_time_points=time_ordering[sel_idx_temp] 
        else:
            logg.error(f"The 'initial_time_points' are not valid. Please select from {time_ordering}")
            return None


    if save_subset:
        if not (np.all(np.diff(smooth_array)<=0) and np.all(np.array(smooth_array)%5==0)):
            logg.error("The smooth_array contains numbers not multiples of 5 or not in descending order.\n"
             "The correct form is like [20,15,10], or [10,10,10,5]."
              "You can also set save_subset=False to explore arbitrary smooth_array structure.")
            return None


    if initialize_method not in ['OT','HighVar']:
        logg.warn("initialize_method not among ['OT','HighVar']. Use initialize_method='OT'")
        initialize_method='OT'

    if OT_cost not in ['GED','SPD']:
        logg.warn("OT_cost not among ['GED','SPD']. Use OT_cost='SPD'")
        OT_cost='SPD'       


    sp_idx=np.zeros(adata_orig.shape[0],dtype=bool)
    time_info_orig=np.array(adata_orig.obs['time_info'])
    all_time_points=list(initial_time_points)+[later_time_point]
    label='t'
    for xx in all_time_points:
        id_array=np.nonzero(time_info_orig==xx)[0]
        sp_idx[id_array]=True
        label=label+'*'+str(xx)

    adata=adata_orig[sp_idx]
    # adata=sc.AnnData(adata_orig.X[sp_idx]);
    # adata.var_names=adata_orig.var_names
    # adata.obsm['X_pca']=adata_orig.obsm['X_pca'][sp_idx]
    # adata.obsm['X_emb']=adata_orig.obsm['X_emb'][sp_idx]
    # adata.obs['state_info']=pd.Categorical(adata_orig.obs['state_info'][sp_idx])
    # adata.obs['time_info']=pd.Categorical(adata_orig.obs['time_info'][sp_idx])
    # clone_annot_orig=adata_orig.obsm['X_clone'].copy()      
    # clone_annot=clone_annot_orig[sp_idx]
    # adata.obsm['X_clone']=ssp.csr_matrix(clone_annot)


    clone_annot_orig=adata_orig.obsm['X_clone'].copy()  
    data_des_orig=adata_orig.uns['data_des'][0]
    data_des_0=adata_orig.uns['data_des'][-1]
    data_des=data_des_0+f'_OneTimeClone_{label}'
    adata.uns['data_des']=[data_des_orig,data_des]

    time_info=np.array(adata.obs['time_info'])
    time_index_t2=time_info==later_time_point
    time_index_t1=~time_index_t2

    ## set cells without a clone ID to have a unique clone ID
    if padding_X_clone:
        logg.info("Generate a unique clonal label for each clonally unlabeled cell.")
        time_index_t2_orig=time_info_orig==later_time_point
        zero_clone_idx=clone_annot_orig[time_index_t2_orig].sum(1).A.flatten()==0
        clone_annot_t2_padding=np.diag(np.ones(np.sum(zero_clone_idx)))
        non_zero_clones_idx=clone_annot_orig[time_index_t2_orig].sum(0).A.flatten()>0
        M0=np.sum(non_zero_clones_idx)
        M1=clone_annot_t2_padding.shape[1]
        clone_annot_new=np.zeros((clone_annot_orig.shape[0],M0+M1))
        clone_annot_new[:,:M0]=clone_annot_orig[:,non_zero_clones_idx].A
        sp_id_t2=np.nonzero(time_index_t2_orig)[0]
        clone_annot_new[sp_id_t2[zero_clone_idx],M0:]=clone_annot_t2_padding         
    else:
        clone_annot_new=clone_annot_orig


    # remove clones without a cell at t2
    valid_clone_id=np.nonzero(clone_annot_new[time_info_orig==later_time_point].sum(0).A.flatten()>0)[0]
    X_clone_temp=clone_annot_new[:,valid_clone_id]
    adata_orig.obsm['X_clone']=ssp.csr_matrix(X_clone_temp)


    #### used for similarity matrix generation
    Tmap_cell_id_t1=np.nonzero(time_index_t1)[0]
    Tmap_cell_id_t2=np.nonzero(time_index_t2)[0]
    adata.uns['Tmap_cell_id_t1']=Tmap_cell_id_t1
    adata.uns['Tmap_cell_id_t2']=Tmap_cell_id_t2
    adata.uns['clonal_cell_id_t1']=Tmap_cell_id_t1
    adata.uns['clonal_cell_id_t2']=Tmap_cell_id_t2
    adata.uns['sp_idx']=sp_idx
    data_path=settings.data_path

    transition_map=np.zeros((len(Tmap_cell_id_t1),len(Tmap_cell_id_t2)))
    ini_transition_map=np.zeros((len(Tmap_cell_id_t1),len(Tmap_cell_id_t2)))
    X_clone_updated=adata_orig.obsm['X_clone'][sp_idx].A #this does not work well if there are empty clones to begin with

    logg.info("--------Infer transition map between initial time points and the later time one-------")
    for yy in initial_time_points:
        logg.info(f"--------Current initial time point: {yy}--------")

        adata_temp=infer_Tmap_from_one_time_clones_twoTime(adata_orig,selected_two_time_points=[yy,later_time_point],
            initialize_method=initialize_method,OT_epsilon=OT_epsilon,OT_dis_KNN=OT_dis_KNN,
            OT_cost=OT_cost,HighVar_gene_pctl=HighVar_gene_pctl,
            normalization_mode=normalization_mode,
            sparsity_threshold=sparsity_threshold,CoSpar_KNN=CoSpar_KNN,use_full_Smatrix=use_full_Smatrix,smooth_array=smooth_array,
            trunca_threshold=trunca_threshold,compute_new=compute_new,
            use_fixed_clonesize_t1=use_fixed_clonesize_t1,sort_clone=sort_clone,save_subset=save_subset,
            use_existing_KNN_graph=use_existing_KNN_graph,max_iter_N=max_iter_N,epsilon_converge=epsilon_converge)

        if (adata_temp is not None) and ('transition_map' in adata_temp.uns.keys()):
            temp_id_t1=np.nonzero(time_info==yy)[0]
            sp_id_t1=hf.converting_id_from_fullSpace_to_subSpace(temp_id_t1,Tmap_cell_id_t1)[0]
            
            transition_map_temp=adata_temp.uns['transition_map'].A
            transition_map[sp_id_t1,:]=transition_map_temp

            if initialize_method=='OT':
                transition_map_ini_temp=adata_temp.uns['OT_transition_map']
            else:
                transition_map_ini_temp=adata_temp.uns['HighVar_transition_map']

            ini_transition_map[sp_id_t1,:]=transition_map_ini_temp.A

            #Update clonal prediction. This does not work well if there are empty clones to begin with
            time_info_idx=np.array(adata_temp.obs['time_info'])==yy
            X_clone_updated[temp_id_t1,:]=adata_temp.obsm['X_clone'][time_info_idx].A
        else:
            return None


    adata.uns['transition_map']=ssp.csr_matrix(transition_map)
    adata.obsm['X_clone']=ssp.csr_matrix(X_clone_updated)
    
    if initialize_method=='OT':
        adata.uns['OT_transition_map']=ssp.csr_matrix(ini_transition_map)
    else:
        adata.uns['HighVar_transition_map']=ssp.csr_matrix(ini_transition_map)


    adata_orig.obsm['X_clone']=clone_annot_orig # reset to the original clonal matrix
    logg.info(f"-----------Total used time: {time.time()-t0} s ------------")
    return adata





# updated version: v1, we initialize the X_clone as isolated cells
def infer_Tmap_from_state_info_alone(adata_orig,initial_time_points=None,later_time_point=None,
    initialize_method='OT',OT_epsilon=0.02,OT_dis_KNN=5,OT_cost='SPD',
    HighVar_gene_pctl=85,normalization_mode=1,
    sparsity_threshold=0.2,CoSpar_KNN=20,use_full_Smatrix=True,smooth_array=[15,10,5],
    trunca_threshold=[0.001,0.01],compute_new=False,max_iter_N=[3,5],epsilon_converge=[0.05,0.05],
    use_fixed_clonesize_t1=False,sort_clone=1,save_subset=True,
    use_existing_KNN_graph=False):
    """
    Infer transition map from state information alone.

    After initializing the clonal matrix as such that each cell has a unique barcode,
    it runs :func:`.infer_Tmap_from_one_time_clones` to infer the transition map.  
    """

    if 'data_des' not in adata_orig.uns.keys():
        adata_orig.uns['data_des']=['cospar']
    logg.info('Step I: Generate pseudo clones where each cell has a unique barcode-----')
    X_clone_0=adata_orig.obsm['X_clone'].copy()
    #adata_orig.obsm['X_clone_old']=adata_orig.obsm['X_clone'].copy()
    X_clone=np.diag(np.ones(adata_orig.shape[0]))
    adata_orig.obsm['X_clone']=ssp.csr_matrix(X_clone)

    if type(later_time_point)==list: later_time_point=later_time_point[0]

    if 'time_ordering' not in adata_orig.uns.keys():
        hf.update_time_ordering(adata_orig)
    time_ordering=adata_orig.uns['time_ordering']

    # use the last time point
    if later_time_point is None:  
        later_time_point=time_ordering[-1]     


    if initial_time_points is None:
        # use the time points preceding the last one.
        sel_id_temp=np.nonzero(np.in1d(time_ordering,[later_time_point]))[0][0]
        initial_time_points=time_ordering[:sel_id_temp]
    else:
        # re-order time points. This also gets rid of invalid time points
        sel_idx_temp=np.in1d(time_ordering,initial_time_points)
        if np.sum(sel_idx_temp)>0:
            initial_time_points=time_ordering[sel_idx_temp] 
        else:
            logg.error(f"The 'initial_time_points' are not valid. Please select from {time_ordering}")
            return None

    logg.info('Step II: Perform joint optimization-----')
    adata=infer_Tmap_from_one_time_clones(adata_orig,initial_time_points=initial_time_points,
        later_time_point=later_time_point,initialize_method=initialize_method,OT_epsilon=OT_epsilon,
        OT_dis_KNN=OT_dis_KNN,
        OT_cost=OT_cost,HighVar_gene_pctl=HighVar_gene_pctl,
        normalization_mode=normalization_mode,sparsity_threshold=sparsity_threshold,
        CoSpar_KNN=CoSpar_KNN,use_full_Smatrix=use_full_Smatrix,smooth_array=smooth_array,
        trunca_threshold=trunca_threshold,compute_new=compute_new,max_iter_N=max_iter_N,epsilon_converge=epsilon_converge,
        use_fixed_clonesize_t1=use_fixed_clonesize_t1,sort_clone=sort_clone,save_subset=save_subset,
        use_existing_KNN_graph=use_existing_KNN_graph)

    if adata is not None:
        # only restore the original X_clone information to adata_orig. adata will carry the new structure
        adata_orig.obsm['X_clone']=X_clone_0
        #sp_idx=adata.uns['sp_idx']
        #adata.obsm['X_clone']=ssp.csr_matrix(X_clone_0[sp_idx])


        # update the data_des tag
        time_info_orig=np.array(adata_orig.obs['time_info'])
        all_time_points=list(initial_time_points)+[later_time_point]
        label='t'
        for xx in all_time_points:
            id_array=np.nonzero(time_info_orig==xx)[0]
            label=label+'*'+str(xx)
        
        data_des_orig=adata_orig.uns['data_des'][0]
        data_des_0=adata_orig.uns['data_des'][-1]
        data_des=data_des_0+f'_StateInfo_{label}'
        adata.uns['data_des']=[data_des_orig,data_des]

        return adata
    else:
        return None



def infer_Tmap_from_one_time_clones_twoTime(adata_orig,selected_two_time_points=['1','2'],
    initialize_method='OT',OT_epsilon=0.02,OT_dis_KNN=5,OT_cost='SPD',HighVar_gene_pctl=80,
    normalization_mode=1,sparsity_threshold=0.2,CoSpar_KNN=20,
    use_full_Smatrix=True,smooth_array=[15,10,5],max_iter_N=[3,5],epsilon_converge=[0.05,0.05],
    trunca_threshold=[0.001,0.01],compute_new=True,use_fixed_clonesize_t1=False,
    sort_clone=1,save_subset=True,joint_optimization=True,use_existing_KNN_graph=False):
    """
    Infer transition map from clones with a single time point

    It is the same as :func:`.infer_Tmap_from_one_time_clones`, except that
    it assumes that the input adata_orig has only two time points. 

    joint_optimization: `bool`, optional (default: True). 
    """

    time_info_orig=np.array(adata_orig.obs['time_info'])
    sort_time_point=np.sort(list(set(time_info_orig)))
    N_valid_time=np.sum(np.in1d(sort_time_point,selected_two_time_points))
    if (N_valid_time!=2): 
        logg.error(f"Must select only two time points among the list {sort_time_point}")
        #The second time point in this list (not necessarily later time point) is assumed to have clonal data.")
    else:
        ####################################
        
        logg.info("Step 0: Pre-processing and sub-sampling cells-------")
        # select cells from the two time points, and sub-sampling, create the new adata object with these cell states
        sp_idx=(time_info_orig==selected_two_time_points[0]) | (time_info_orig==selected_two_time_points[1])
  
        adata=adata_orig[sp_idx]
        # adata=sc.AnnData(adata_orig.X[sp_idx]);
        # adata.var_names=adata_orig.var_names
        # adata.obsm['X_pca']=adata_orig.obsm['X_pca'][sp_idx]
        # adata.obsm['X_emb']=adata_orig.obsm['X_emb'][sp_idx]
        # adata.obs['state_info']=pd.Categorical(adata_orig.obs['state_info'][sp_idx])
        # adata.obs['time_info']=pd.Categorical(adata_orig.obs['time_info'][sp_idx])
        # clone_annot_orig=adata_orig.obsm['X_clone']        
        # barcode_id=np.nonzero(clone_annot_orig[sp_idx].A.sum(0).flatten()>0)[0]
        # clone_annot=clone_annot_orig[sp_idx][:,barcode_id]
        # adata.obsm['X_clone']=clone_annot

        
        data_des_0=adata_orig.uns['data_des'][-1]
        data_des_orig=adata_orig.uns['data_des'][0]
        data_des=data_des_0+f'_t*{selected_two_time_points[0]}*{selected_two_time_points[1]}'
        adata.uns['data_des']=[data_des_orig,data_des]
        
        time_info=np.array(adata.obs['time_info'])
        time_index_t1=time_info==selected_two_time_points[0]
        time_index_t2=time_info==selected_two_time_points[1]

        #### used for similarity matrix generation
        Tmap_cell_id_t1=np.nonzero(time_index_t1)[0]
        Tmap_cell_id_t2=np.nonzero(time_index_t2)[0]
        adata.uns['Tmap_cell_id_t1']=Tmap_cell_id_t1
        adata.uns['Tmap_cell_id_t2']=Tmap_cell_id_t2
        adata.uns['clonal_cell_id_t1']=Tmap_cell_id_t1
        adata.uns['clonal_cell_id_t2']=Tmap_cell_id_t2
        adata.uns['sp_idx']=sp_idx
        data_path=settings.data_path


        cell_id_array_t1=Tmap_cell_id_t1
        cell_id_array_t2=Tmap_cell_id_t2

        ###############################
        # prepare the similarity matrix with all state info, all subsequent similarity will be down-sampled from this one.
        if use_full_Smatrix and (joint_optimization or (initialize_method!='OT')): 

            temp_str='0'+str(trunca_threshold[0])[2:]
            round_of_smooth=np.max(smooth_array)
            data_des=adata_orig.uns['data_des'][0]
            similarity_file_name=f'{data_path}/{data_des}_Similarity_matrix_with_all_cell_states_kNN{CoSpar_KNN}_Truncate{temp_str}'
            if not (os.path.exists(similarity_file_name+f'_SM{round_of_smooth}.npz') and (not compute_new)):
                similarity_matrix_full=generate_similarity_matrix(adata_orig,similarity_file_name,round_of_smooth=round_of_smooth,
                            neighbor_N=CoSpar_KNN,truncation_threshold=trunca_threshold[0],save_subset=save_subset,compute_new_Smatrix=compute_new)

        

        if initialize_method=='OT':
            
            #logg.info("----------------")
            logg.info("Step 1: Use OT method for initialization-------")

            infer_Tmap_from_optimal_transport(adata,OT_epsilon=OT_epsilon,OT_cost=OT_cost,OT_dis_KNN=OT_dis_KNN,
                compute_new=compute_new,use_existing_KNN_graph=use_existing_KNN_graph)

            if 'OT_transition_map' in adata.uns.keys():
                OT_transition_map=adata.uns['OT_transition_map']
                initialized_map=OT_transition_map
            else:
                logg.error(f"Computation failed")
                return None   
        else:
            
            #logg.info("----------------")
            logg.info("Step 1: Use the HighVar method for initialization-------")

            t=time.time()
            infer_Tmap_from_HighVar(adata,min_counts=3,min_cells=3,min_gene_vscore_pctl=HighVar_gene_pctl,sparsity_threshold=sparsity_threshold,neighbor_N=CoSpar_KNN,
                normalization_mode=normalization_mode,use_full_Smatrix=use_full_Smatrix,smooth_array=smooth_array,trunca_threshold=trunca_threshold,
                compute_new_Smatrix=compute_new,max_iter_N=max_iter_N[1],epsilon_converge=epsilon_converge[1])

            if 'HighVar_transition_map' in adata.uns.keys():
                HighVar_transition_map=adata.uns['HighVar_transition_map']
                initialized_map=HighVar_transition_map
                logg.info(f"Finishing initialization using HighVar, used time {time.time()-t}")
            else:
                logg.error(f"Computation failed")
                return None

        if joint_optimization:
            ########### Jointly optimize the transition map and the initial clonal states
            if selected_two_time_points[1] in adata_orig.uns['clonal_time_points']:
            
                #logg.info("----------------")
                logg.info("Step 2: Jointly optimize the transition map and the initial clonal states-------")

                t=time.time()

                refine_Tmap_through_joint_optimization(adata,initialized_map,normalization_mode=normalization_mode,
                    sparsity_threshold=sparsity_threshold,
                    CoSpar_KNN=CoSpar_KNN,use_full_Smatrix=use_full_Smatrix,smooth_array=smooth_array,
                    max_iter_N=max_iter_N,epsilon_converge=epsilon_converge,
                    trunca_threshold=trunca_threshold,compute_new=compute_new,
                    use_fixed_clonesize_t1=use_fixed_clonesize_t1,sort_clone=sort_clone,save_subset=save_subset)


                
                logg.info(f"Finishing Joint Optimization, used time {time.time()-t}")
            else:
                logg.warn("No clonal information available. Skip the joint optimization of clone and scRNAseq data")

        if 'Smatrix' in adata.uns.keys():
            adata.uns.pop('Smatrix')
        return adata


def infer_Tmap_from_clonal_info_alone_private(adata_orig,method='naive',clonal_time_points=None,
    selected_fates=None):
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

    adata_1=select_time_points(adata_orig,time_point=clonal_time_points,extend_Tmap_space=True)
    if method not in ['naive','weinreb']:
        logg.warn("method not in ['naive','weinreb']; set to be 'weinreb'")
        method='weinreb'

    cell_id_t2_all=adata_1.uns['Tmap_cell_id_t2']
    cell_id_t1_all=adata_1.uns['Tmap_cell_id_t1']            

    T_map=np.zeros((len(cell_id_t1_all),len(cell_id_t2_all)))
    clone_annot=adata_1.obsm['X_clone']

    N_points=len(adata_1.uns['multiTime_cell_id_t1'])
    for k in range(N_points):

        cell_id_t1_temp=adata_1.uns['multiTime_cell_id_t1'][k]
        cell_id_t2_temp=adata_1.uns['multiTime_cell_id_t2'][k]
        if method=='naive':
            logg.info("Use all clones (naive method)")
            T_map_temp=clone_annot[cell_id_t1_temp]*clone_annot[cell_id_t2_temp].T

        else:
            logg.info("Use only uni-potent clones (weinreb et al., 2020)")
            state_annote=np.array(adata_1.obs['state_info'])
            if selected_fates==None:
                selected_fates=list(set(state_annote))
            potential_vector_clone, fate_entropy_clone=hf.compute_state_potential(clone_annot[cell_id_t2_temp].T,state_annote[cell_id_t2_temp],selected_fates,fate_count=True)

            sel_unipotent_clone_id=np.array(list(set(np.nonzero(fate_entropy_clone==1)[0])))
            clone_annot_unipotent=clone_annot[:,sel_unipotent_clone_id]
            T_map_temp=clone_annot_unipotent[cell_id_t1_temp]*clone_annot_unipotent[cell_id_t2_temp].T
            logg.info(f"Used uni-potent clone fraction {len(sel_unipotent_clone_id)/clone_annot.shape[1]}")

        idx_t1=np.nonzero(np.in1d(cell_id_t1_all,cell_id_t1_temp))[0]
        idx_t2=np.nonzero(np.in1d(cell_id_t2_all,cell_id_t2_temp))[0]
        idx_t1_temp=np.nonzero(np.in1d(cell_id_t1_temp,cell_id_t1_all))[0]
        idx_t2_temp=np.nonzero(np.in1d(cell_id_t2_temp,cell_id_t2_all))[0]
        T_map[idx_t1[:,np.newaxis],idx_t2]=T_map_temp[idx_t1_temp][:,idx_t2_temp].A
        
    T_map=T_map.astype(int)
    adata_1.uns['clonal_transition_map']=ssp.csr_matrix(T_map)
    return adata_1


# the v2 version, it is the same format as infer_Tmap_from_multiTime_clones.
# We return a new adata object that will throw away existing annotations in uns. 
def infer_Tmap_from_clonal_info_alone(adata_orig,method='naive',clonal_time_points=None,
    later_time_point=None,selected_fates=None):
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

    if 'data_des' not in adata_orig.uns.keys():
        adata_orig.uns['data_des']=['cospar']
    hf.check_available_clonal_info(adata_orig)
    clonal_time_points_0=np.array(adata_orig.uns['clonal_time_points'])
    if len(clonal_time_points_0)<2:
        logg.error("There are no multi-time clones. Abort the inference.")

    else:
        if clonal_time_points is None:
            clonal_time_points=clonal_time_points_0

        if (later_time_point is not None) and (later_time_point not in clonal_time_points_0):
            logg.error(f"later_time_point is not all among {clonal_time_points_0}. Computation aborted!")
            return None 

        if (later_time_point is not None):
            clonal_time_points=list(clonal_time_points)+[later_time_point]
            clonal_time_points=list(set(clonal_time_points))

        N_valid_time=np.sum(np.in1d(clonal_time_points_0,clonal_time_points))
        if (N_valid_time!=len(clonal_time_points)) or (N_valid_time<2): 
            logg.error(f"Selected time points are not all among {clonal_time_points_0}, or less than 2 time points are selected. Computation aborted!")
            return None

        # adjust the order of time points
        time_ordering=adata_orig.uns['time_ordering']
        sel_idx_temp=np.in1d(time_ordering,clonal_time_points)
        clonal_time_points=time_ordering[sel_idx_temp]

        if later_time_point is None:
            logg.info("Infer transition map between neighboring time points.")              
            adata=infer_Tmap_from_clonal_info_alone_private(adata_orig,method=method,clonal_time_points=clonal_time_points,
                selected_fates=selected_fates)

            return adata
        else:
            logg.info(f"Infer transition map between initial time points and the later time point.")   
            # compute transition map between initial time points and the later time point
            sel_id=np.nonzero(np.in1d(clonal_time_points,later_time_point))[0][0]
            initial_time_points=clonal_time_points[:sel_id]

            time_info_orig=np.array(adata_orig.obs['time_info'])
            sp_idx=np.zeros(adata_orig.shape[0],dtype=bool)
            all_time_points=list(initial_time_points)+[later_time_point]
            label='t'
            for xx in all_time_points:
                id_array=np.nonzero(time_info_orig==xx)[0]
                sp_idx[id_array]=True
                label=label+'*'+str(xx)

            adata=adata_orig[sp_idx]
            data_des_orig=adata_orig.uns['data_des'][0]
            data_des_0=adata_orig.uns['data_des'][-1]
            data_des=data_des_0+f'_ClonalMap_Later_{label}'
            adata_orig.uns['data_des']=[data_des_orig,data_des]

            time_info=np.array(adata_orig.obs['time_info'])
            time_index_t2=time_info==later_time_point
            time_index_t1=~time_index_t2

            #### used for similarity matrix generation
            Tmap_cell_id_t1=np.nonzero(time_index_t1)[0]
            Tmap_cell_id_t2=np.nonzero(time_index_t2)[0]
            adata.uns['Tmap_cell_id_t1']=Tmap_cell_id_t1
            adata.uns['Tmap_cell_id_t2']=Tmap_cell_id_t2
            adata.uns['clonal_cell_id_t1']=Tmap_cell_id_t1
            adata.uns['clonal_cell_id_t2']=Tmap_cell_id_t2
            adata.uns['sp_idx']=sp_idx
            data_path=settings.data_path

            transition_map=np.zeros((len(Tmap_cell_id_t1),len(Tmap_cell_id_t2)))

            #logg.info("------Infer transition map between initial time points and the later time one-------")
            for yy in initial_time_points:
                logg.info(f"--------Current initial time point: {yy}--------")
                
                # by default, we extend the state space to all cells at the given time point.
                adata_temp=infer_Tmap_from_clonal_info_alone_private(adata_orig,method=method,clonal_time_points=[yy,later_time_point],
                    selected_fates=selected_fates)

                temp_id_t1=np.nonzero(time_info==yy)[0]
                sp_id_t1=hf.converting_id_from_fullSpace_to_subSpace(temp_id_t1,Tmap_cell_id_t1)[0]
                
                # by default, we extend the state space to all cells at the given time point.
                # so we only need to care about t1. 
                transition_map[sp_id_t1,:]=adata_temp.uns['clonal_transition_map'].A

            adata.uns['clonal_transition_map']=ssp.csr_matrix(transition_map)

            return adata


