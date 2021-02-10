# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd
import time
import scanpy as sc
import scipy.sparse as ssp 

import cospar.help_functions as hf
import cospar.plotting as CSpl
from .optimal_transport import *
from .. import settings
from .. import logging as logg


####################

# Constructing the similarity matrix (similarity matrix)

####################


def generate_similarity_matrix(adata,file_name,round_of_smooth=10,neighbor_N=20,beta=0.1,truncation_threshold=0.001,save_subset=True,compute_new_Smatrix=False):
    """
    Generate similarity matrix (Smatrix) through graph diffusion

    It generates the similarity matrix via iteratively graph diffusion. 
    Similarity matrix from each round of diffusion will be saved, after truncation 
    to promote sparsity and save space. If save_subset is activated, only save 
    Smatrix for smooth round [5,10,15,...]. If a Smatrix is pre-computed, 
    it will be loaded directly if compute_new_Smatrix=Flase.  

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    file_name: str 
        file name to load pre-computed similarity matrix or save the newly 
        computed similarity matrix 
    round_of_smooth: `int`, optional (default: 10)
        The rounds of graph diffusion.
    neighbor_N: `int`, optional (default: 20)
        Neighber number for constructing the KNN graph, using the UMAP method. 
    beta: `float`, option (default: 0.1)
        Probability to stay at origin in a unit diffusion step, in the range [0,1]
    truncation_threshold: `float`, optional (default: 0.001)
        At each iteration, truncate the similarity matrix (the similarity) using 
        truncation_threshold. This promotes the sparsity of the matrix, 
        thus the speed of computation. We set the truncation threshold to be small, 
        to guarantee accracy.
    save_subset: `bool`, optional (default: True)
        If true, save only Smatrix at smooth round [5,10,15,...]
        Else, save Smatrix at each round. 
    compute_new_Smatrix: `bool`, optional (default: False)
        If true, compute new Smatrix, even if there is pre-computed Smatrix with the 
        same parameterization.  

    Returns
    -------
        similarity_matrix: `sp.spmatrix` 
    """

    if os.path.exists(file_name+f'_SM{round_of_smooth}.npz') and (not compute_new_Smatrix):
        
        logg.info("Compute similarity matrix: load existing data")
        similarity_matrix=ssp.load_npz(file_name+f'_SM{round_of_smooth}.npz')
    else: # compute now
        
        logg.info(f"Compute similarity matrix: computing new; beta={beta}")

        # add a step to compute PCA in case this is not computed 

        # here, we assume that adata already has pre-computed PCA
        sc.pp.neighbors(adata, n_neighbors=neighbor_N)

        ## compute the similarity matrix (smooth matrix)
        
        #nrow = adata.shape[0]
        #initial_clones = ssp.lil_matrix((nrow, nrow))
        #initial_clones.setdiag(np.ones(nrow))
        #similarity_matrix=hf.get_smooth_values_SW(initial_clones, adata_sp.uns['neighbors']['connectivities'], beta=0, n_rounds=round_of_smooth)
        #similarity_matrix=get_smooth_values_sparseMatrixForm(initial_clones, adata.uns['neighbors']['connectivities'], beta=0, n_rounds=round_of_smooth)
        # this similarity_matrix is column-normalized, our B here

        
        #adjacency_matrix=adata.uns['neighbors']['connectivities'];
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
                    
                    logg.info("Save the matrix~~~")
                    ssp.save_npz(file_name+f'_SM{SM}.npz',similarity_matrix)
            else: # save all
                
                logg.info("Save the matrix")
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



def select_time_points(adata_orig,time_point=['day_1','day_2'],use_all_cells=False):
    """
    Select barcoded cells at given time points for Tmap inference

    Select cells at given time points, and prepare the right data structure 
    for running core cospar function to infer the Tmap. 
    
    Parameters
    ----------
    adata_orig: original :class:`~anndata.AnnData` object
    time_point: `list` optional (default: ['day_1','day_2'])
        Require at least two time points, arranged in ascending order.
    use_all_cells: `bool` optional (default: `False`)
        If true, all cells at selected time points will be used for computing Tmap
        If false, only cells belonging to multi-time clones will be used for computing Tmap.
        The latter case usually speed up the computation, which is recommended.  

    Returns
    -------
    Subsampled :class:`~anndata.AnnData` object
    """
    
    #x_emb_orig=adata_orig.obsm['X_umap'][:,0]
    #y_emb_orig=adata_orig.obsm['X_umap'][:,1]
    time_info_orig=np.array(adata_orig.obs['time_info'])
    clone_annot_orig=adata_orig.obsm['X_clone']
    if len(time_point)==0: # use all clonally labelled cell states 
        time_point=np.sort(list(set(time_info_orig)))

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

        
        logg.info("Valid clone number 'FOR' post selection",valid_clone_N_FOR)
        #logg.info("Valid clone number 'BACK' post selection",valid_clone_N_BACK)


        ###################### select initial and later cell states

        if use_all_cells:
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

        adata=sc.AnnData(adata_orig.X[sp_idx]);
        adata.var_names=adata_orig.var_names
        adata.obsm['X_pca']=adata_orig.obsm['X_pca'][sp_idx]
        adata.obsm['X_umap']=adata_orig.obsm['X_umap'][sp_idx]
        adata.obs['state_info']=pd.Categorical(adata_orig.obs['state_info'][sp_idx])
        adata.obs['time_info']=pd.Categorical(adata_orig.obs['time_info'][sp_idx])
        
        

        adata.obsm['X_clone']=clone_annot
        adata.uns['clonal_cell_id_t1']=clonal_cell_id_t1
        adata.uns['clonal_cell_id_t2']=clonal_cell_id_t2
        adata.uns['Tmap_cell_id_t1']=Tmap_cell_id_t1
        adata.uns['Tmap_cell_id_t2']=Tmap_cell_id_t2
        adata.uns['multiTime_cell_id_t1']=Clonal_cell_ID_FOR_t_new
        adata.uns['multiTime_cell_id_t2']=Clonal_cell_ID_BACK_t_new
        adata.uns['proportion']=np.ones(len(time_point)-1)
        adata.uns['sp_idx']=sp_idx

        data_des_0=adata_orig.uns['data_des'][0]
        time_label='t'
        for x in time_point:
            time_label=time_label+f'*{x}'

        data_des=data_des_0+f'_TwoTimeClone_{time_label}'
        adata.uns['data_des']=[data_des]

        if logg._settings_verbosity_greater_or_equal_than(2):
            N_cell,N_clone=clone_annot.shape;
            logg.info(f"Cell number={N_cell}, Clone number={N_clone}")
            x_emb=adata.obsm['X_umap'][:,0]
            y_emb=adata.obsm['X_umap'][:,1]
            CSpl.embedding_plot(x_emb,y_emb,-x_emb)

        return adata        



####################

# CoSpar: two-time points

####################


def refine_Tmap_through_cospar(MultiTime_cell_id_array_t1,MultiTime_cell_id_array_t2,
    proportion,transition_map,X_clone,initial_similarity,final_similarity,
    noise_threshold=0.1,normalization_mode=1):
    """
    This performs one iteration of coherent sparsity optimization

    This is our core algorithm for coherent sparsity optimization for multi-time
    clones. It upates a map by considering clones spanning multiple time points.

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
    noise_threshold: `float`, optional (default: 0.1)
        noise threshold to remove noises in the updated transition map,
        in the range [0,1]
    normalization_mode: `int`, optional (default: 1)
        Method for normalization. Choice: [0,1]
        0, single-cell normalization
        1, Clone normalization

    Returns
    -------
    smoothed_new_transition_map: `np.array`
    un_SM_transition_map: `np.array`
    """

    resol=10**(-10)

    transition_map=hf.matrix_row_or_column_thresholding(transition_map,noise_threshold,row_threshold=True)

    
    if normalization_mode==0: logg.info("Single-cell normalization")
    if normalization_mode==1: logg.info("Clone normalization")

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
                    logg.hint("marker gene weight",weight_factor)

                #Use the add mode, add up contributions from each clone
                new_coupling_matrix[id_1[:,np.newaxis],id_2]=new_coupling_matrix[id_1[:,np.newaxis],id_2]+proportion[j]*prob*weight_factor 

        ## update offset
        offset_N1=offset_N1+len(cell_id_array_t1)
        offset_N2=offset_N2+len(cell_id_array_t2)
            

    ## rescale
    new_coupling_matrix=new_coupling_matrix/(new_coupling_matrix.A.max())

    ## convert to sparse matrix form
    new_coupling_matrix=new_coupling_matrix.tocsr()

    
    logg.info("Start to smooth the refined clonal map")
    t=time.time()
    temp=new_coupling_matrix*final_similarity
    
    logg.info("Phase I: time elapsed -- ", time.time()-t)
    smoothed_new_transition_map=initial_similarity.dot(temp)
    
    logg.info("Phase II: time elapsed -- ", time.time()-t)

    # both return are numpy array
    un_SM_transition_map=new_coupling_matrix.A
    return smoothed_new_transition_map, un_SM_transition_map




def refine_Tmap_through_cospar_noSmooth(MultiTime_cell_id_array_t1,
    MultiTime_cell_id_array_t2,proportion,transition_map,
    X_clone,noise_threshold=0.1,normalization_mode=1):
    """
    This performs one iteration of coherent sparsity optimization

    This is the same as 'refine_Tmap_through_cospar', except that 
    there is no smoothing afterwards for demultiplexing.

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
    noise_threshold: `float`, optional (default: 0.1)
        noise threshold to remove noises in the updated transition map,
        in the range [0,1]
    normalization_mode: `int`, optional (default: 1)
        Method for normalization. Choice: [0,1]
        0, single-cell normalization
        1, Clone normalization

    Returns
    -------
    un_SM_transition_map: `np.array`
    """

    if not isinstance(X_clone[0,0], bool):
        X_clone=X_clone.astype(bool)

    resol=10**(-10)
    
    if normalization_mode==0: logg.info("Single-cell normalization")
    if normalization_mode==1: logg.info("Clone normalization")

    transition_map=hf.matrix_row_or_column_thresholding(transition_map,noise_threshold,row_threshold=True)
    
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

def infer_Tmap_from_multitime_clones(adata_orig,selected_clonal_time_points,
    smooth_array=[15,10,5],CoSpar_KNN=20,noise_threshold=0.1,demulti_threshold=0.05,
    normalization_mode=1,use_all_cells=False,save_subset=True,use_full_Smatrix=False,
    trunca_threshold=0.001,compute_new=False):
    """
    Infer Tmap for clonal data with multiple time points.

    It propares adata object for cells of targeted time points by 
    :func:`.select_time_points`, generate the similarity matrix 
    via :func:`.generate_similarity_matrix`, and iterately calls 
    the core function :func:`.refine_Tmap_through_cospar` to update 
    the transition map. 

    Parameters
    ----------
    adata_orig: :class:`~anndata.AnnData` object
        Should be prepared from our anadata initialization.
    selected_clonal_time_points: `list` of `str`
        List of time points to be included for analysis. 
        We assume that each selected time point has clonal measurement. 
        It should be in ascending order: 'day_1','day_2'.... 
    smooth_array: `list`, optional (default: [15,10,5])
        List of smooth rounds at each iteration. 
        The n-th entry determines the smooth round for the Smatrix 
        at the n-th iteration. Its length determins the number of
        iteration.  
    CoSpar_KNN: `int`, optional (default: 20)
        the number of neighbors for KNN graph used for computing the 
        similarity matrix.
    trunca_threshold: `float`, optional (default: 0.001)
        We set entries to zero in the computed similarity matrix that 
        are smaller than this threshold. This is to promote the Smatrix 
        sparsity, which leads to faster computation, and smaller file size. 
        This threshld should be small, but not too small. 
    noise_threshold: `float`, optional (default: 0.1)
        noise threshold to remove noises in the updated transition map,
        in the range [0,1]
    demulti_threshold: `float`, optional (default: 0.05)
        noise threshold to remove noises in demultiplexed (un-smoothed) map,
        in the range [0,1]
    normalization_mode: `int`, optional (default: 1)
        Method for normalization. Choice: [0,1]
        0, single-cell normalization
        1, Clone normalization
    use_all_cells: `bool` optional (default: `False`)
        If true, all cells at selected time points will be used for computing 
        Tmap. If false, only cells belonging to multi-time clones will be used 
        for computing Tmap. The latter case usually speed up the computation, 
        which is recommended.  
    save_subset: `bool`, optional (default: True)
        If true, save only Smatrix at smooth round [5,10,15,...].
        Else, save Smatrix at each round. 
    use_full_Smatrix: `bool`, optional (default: False)
        use the Smatrix as defined by all cells, whether they are clonally 
        barcoded or not. We sub-sample cell states relevant for downstream 
        analysis from this full Smatrix. This may refine the Smatrix. 
        But will also increase the computation time significantly.
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

    for xx in selected_clonal_time_points:
        if xx not in adata_orig.uns['clonal_time_points']:
            logg.error(f"'selected_clonal_time_points' contain time points without clonal information. Please set clonal_time_point to be at least two of {adata_orig.uns['clonal_time_points']}. If there is only one clonal time point, plesae run ----cospar.tmap.infer_Tmap_from_one_time_clones----")
            return adata_orig


    
    logg.info("-------Step 1: Select time points---------")
    data_path=settings.data_path
    adata=select_time_points(adata_orig,time_point=selected_clonal_time_points,use_all_cells=use_all_cells)

    
    logg.info("-------Step 2: Compute the full Similarity matrix if necessary---------")

    if use_full_Smatrix: # prepare the similarity matrix with all state info, all subsequent similarity will be down-sampled from this one.

        temp_str='0'+str(trunca_threshold)[2:]
        round_of_smooth=np.max(smooth_array)

        similarity_file_name=f'{data_path}/Similarity_matrix_with_all_cell_states_kNN{CoSpar_KNN}_Truncate{temp_str}_v0_fullsimilarity{use_full_Smatrix}'
        if not (os.path.exists(similarity_file_name+f'_SM{round_of_smooth}.npz') and (not compute_new)):
            similarity_matrix_full=generate_similarity_matrix(adata_orig,similarity_file_name,round_of_smooth=round_of_smooth,
                        neighbor_N=CoSpar_KNN,truncation_threshold=trunca_threshold,save_subset=True,compute_new_Smatrix=compute_new)
    
    logg.info("-------Step 3: Optimize the transition map recursively---------")

    infer_Tmap_from_multitime_clones_private(adata,smooth_array=smooth_array,neighbor_N=CoSpar_KNN,noise_threshold=noise_threshold,demulti_threshold=demulti_threshold,normalization_mode=normalization_mode,
            save_subset=save_subset,use_full_Smatrix=use_full_Smatrix,trunca_threshold=trunca_threshold,compute_new_Smatrix=compute_new)


    return adata
    

def infer_Tmap_from_multitime_clones_private(adata,smooth_array=[15,10,5],neighbor_N=20,noise_threshold=0.1,demulti_threshold=0.05,normalization_mode=1,save_subset=True,use_full_Smatrix=False,trunca_threshold=0.001,compute_new_Smatrix=False):
    """
    Internal function for Tmap inference from multiTime clonal data.

    Same as :func:`.infer_Tmap_from_multitime_clones` except that it 
    assumes that the adata object has been prepared for targeted 
    time points. It generate the similarity matrix 
    via :func:`.generate_similarity_matrix`, and iterately calls 
    the core function :func:`.refine_Tmap_through_cospar` to update 
    the transition map. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Should be prepared by :func:`.select_time_points`
    smooth_array: `list`, optional (default: [15,10,5])
        List of smooth rounds at each iteration. 
        The n-th entry determines the smooth round for the Smatrix 
        at the n-th iteration. Its length determins the number of
        iteration.  
    neighbor_N: `int`, optional (default: 20)
        the number of neighbors for KNN graph used for computing the similarity matrix.
    trunca_threshold: `float`, optional (default: 0.001)
        We set entries to zero in the computed similarity matrix that 
        are smaller than this threshold. This is to promote the Smatrix sparsity, which
        leads to faster computation, and smaller file size. 
        This threshld should be small, but not too small. 
    noise_threshold: `float`, optional (default: 0.1)
        noise threshold to remove noises in the updated transition map,
        in the range [0,1]
    demulti_threshold: `float`, optional (default: 0.05)
        noise threshold to remove noises in demultiplexed (un-smoothed) map,
        in the range [0,1]
    normalization_mode: `int`, optional (default: 1)
        Method for normalization. Choice: [0,1]
        0, single-cell normalization
        1, Clone normalization
    save_subset: `bool`, optional (default: True)
        If true, save only Smatrix at smooth round [5,10,15,...].
        Else, save Smatrix at each round. 
    use_full_Smatrix: `bool`, optional (default: False)
        use the Smatrix as defined by all cells, whether they are clonally 
        barcoded or not. We sub-sample cell states relevant for downstream 
        analysis from this full Smatrix. This may refine the Smatrix. 
        But will also increase the computation time significantly.
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
    data_des=adata.uns['data_des'][0]
    multiTime_cell_id_t1=adata.uns['multiTime_cell_id_t1']
    multiTime_cell_id_t2=adata.uns['multiTime_cell_id_t2']
    proportion=adata.uns['proportion']
    data_path=settings.data_path

    #########

    
    ########################### Compute the transition map 
    
    logg.info("---------Compute the transition map-----------")

    #trunca_threshold=0.001 # this value is only for reducing the computed matrix size for saving
    temp_str='0'+str(trunca_threshold)[2:]

    if use_full_Smatrix:
        similarity_file_name=f'{data_path}/Similarity_matrix_with_all_cell_states_kNN{neighbor_N}_Truncate{temp_str}_v0_fullsimilarityTrue'
        for round_of_smooth in smooth_array:
            if not os.path.exists(similarity_file_name+f'_SM{round_of_smooth}.npz'):
                logg.error(f"Similarity matrix at given parameters have not been computed before! Name: {similarity_file_name}")     
                return   

    else:
        similarity_file_name=f'{data_path}/{data_des}_Similarity_matrix_with_states_kNN{neighbor_N}_Truncate{temp_str}_v0_fullsimilarityFalse'

    initial_similarity_array=[]
    final_similarity_array=[]
    initial_similarity_array_ext=[]
    final_similarity_array_ext=[]

    for round_of_smooth in smooth_array:
        # we cannot force it to compute new at this time. Otherwise, if we use_full_Smatrix, the resulting similarity is actually from adata, thus not full similarity. 

        re_compute=(not use_full_Smatrix) and (compute_new_Smatrix) # re-compute only when not using full similarity 
        similarity_matrix_full=generate_similarity_matrix(adata,similarity_file_name,round_of_smooth=round_of_smooth,
                    neighbor_N=neighbor_N,truncation_threshold=trunca_threshold,save_subset=save_subset,compute_new_Smatrix=re_compute)

        if use_full_Smatrix:
            #pdb.set_trace()
            similarity_matrix_full_sp=similarity_matrix_full[sp_idx][:,sp_idx]

            #pdb.set_trace()
            ### extended similarity matrix
            initial_similarity_ext=generate_initial_similarity(similarity_matrix_full_sp,Tmap_cell_id_t1,clonal_cell_id_t1)
            final_similarity_ext=generate_final_similarity(similarity_matrix_full_sp,clonal_cell_id_t2,Tmap_cell_id_t2)
            
            ### minimum similarity matrix that only involves the multi-time clones
            initial_similarity=generate_initial_similarity(similarity_matrix_full_sp,clonal_cell_id_t1,clonal_cell_id_t1)
            final_similarity=generate_final_similarity(similarity_matrix_full_sp,clonal_cell_id_t2,clonal_cell_id_t2)
        else:
            initial_similarity_ext=generate_initial_similarity(similarity_matrix_full,Tmap_cell_id_t1,clonal_cell_id_t1)
            final_similarity_ext=generate_final_similarity(similarity_matrix_full,clonal_cell_id_t2,Tmap_cell_id_t2)
            initial_similarity=generate_initial_similarity(similarity_matrix_full,clonal_cell_id_t1,clonal_cell_id_t1)
            final_similarity=generate_final_similarity(similarity_matrix_full,clonal_cell_id_t2,clonal_cell_id_t2)


        initial_similarity_array.append(initial_similarity)
        final_similarity_array.append(final_similarity)
        initial_similarity_array_ext.append(initial_similarity_ext)
        final_similarity_array_ext.append(final_similarity_ext)


    #### Compute the core of the transition map that involve multi-time clones, then extend to other cell states
    clonal_coupling_v1=np.ones((len(clonal_cell_id_t1),len(clonal_cell_id_t2)))
    transition_map_array=[clonal_coupling_v1]



    X_clone=clone_annot.copy()
    if not ssp.issparse(X_clone):
        X_clone=ssp.csr_matrix(X_clone)

    CoSpar_iter_N=len(smooth_array)
    for j in range(CoSpar_iter_N):
        
        logg.info("Current iteration:",j)
        transition_map=transition_map_array[j]
        if j<len(smooth_array):
            
            logg.info(f"Use smooth_round={smooth_array[j]}")
            used_initial_similarity=initial_similarity_array[j]
            used_final_similarity=final_similarity_array[j]
        else:
            
            logg.info(f"Use smooth_round={smooth_array[-1]}")
            used_initial_similarity=initial_similarity_array[-1]
            used_final_similarity=final_similarity_array[-1]

        # clonal_coupling, unSM_sc_coupling=refine_transition_map_by_integrating_clonal_info(clonal_cell_id_t1,clonal_cell_id_t2,
        #        transition_map,X_clone,used_initial_similarity,used_final_similarity,noise_threshold,row_normalize=True,normalization_mode=normalization_mode)

        
        clonal_coupling, unSM_sc_coupling=refine_Tmap_through_cospar(multiTime_cell_id_t1,multiTime_cell_id_t2,
            proportion,transition_map,X_clone,used_initial_similarity,used_final_similarity,noise_threshold=noise_threshold,normalization_mode=normalization_mode)


        transition_map_array.append(clonal_coupling)



    ### expand the map to other cell states
    ratio_t1=np.sum(np.in1d(Tmap_cell_id_t1,clonal_cell_id_t1))/len(Tmap_cell_id_t1)
    ratio_t2=np.sum(np.in1d(Tmap_cell_id_t2,clonal_cell_id_t2))/len(Tmap_cell_id_t2)
    if (ratio_t1==1) and (ratio_t2==1): # no need to SM the map
        
        logg.info("No need for Final Smooth (i.e., clonally states are the final state space for Tmap)")
        
        adata.uns['transition_map']=ssp.csr_matrix(clonal_coupling)
    else:
        
        logg.info("Final round of Smooth (to expand the state space of Tmap to include non-clonal states)")

        if j<len(smooth_array):
            used_initial_similarity_ext=initial_similarity_array_ext[j]
            used_final_similarity_ext=final_similarity_array_ext[j]
        else:
            used_initial_similarity_ext=initial_similarity_array_ext[-1]
            used_final_similarity_ext=final_similarity_array_ext[-1]

        unSM_sc_coupling=ssp.csr_matrix(unSM_sc_coupling)
        t=time.time()
        temp=unSM_sc_coupling*used_final_similarity_ext
        
        logg.info("Phase I: time elapsed -- ", time.time()-t)
        transition_map_1=used_initial_similarity_ext.dot(temp)
        
        logg.info("Phase II: time elapsed -- ", time.time()-t)


        adata.uns['transition_map']=ssp.csr_matrix(transition_map_1)
        #adata.uns['transition_map_unExtended']=ssp.csr_matrix(clonal_coupling)


    
    logg.info("----Demultiplexed transition map----")

    #pdb.set_trace()
    demultiplexed_map_0=refine_Tmap_through_cospar_noSmooth(multiTime_cell_id_t1,multiTime_cell_id_t2,proportion,clonal_coupling,
        X_clone,noise_threshold=demulti_threshold,normalization_mode=normalization_mode)

    idx_t1=hf.converting_id_from_fullSpace_to_subSpace(clonal_cell_id_t1,Tmap_cell_id_t1)[0]
    idx_t2=hf.converting_id_from_fullSpace_to_subSpace(clonal_cell_id_t2,Tmap_cell_id_t2)[0]
    demultiplexed_map=np.zeros((len(Tmap_cell_id_t1),len(Tmap_cell_id_t2)))
    demultiplexed_map[idx_t1[:,np.newaxis],idx_t2]=demultiplexed_map_0.A
    adata.uns['intraclone_transition_map']=ssp.csr_matrix(demultiplexed_map)



def infer_intraclone_Tmap(adata,demulti_threshold=0.05,normalization_mode=1):
    """
    Infer intra-clone transition map.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Should be prepared by :func:`.select_time_points`
    demulti_threshold: `float`, optional (default: 0.05)
        noise threshold to remove noises in transition_map,
        in the range [0,1]
    normalization_mode: `int`, optional (default: 1)
        Method for normalization. Choice: [0,1]
        0, single-cell normalization
        1, Clone normalization

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
            X_clone,noise_threshold=demulti_threshold,normalization_mode=normalization_mode)

        adata.uns['intraclone_transition_map']=ssp.csr_matrix(demultiplexed_map)


# v1: do not avoid cells that are selected. We tested that it is not as good as avoiding sharing cells.
def Tmap_from_highly_variable_genes_v1(adata,min_counts=3,min_cells=3,
    min_gene_vscore_pctl=85,smooth_array=[15,10,5],neighbor_N=20,
    noise_threshold=0.2,normalization_mode=1,use_full_Smatrix=False,
    trunca_threshold=0.001,compute_new_Smatrix=True,
    save_subset=True):
    """
    Generate Tmap based on state info using HighVar.

    We convert differentially expressed genes into `pseudo-clones`,
    and run cospar to infer the transition map. Different clones may
    share some cells. 
    

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        assumed to be preprocessed, only has two time points.
    min_counts: int, optional (default: 3)  
        Minimum number of UMIs per cell to be considered for selecting highly variable genes. 
    min_cells: int, optional (default: 3)
        Minimum number of cells per gene to be considered for selecting highly variable genes. 
    min_gene_vscore_pctl: int, optional (default: 85)
        Genes wht a variability percentile higher than this threshold are marked as 
        highly variable genes for dimension reduction. Range: [0,100]
        
    smooth_array: `list`, optional (default: [15,10,5])
        List of smooth rounds at each iteration. 
        The n-th entry determines the smooth round for the Smatrix 
        at the n-th iteration. Its length determins the number of
        iteration.  
    neighbor_N: `int`, optional (default: 20)
        the number of neighbors for KNN graph used for computing the similarity matrix.
    trunca_threshold: `float`, optional (default: 0.001)
        We set entries to zero in the computed similarity matrix that 
        are smaller than this threshold. This is to promote the Smatrix sparsity, which
        leads to faster computation, and smaller file size. 
        This threshld should be small, but not too small. 
    noise_threshold: `float`, optional (default: 0.1)
        noise threshold to remove noises in the updated transition map,
        in the range [0,1]
    normalization_mode: `int`, optional (default: 2)
        Method for normalization. Choice: [0,1,2]
        0, single-cell normalization
        1, Clone normalization: N2/N1 (this one does not make sense)
        2, Clone normalization
    save_subset: `bool`, optional (default: True)
        If true, save only Smatrix at smooth round [5,10,15,...].
        Else, save Smatrix at each round. 
    use_full_Smatrix: `bool`, optional (default: False)
        use the Smatrix as defined by all cells, whether they are clonally 
        barcoded or not. We sub-sample cell states relevant for downstream 
        analysis from this full Smatrix. This may refine the Smatrix. 
        But will also increase the computation time significantly.
    compute_new_Smatrix: `bool`, optional (default: False)
        If True, compute Smatrix from scratch, whether it was 
        computed and saved before or not.

    Returns
    -------
    None. Results are stored at adata.uns['HighVar_transition_map']. 
    """

    logg.info("HighVar-v1: do not avoid cells that have been selected")

    weight=1 # wehight of each gene. 

    cell_id_array_t1=adata.uns['Tmap_cell_id_t1']
    cell_id_array_t2=adata.uns['Tmap_cell_id_t2']
    real_clone_annot=adata.obsm['X_clone']

    time_info=np.array(adata.obs['time_info'])
    selected_time_points=[time_info[cell_id_array_t1][0],time_info[cell_id_array_t2][0]]


    
    logg.info("----------------")
    logg.info('Step a: find the commonly shared highly variable genes')
    adata_t1=sc.AnnData(adata.X[cell_id_array_t1]);
    adata_t2=sc.AnnData(adata.X[cell_id_array_t2]);

    ## use marker genes
    gene_list=adata.var_names

    verbose=logg._settings_verbosity_greater_or_equal_than(2)

    highvar_genes_t1 = gene_list[hf.filter_genes(
        adata_t1.X, 
        min_counts=min_counts, 
        min_cells=min_cells, 
        min_vscore_pctl=min_gene_vscore_pctl, 
        show_vscore_plot=verbose)]

    highvar_genes_t2 = gene_list[hf.filter_genes(
        adata_t2.X, 
        min_counts=min_counts, 
        min_cells=min_cells, 
        min_vscore_pctl=min_gene_vscore_pctl, 
        show_vscore_plot=verbose)]

    common_gene=list(set(highvar_genes_t1).intersection(highvar_genes_t2))
    
    logg.info(f"Highly varable gene number at t1 is {len(highvar_genes_t1)}, Highly varable gene number at t2 is {len(highvar_genes_t2)}")
    logg.info(f"Common gene set is {len(common_gene)}")

    logg.info("----------------")
    logg.info('Step b: convert the shared highly variable genes into clonal info')

    sel_marker_gene_list=common_gene.copy()
    clone_annot_gene=np.zeros((adata.shape[0],len(sel_marker_gene_list)))
    N_t1=len(cell_id_array_t1)
    N_t2=len(cell_id_array_t2)
    #cumu_sel_idx_t1=np.zeros(N_t1,dtype=bool)
    #cumu_sel_idx_t2=np.zeros(N_t2,dtype=bool)
    cell_fraction_per_gene=1/len(sel_marker_gene_list) # fraction of cells as clonally related by this gene
    cutoff_t1=int(np.ceil(len(cell_id_array_t1)*cell_fraction_per_gene))
    cutoff_t2=int(np.ceil(len(cell_id_array_t2)*cell_fraction_per_gene))
    for j,gene_id in enumerate(sel_marker_gene_list): 
        temp_t1=adata.obs_vector(gene_id)[cell_id_array_t1]
        #temp_t1[cumu_sel_idx_t1]=0 # set selected cell id to have zero expression
        sel_id_t1=np.argsort(temp_t1,kind='stable')[::-1][:cutoff_t1]
        clone_annot_gene[cell_id_array_t1[sel_id_t1],j]=weight
        #cumu_sel_idx_t1[sel_id_t1]=True 
        #logg.info(f"Gene id {gene_id}, cell number at t1 is {sel_id_t1.shape[0]}, fraction at t1: {sel_id_t1.shape[0]/len(cell_id_array_t1)}")

        temp_t2=adata.obs_vector(gene_id)[cell_id_array_t2]
        #temp_t2[cumu_sel_idx_t2]=0 # set selected cell id to have zero expression
        sel_id_t2=np.argsort(temp_t2,kind='stable')[::-1][:cutoff_t2]
        clone_annot_gene[cell_id_array_t2[sel_id_t2],j]=weight
        #cumu_sel_idx_t2[sel_id_t2]=True 
        #logg.info(f"Gene id {gene_id}, cell number at t2 is {sel_id_t2.shape[0]}, fraction at t2: {sel_id_t2.shape[0]/len(cell_id_array_t2)}")
        
        # if (np.sum(~cumu_sel_idx_t1)==0) or (np.sum(~cumu_sel_idx_t2)==0):
        #     logg.info(f'No cells left for assignment, total used genes={j}')
        #     break

    #logg.info(f"Selected cell fraction: t1 -- {np.sum(cumu_sel_idx_t1)/len(cell_id_array_t1)}; t2 -- {np.sum(cumu_sel_idx_t2)/len(cell_id_array_t2)}")


    
    logg.info("----------------")
    logg.info("Step c: compute the transition map based on clonal info from highly variable genes")
    
    adata.obsm['X_clone']=ssp.csr_matrix(clone_annot_gene)
    adata.uns['multiTime_cell_id_t1']=[cell_id_array_t1]
    adata.uns['multiTime_cell_id_t2']=[cell_id_array_t2]
    adata.uns['proportion']=[1]
    data_des_0=adata.uns['data_des'][0]
    data_des_1=data_des_0+'_HighVar0' # to distinguish Similarity matrix for this step and the next step of CoSpar (use _HighVar0, instead of _HighVar1)
    adata.uns['data_des'][0]=[data_des_1]

    infer_Tmap_from_multitime_clones_private(adata,smooth_array=smooth_array,neighbor_N=neighbor_N,noise_threshold=noise_threshold,
        normalization_mode=normalization_mode,save_subset=save_subset,use_full_Smatrix=use_full_Smatrix,
        trunca_threshold=trunca_threshold,compute_new_Smatrix=compute_new_Smatrix)

    adata.uns['HighVar_transition_map']=adata.uns['transition_map']
    adata.obsm['X_clone']=real_clone_annot # This entry has been changed previously. Note correct the clonal matrix
    data_des_1=data_des_0+'_HighVar1' # to record which initialization is used
    adata.uns['data_des']=[data_des_1]


# v0: avoid cells that are already selected. We tested, this is better than not avoiding...
def Tmap_from_highly_variable_genes(adata,min_counts=3,min_cells=3,
    min_gene_vscore_pctl=85,smooth_array=[15,10,5],neighbor_N=20,
    noise_threshold=0.2,normalization_mode=1,use_full_Smatrix=False,
    trunca_threshold=0.001,compute_new_Smatrix=True,
    save_subset=True):
    """
    Generate Tmap based on state info using HighVar.

    We convert differentially expressed genes into `pseudo-clones`,
    and run cospar to infer the transition map. Each clone occupies 
    a different set of cells. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        assumed to be preprocessed, only has two time points.
    min_counts: int, optional (default: 3)  
        Minimum number of UMIs per cell to be considered for selecting highly variable genes. 
    min_cells: int, optional (default: 3)
        Minimum number of cells per gene to be considered for selecting highly variable genes. 
    min_gene_vscore_pctl: int, optional (default: 85)
        Genes wht a variability percentile higher than this threshold are marked as 
        highly variable genes for dimension reduction. Range: [0,100]
        
    smooth_array: `list`, optional (default: [15,10,5])
        List of smooth rounds at each iteration. 
        The n-th entry determines the smooth round for the Smatrix 
        at the n-th iteration. Its length determins the number of
        iteration.  
    neighbor_N: `int`, optional (default: 20)
        the number of neighbors for KNN graph used for computing the similarity matrix.
    trunca_threshold: `float`, optional (default: 0.001)
        We set entries to zero in the computed similarity matrix that 
        are smaller than this threshold. This is to promote the Smatrix sparsity, which
        leads to faster computation, and smaller file size. 
        This threshld should be small, but not too small. 
    noise_threshold: `float`, optional (default: 0.1)
        noise threshold to remove noises in the updated transition map,
        in the range [0,1]
    normalization_mode: `int`, optional (default: 2)
        Method for normalization. Choice: [0,1,2]
        0, single-cell normalization
        1, Clone normalization: N2/N1 (this one does not make sense)
        2, Clone normalization
    save_subset: `bool`, optional (default: True)
        If true, save only Smatrix at smooth round [5,10,15,...].
        Else, save Smatrix at each round. 
    use_full_Smatrix: `bool`, optional (default: False)
        use the Smatrix as defined by all cells, whether they are clonally 
        barcoded or not. We sub-sample cell states relevant for downstream 
        analysis from this full Smatrix. This may refine the Smatrix. 
        But will also increase the computation time significantly.
    compute_new_Smatrix: `bool`, optional (default: False)
        If True, compute Smatrix from scratch, whether it was 
        computed and saved before or not.

    Returns
    -------
    None. Results are stored at adata.uns['HighVar_transition_map']. 
    """
    logg.info("HighVar-v0: avoid cells that have been selected")
    weight=1 # wehight of each gene. 

    cell_id_array_t1=adata.uns['Tmap_cell_id_t1']
    cell_id_array_t2=adata.uns['Tmap_cell_id_t2']
    real_clone_annot=adata.obsm['X_clone']

    time_info=np.array(adata.obs['time_info'])
    selected_time_points=[time_info[cell_id_array_t1][0],time_info[cell_id_array_t2][0]]


    
    logg.info("----------------")
    logg.info('Step a: find the commonly shared highly variable genes')
    adata_t1=sc.AnnData(adata.X[cell_id_array_t1]);
    adata_t2=sc.AnnData(adata.X[cell_id_array_t2]);

    ## use marker genes
    gene_list=adata.var_names

    verbose=logg._settings_verbosity_greater_or_equal_than(2)

    highvar_genes_t1 = gene_list[hf.filter_genes(
        adata_t1.X, 
        min_counts=min_counts, 
        min_cells=min_cells, 
        min_vscore_pctl=min_gene_vscore_pctl, 
        show_vscore_plot=verbose)]

    highvar_genes_t2 = gene_list[hf.filter_genes(
        adata_t2.X, 
        min_counts=min_counts, 
        min_cells=min_cells, 
        min_vscore_pctl=min_gene_vscore_pctl, 
        show_vscore_plot=verbose)]

    common_gene=list(set(highvar_genes_t1).intersection(highvar_genes_t2))
    
    logg.info(f"Highly varable gene number at t1 is {len(highvar_genes_t1)}, Highly varable gene number at t2 is {len(highvar_genes_t2)}")
    logg.info(f"Common gene set is {len(common_gene)}")

    logg.info("----------------")
    logg.info('Step b: convert the shared highly variable genes into clonal info')

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
            logg.info(f'No cells left for assignment, total used genes={j}')
            break

    #logg.info(f"Selected cell fraction: t1 -- {np.sum(cumu_sel_idx_t1)/len(cell_id_array_t1)}; t2 -- {np.sum(cumu_sel_idx_t2)/len(cell_id_array_t2)}")


    
    logg.info("----------------")
    logg.info("Step c: compute the transition map based on clonal info from highly variable genes")
    
    adata.obsm['X_clone']=ssp.csr_matrix(clone_annot_gene)
    adata.uns['multiTime_cell_id_t1']=[cell_id_array_t1]
    adata.uns['multiTime_cell_id_t2']=[cell_id_array_t2]
    adata.uns['proportion']=[1]
    data_des_0=adata.uns['data_des'][0]
    data_des_1=data_des_0+'_HighVar0' # to distinguish Similarity matrix for this step and the next step of CoSpar (use _HighVar0, instead of _HighVar1)
    adata.uns['data_des'][0]=[data_des_1]

    infer_Tmap_from_multitime_clones_private(adata,smooth_array=smooth_array,neighbor_N=neighbor_N,noise_threshold=noise_threshold,
        normalization_mode=normalization_mode,save_subset=save_subset,use_full_Smatrix=use_full_Smatrix,
        trunca_threshold=trunca_threshold,compute_new_Smatrix=compute_new_Smatrix)

    adata.uns['HighVar_transition_map']=adata.uns['transition_map']
    adata.obsm['X_clone']=real_clone_annot # This entry has been changed previously. Note correct the clonal matrix
    data_des_1=data_des_0+'_HighVar1' # to record which initialization is used
    adata.uns['data_des']=[data_des_1]



# this is the new version: v1
def compute_custom_OT_transition_map(adata,OT_epsilon=0.02,OT_dis_KNN=5,
    OT_solver='duality_gap',OT_cost='SPD',compute_new=True):
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
        The entropic regularization, >0, a larger one increases 
        uncertainty of the transition
    OT_dis_KNN: `int`, optional (default: 5)
        Number of nearest neighbors to construct the KNN graph for
        computing the shortest path distance. 
    OT_solver: `str`, optional (default: `duality_gap`)
        The method used to compute the optimal transport map. Availabel choice: 
        {'duality_gap','fixed_iters'}. Our test shows that they produce the same 
        results, while 'duality_gap' is almost twice faster. 
    OT_cost: `str`, optional (default: `SPD`), options {'GED','SPD'}
        The cost metric. We provide gene expression distance (GED), and also
        shortest path distance (SPD). GED is much faster, but SPD is more accurate.
        However, cospar is robust to the initialization. 
    compute_new: `bool`, optional (default: False)
        If True, compute OT_map and also the shortest path distance from scratch, 
        whether it was computed and saved before or not.

    Returns
    -------
    None. Results are stored at adata.uns['OT_transition_map'].
    """

    cell_id_array_t1=adata.uns['Tmap_cell_id_t1']
    cell_id_array_t2=adata.uns['Tmap_cell_id_t2']
    data_des=adata.uns['data_des'][0]
    data_path=settings.data_path


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
            ShortPath_dis=hf.compute_shortest_path_distance(adata,num_neighbors_target=OT_dis_KNN,mode='distances',method='umap')
            
            idx0=cell_id_array_t1
            idx1=cell_id_array_t2
            ShortPath_dis_t0t1=ShortPath_dis[idx0[:,np.newaxis],idx1]; 
            OT_cost_matrix=ShortPath_dis_t0t1/ShortPath_dis_t0t1.max()


            np.save(SPD_file_name,OT_cost_matrix)


            logg.info(f"Finishing computing shortest-path distance, used time {time.time()-t}")
    else:
        t=time.time()
        pc_n=adata.obsm['X_pca'].shape[1]
        OT_cost_matrix=hf.compute_gene_exp_distance(adata,cell_id_array_t1,cell_id_array_t2,pc_n=pc_n)
        logg.info(f"Finishing computing gene expression distance, used time {time.time()-t}")  
            

    ######## apply optimal transport
    CustomOT_file_name=f'{data_path}/{data_des}_CustomOT_map_epsilon{OT_epsilon}_KNN{OT_dis_KNN}.npy'
    if os.path.exists(CustomOT_file_name) and (not compute_new):

        logg.info("Load pre-computed custon OT matrix")
        OT_transition_map=np.load(CustomOT_file_name)

    else:
        logg.info("Compute new custon OT matrix")

        t=time.time()
        mu1=np.ones(len(cell_id_array_t1));
        nu1=np.ones(len(cell_id_array_t2));
        input_mu=mu1 # initial distribution
        input_nu=nu1 # final distribution

        ######### We have tested that it is at least 3 times slower than WOT's builtin method, 
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

        np.save(CustomOT_file_name,OT_transition_map)

        logg.info(f"Finishing computing optial transport map, used time {time.time()-t}")


    adata.uns['OT_transition_map']=ssp.csr_matrix(OT_transition_map)
    data_des_0=adata.uns['data_des'][0]
    data_des_1=data_des_0+'_OT' # to record which initialization is used
    adata.uns['data_des']=[data_des_1]


# this is the old version    
def compute_custom_OT_transition_map_v0(adata,OT_epsilon=0.02,OT_max_iter=1000,
    OT_stopThr=1e-09,OT_dis_KNN=5,OT_method='OT',compute_new=True):
    """
    Compute Tmap from state info using optimal transport (OT).

    This realization is based on :func:`ot.bregman.sinkhorn_stabilized`. 
    The cost function is from shortest path distance, and we set the 
    initial and final distribution to be both uniform. 

    The KNN graph generation does not consider local cell density 
    heterogeneity. The shortest path distance is the cumulative gene expression 
    distance along the path, not the cumulative connectivity. The function tends
    to be very slow, both the computation of shortest path distance, and also the 
    generation of OT_map. Since CoSpar is not so sensitive to initialization,
    we might provide another implementation that sacrifice accuracy, but boost speed. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assumed to be preprocessed, only has two time points.
    OT_epsilon: `float`, optional (default: 0.02)  
        The entropic regularization, >0, a larger one increases 
        uncertainty of the transition
    OT_max_iter: `int`, optional (default: 1000)
        Maximum round of iteration.
    OT_stopThr: `float`, optional (default: 1e-09)
        Stop if the cost function change per iteration is 
        less than this threshold. 
    OT_dis_KNN: `int`, optional (default: 5)
        Number of nearest neighbors to construct the KNN graph for
        computing the shortest path distance. 
    compute_new: `bool`, optional (default: False)
        If True, compute OT_map and also the shortest path distance from scratch, 
        whether it was computed and saved before or not.

    Returns
    -------
    None. Results are stored at adata.uns['OT_transition_map'].
    """

    import ot.bregman as otb
    cell_id_array_t1=adata.uns['Tmap_cell_id_t1']
    cell_id_array_t2=adata.uns['Tmap_cell_id_t2']
    data_des=adata.uns['data_des'][0]
    data_path=settings.data_path


    ############ Compute shorted-path distance
    # use sklearn KNN graph construction method and select the connectivity option, not related to UMAP
    # use the mode 'distance' to obtain the shortest-path *distance*, rather than 'connectivity'
    if OT_method=='OT':
        SPD_file_name=f'{data_path}/{data_des}_ShortestPathDistanceMatrix_t0t1_KNN{OT_dis_KNN}.npy'
        if os.path.exists(SPD_file_name) and (not compute_new):
            
            logg.info("Load pre-computed shortest path distance matrix")
            ShortPath_dis_t0t1=np.load(SPD_file_name)

        else:
            
            logg.info("Compute new shortest path distance matrix")
            t=time.time()       
            #data_matrix=adata.obsm['X_pca']
            #ShortPath_dis=hf.compute_shortest_path_distance_from_raw_matrix(data_matrix,num_neighbors_target=OT_dis_KNN,mode='distance')
            ShortPath_dis=hf.compute_shortest_path_distance(adata,num_neighbors_target=OT_dis_KNN,mode='distances',method='umap')

            idx0=cell_id_array_t1
            idx1=cell_id_array_t2
            ShortPath_dis_t0t1=ShortPath_dis[idx0[:,np.newaxis],idx1]; ShortPath_dis_t0t1=ShortPath_dis_t0t1/ShortPath_dis_t0t1.max()
            ShortPath_dis_t0=ShortPath_dis[idx0[:,np.newaxis],idx0]; ShortPath_dis_t0=ShortPath_dis_t0/ShortPath_dis_t0.max()
            ShortPath_dis_t1=ShortPath_dis[idx1[:,np.newaxis],idx1]; ShortPath_dis_t1=ShortPath_dis_t1/ShortPath_dis_t1.max()

            np.save(SPD_file_name,ShortPath_dis_t0t1)

            
            logg.info(f"Finishing computing shortest-path distance, used time {time.time()-t}")


        ######## apply optimal transport
        CustomOT_file_name=f'{data_path}/{data_des}_CustomOT_map_epsilon{OT_epsilon}_IterN{OT_max_iter}_stopThre{OT_stopThr}_KNN{OT_dis_KNN}.npy'
        if os.path.exists(CustomOT_file_name) and (not compute_new):
            
            logg.info("Load pre-computed custon OT matrix")
            OT_transition_map=np.load(CustomOT_file_name)

        else:
            
            logg.info("Compute new custon OT matrix")


            t=time.time()
            mu1=np.ones(len(cell_id_array_t1));
            nu1=np.ones(len(cell_id_array_t2));
            input_mu=mu1 # initial distribution
            input_nu=nu1 # final distribution
            OT_transition_map=otb.sinkhorn_stabilized(input_mu,input_nu,ShortPath_dis_t0t1,OT_epsilon,numItermax=OT_max_iter,stopThr=OT_stopThr)

            np.save(CustomOT_file_name,OT_transition_map)

            
            logg.info(f"Finishing computing optial transport map, used time {time.time()-t}")
    else:
        logg.info("Use WOT")
        import wot
        time_info=np.zeros(adata.shape[0])
        time_info[cell_id_array_t1]=1
        time_info[cell_id_array_t2]=2
        adata.obs['day']=time_info
        adata.obs['cell_growth_rate']=np.ones(len(time_info))
        epsilon=0.02
        ot_model = wot.ot.OTModel(adata,epsilon = epsilon, lambda1 = 1,lambda2 = 50)
        OT_transition_map = ot_model.compute_transport_map(1,2).X 

    adata.uns['OT_transition_map']=ssp.csr_matrix(OT_transition_map)
    data_des_0=adata.uns['data_des'][0]
    data_des_1=data_des_0+'_OT' # to record which initialization is used
    adata.uns['data_des']=[data_des_1]


def infer_initial_states_of_a_clone(current_clone_id,clonal_cell_id_t2_subspace,
    available_cell_id_t1_subspace,cell_N_to_extract,cell_id_array_t1,
    cell_id_array_t2,clone_annot_new,transition_map):
    """
    Infer initial states of a clone from given transition map

    Parameters
    ----------
    current_clone_id: `int`
        current clone id among all clones.
    clonal_cell_id_t2_subspace: `np.array`
        Id's of cells belonging to this clone among t2-cells.
    available_cell_id_t1_subspace: `np.array`
        Id's of un-clonally-labeled cells among t1-cells. 
    cell_N_to_extract: `int`
        Number of t1-cells to extract for this clone.
    cell_id_array_t1: `np.array`
        Id of all t1 cells.
    cell_id_array_t2: `np.array`
        Id of all t2 cells.
    clone_annot_new: `np.array`
        X_clone matrix to store inferred clonal information
    transition_map: `sp.spsparse`
        Transition map used for inferring initial state likelihood 
    
    Returns
    -------
    clone_annot_new: `np.array`
        Updated X_clone matrix
    sel_id_t1: `np.array`
        Selected cell ids at t1 for this clone. To be excluded for later inference. 
    """

    # add cell states on t2 for this clone
    clone_annot_new[cell_id_array_t2[clonal_cell_id_t2_subspace],current_clone_id]=True
    
    # infer the earlier clonal states for each clone
    ### select the early states using the grouped distribution of a clone
    ### clones are not overlapping, and all early states should be attached to clones at the end
    sorted_id_array=np.argsort(transition_map[available_cell_id_t1_subspace][:,clonal_cell_id_t2_subspace].sum(1).A.flatten(),kind='stable')[::-1]
    #available_cell_id=np.nonzero(available_cell_idx_t1_subspace)[0]

    #pdb.set_trace()
    if len(sorted_id_array)>cell_N_to_extract:
        sel_id_t1=available_cell_id_t1_subspace[sorted_id_array][:cell_N_to_extract]
    else:
        sel_id_t1=available_cell_id_t1_subspace

    # add cell states on t1 for this clone
    clone_annot_new[cell_id_array_t1[sel_id_t1],current_clone_id]=np.ones(len(sel_id_t1),dtype=bool)

    return clone_annot_new,sel_id_t1


def round_number_probabilistically(x):
    """
    Round a float number probabilistically into its closet integer.

    This is used to partition number of initial states to each clone.  
    """

    y0=int(x) # round the number directly
    x1=x-y0 # between 0 and 1
    if np.random.rand()<x1:
        y=y0+1
    else:
        y=y0
    return int(y)



# We tested that, for clones of all different sizes, where np.argsort gives unique results, 
# this method reproduces the v01, v1 results, when use_fixed_clonesize_t1=True, and when change
# sort_clone=0,1,-1.
def infer_Tmap_from_one_time_clones_private(adata,initialized_map,Clone_update_iter_N=1,
    smooth_array=[15,10,5],CoSpar_KNN=20,normalization_mode=1,noise_threshold=0.2,
    use_full_Smatrix=False,trunca_threshold=0.001,compute_new=True,
    use_fixed_clonesize_t1=False,sort_clone=1):
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
    Clone_update_iter_N: `int`, optional (default: 1)
        Number of iteration for the joint optimization.
    normalization_mode: `int`, optional (default: 1)
        Method for normalization. Choice: [0,1]
        0, single-cell normalization
        1, Clone normalization
    smooth_array: `list`, optional (default: [15,10,5])
        List of smooth rounds at each iteration. 
        The n-th entry determines the smooth round for the Smatrix 
        at the n-th iteration. Its length determins the number of
        iteration.  
    CoSpar_KNN: `int`, optional (default: 20)
        the number of neighbors for KNN graph used for computing the similarity matrix.
    trunca_threshold: `float`, optional (default: 0.001)
        We set entries to zero in the computed similarity matrix that 
        are smaller than this threshold. This is to promote the Smatrix sparsity, which
        leads to faster computation, and smaller file size. 
        This threshld should be small, but not too small. 
    noise_threshold: `float`, optional (default: 0.1)
        threshold to remove noises in the updated transition map,
        in the range [0,1]
    save_subset: `bool`, optional (default: True)
        If true, save only Smatrix at smooth round [5,10,15,...].
        Else, save Smatrix at each round. 
    use_full_Smatrix: `bool`, optional (default: False)
        use the Smatrix as defined by all cells, whether they are clonally 
        barcoded or not. We sub-sample cell states relevant for downstream 
        analysis from this full Smatrix. This may refine the Smatrix. 
        But will also increase the computation time significantly.
    use_fixed_clonesize_t1: `bool`, optional (default: False)
        If true, fix the number of initial states as the same for all clones
    sort_clone: `int`, optional (default: 1)
        The order to infer initial states for each clone: {1,-1,others}
        1, sort clones by size from small to large
        -1,sort clones by size from large to small
        others, do not sort. 
    compute_new: `bool`, optional (default: False)
        If True, compute everthing (ShortestPathDis,OT_map etc.) from scratch, 
        whether it was computed and saved before or not.

    Returns
    ------
    None. Update adata.obsm['X_clone'] and adata.uns['transition_map'],
    as well as adata.uns['OT_transition_map'] or 
    adata.uns['intraclone_transition_map'], depending on the initialization.
    """

    # I found the error: 1) we should use clonally related cell number at t2 as a factor to determine the clonally cell number at t1
    #                    2) update the whole t2 clonal info at once

    logg.info("Joint optimization that consider possibility of clonal overlap: v2")

    cell_id_array_t1=adata.uns['Tmap_cell_id_t1']
    cell_id_array_t2=adata.uns['Tmap_cell_id_t2']
    data_des=adata.uns['data_des'][0]
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
        logg.info("Use fixed clone size at t1")

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
        logg.info("Sort clones by size (small to large)")

        sort_clone_id=np.argsort(clone_size_t2_temp,kind='stable')
        clone_size_t2=clone_size_t2_temp[sort_clone_id]
        X_clone_sort=X_clone_newBC[:,sort_clone_id]
        clone_mapping_sort=clone_mapping[:,sort_clone_id]

    elif sort_clone==-1:
        logg.info("Sort clones by size (large to small)")

        sort_clone_id=np.argsort(clone_size_t2_temp,kind='stable')[::-1]
        clone_size_t2=clone_size_t2_temp[sort_clone_id]
        X_clone_sort=X_clone_newBC[:,sort_clone_id]
        clone_mapping_sort=clone_mapping[:,sort_clone_id]

    else:
        logg.info("Do not order clones by size ")
        clone_size_t2=clone_size_t2_temp
        X_clone_sort=X_clone_newBC
        clone_mapping_sort=clone_mapping


    logg.info("Infer the number of initial cells to extract for each clone in advance")
    clone_N1=X_clone_sort.shape[1]
    ave_clone_size_t1=int(np.ceil(len(cell_id_array_t1)/clone_N1));
    cum_cell_N=np.ceil(np.cumsum(clone_size_t2)*len(cell_id_array_t1)/clonal_cells_t2)
    cell_N_to_extract=np.zeros(len(cum_cell_N),dtype=int)
    if use_fixed_clonesize_t1:
        cell_N_to_extract += ave_clone_size_t1
    else:
        cell_N_to_extract[0]=cum_cell_N[0]
        cell_N_to_extract[1:]=np.diff(cum_cell_N)


    for x0 in range(Clone_update_iter_N):


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

        infer_Tmap_from_multitime_clones_private(adata,smooth_array=smooth_array,neighbor_N=CoSpar_KNN,noise_threshold=noise_threshold,
            normalization_mode=normalization_mode,save_subset=True,use_full_Smatrix=use_full_Smatrix,
            trunca_threshold=trunca_threshold,compute_new_Smatrix=compute_new)

        # update, for the next iteration
        map_temp=adata.uns['transition_map']




# This is good, but time consuming. We tested that it gives exactly the same result as the v0 version.
def infer_Tmap_from_one_time_clones_private_v1(adata,initialized_map,Clone_update_iter_N=1,
    smooth_array=[15,10,5],CoSpar_KNN=20,normalization_mode=1,noise_threshold=0.2,
    use_full_Smatrix=False,trunca_threshold=0.001,compute_new=True,
    use_fixed_clonesize_t1=False,sort_clone=1):
    """
    Infer Tmap from clones with a single time point

    Starting from an initialized transitin map from state information,
    we jointly infer the initial clonal states and the transition map.
    This method is time consuming as compared to 
    :func:`infer_Tmap_from_one_time_clones_private_v2`
    and is also suffers from stochasticity in choosing initial cells. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Should have only two time points. 
    initialized_map: `sp.spmatrix`
        Initialized transition map based on state information alone.
    Clone_update_iter_N: `int`, optional (default: 1)
        Number of iteration for the joint optimization
    normalization_mode: `int`, optional (default: 1)
        Method for normalization. Choice: [0,1]
        0, single-cell normalization
        1, Clone normalization
    smooth_array: `list`, optional (default: [15,10,5])
        List of smooth rounds at each iteration. 
        The n-th entry determines the smooth round for the Smatrix 
        at the n-th iteration. Its length determins the number of
        iteration.  
    CoSpar_KNN: `int`, optional (default: 20)
        the number of neighbors for KNN graph used for computing the similarity matrix.
    trunca_threshold: `float`, optional (default: 0.001)
        We set entries to zero in the computed similarity matrix that 
        are smaller than this threshold. This is to promote the Smatrix sparsity, which
        leads to faster computation, and smaller file size. 
        This threshld should be small, but not too small. 
    noise_threshold: `float`, optional (default: 0.1)
        noise threshold to remove noises in the updated transition map,
        in the range [0,1]

    save_subset: `bool`, optional (default: True)
        If true, save only Smatrix at smooth round [5,10,15,...].
        Else, save Smatrix at each round. 
    use_full_Smatrix: `bool`, optional (default: False)
        use the Smatrix as defined by all cells, whether they are clonally 
        barcoded or not. We sub-sample cell states relevant for downstream 
        analysis from this full Smatrix. This may refine the Smatrix. 
        But will also increase the computation time significantly.
    use_fixed_clonesize_t1: `bool`, optional (default: False)
        If true, fix the number of initial states as the same for all clones
    sort_clone: `int`, optional (default: 1)
        The order to infer initial states for each clone: {1,-1,others}
        1, sort clones by size from small to large
        -1,sort clones by size from large to small
        others, do not sort. 
    compute_new: `bool`, optional (default: False)
        If True and `use_full_Smatrix=False`, compute Smatrix again. 

    Returns
    ------
    None. Update adata.obsm['X_clone'] and adata.uns['transition_map'],
    as well as adata.uns['OT_transition_map'] or 
    adata.uns['intraclone_transition_map'], depending on the initialization.
    """

    # I found the error: 1) we should use clonally related cell number at t2 as a factor to determine the clonal cell number at t1
    #                    2) update the whole t2 clonal info at once


    logg.info("Joint optimization that consider possibility of clonal overlap: v1")

    cell_id_array_t1=adata.uns['Tmap_cell_id_t1']
    cell_id_array_t2=adata.uns['Tmap_cell_id_t2']
    data_des=adata.uns['data_des'][0]
    data_path=settings.data_path
    clone_annot=adata.obsm['X_clone']
    if not ssp.issparse(clone_annot): clone_annot=ssp.csr_matrix(clone_annot) 

    time_info=np.array(adata.obs['time_info'])
    time_index_t1=time_info==(time_info[cell_id_array_t1[0]])
    time_index_t2=time_info==(time_info[cell_id_array_t2[0]])

    if not ssp.issparse(initialized_map):
        map_temp=ssp.csr_matrix(initialized_map)
    else:
        map_temp=initialized_map


    valid_clone_id=np.nonzero(clone_annot[cell_id_array_t2].sum(0).A.flatten()>0)[0]
    clone_annot_temp=clone_annot[:,valid_clone_id]
    clone_N1=clone_annot_temp.shape[1]
    clonal_cells_t2=np.sum(clone_annot_temp[cell_id_array_t2].sum(1).flatten())

    logg.hint(f"original clone shape: {clone_annot.shape}")
    logg.hint(f"After excluding zero-sized clones: {clone_annot_temp.shape}")


    flag=True # to check whether overlapping clones are found or not
    if use_fixed_clonesize_t1:
        logg.info("Use fixed clone size at t1")

    ave_clone_size_t1=int(np.ceil(len(cell_id_array_t1)/clone_N1));

    ### select the early states using the grouped distribution of a clone
    ### clones are not overlapping, and all early states should be attached to clones at the end

    # we sort clones according to their sizes. The order of cells are not affected. So, it should not affect downstream analysis
    # small clones tend to be the ones that are barcoded/mutated later, while large clones tend to be early mutations...
    clone_size_t2_temp=clone_annot_temp[cell_id_array_t2].sum(0).A.flatten()

    if sort_clone==1:
        logg.info("Sort clones by size (small to large)")
            
        sort_clone_id=np.argsort(clone_size_t2_temp,kind='stable')
        clone_size_t2=clone_size_t2_temp[sort_clone_id]
        clone_annot_sort=clone_annot_temp[:,sort_clone_id]
        
    elif sort_clone==-1:
        
        logg.info("Sort clones by size (large to small)")
            
        sort_clone_id=np.argsort(clone_size_t2_temp,kind='stable')[::-1]
        clone_size_t2=clone_size_t2_temp[sort_clone_id]
        clone_annot_sort=clone_annot_temp[:,sort_clone_id]
        
    else:
        
        logg.info("Do not order clones by size ")
        clone_size_t2=clone_size_t2_temp
        clone_annot_sort=clone_annot_temp
        


    for x0 in range(Clone_update_iter_N):

        ########## begin: update clones
        remaining_ids_t1=list(np.arange(len(cell_id_array_t1),dtype=int))

        clone_annot_new=np.zeros(clone_annot_sort.shape,dtype=bool)
        clone_annot_new[cell_id_array_t2]=clone_annot_sort[cell_id_array_t2].A.astype(bool) # update the whole t2 clones at once
        for j in range(clone_N1):
        #for j in range(1):
            if (j%50==0):
                #pdb.set_trace()
                logg.hint(f"Inferring early clonal states: current clone id {j}")


            # identify overlapped clones at t2
            overlap_cell_N_per_clone=(clone_annot_sort[cell_id_array_t2,j].T*clone_annot_sort[cell_id_array_t2,:j]).A.flatten()
            overlap_id=np.nonzero(overlap_cell_N_per_clone>0)[0]

            available_t2_idx=np.ones(len(cell_id_array_t2),dtype=bool)
            if len(overlap_id)>0:
                if flag:
                    logg.info("Use overlapping information")
                    flag=False

                for current_clone_id in overlap_id:
                    available_cell_id_t1_subspace=np.nonzero(clone_annot_new[cell_id_array_t1,current_clone_id].flatten()>0)[0]
                    #available_cell_idx_t1_subspace=cell_id_array_t1[temp_idx]
                    overlapped_idx_t2_subspace=(clone_annot_sort[cell_id_array_t2,current_clone_id].A.flatten()>0) & (clone_annot_sort[cell_id_array_t2,j].A.flatten()>0)
                    overlapped_cell_id_t2_subspace=np.nonzero(overlapped_idx_t2_subspace)[0]
                    cell_N_to_extract_0=len(overlapped_cell_id_t2_subspace)*len(cell_id_array_t1)/clonal_cells_t2
                    cell_N_to_extract=round_number_probabilistically(cell_N_to_extract_0)

                    clone_annot_new,sel_cell_id_t1=infer_initial_states_of_a_clone(current_clone_id,overlapped_cell_id_t2_subspace,available_cell_id_t1_subspace,cell_N_to_extract,cell_id_array_t1,cell_id_array_t2,clone_annot_new,map_temp)

                    available_t2_idx[overlapped_cell_id_t2_subspace]=False # These states have been used

            # Compute for cells that are unique to this clone
            current_clone_id=j

            overlapped_idx_t2_subspace= available_t2_idx & (clone_annot_sort[cell_id_array_t2,j].A.flatten()>0)
            overlapped_cell_id_t2_subspace=np.nonzero(overlapped_idx_t2_subspace)[0]
            available_t2_idx[overlapped_cell_id_t2_subspace]=False # These states have been used

            #overlapped_cell_id_t2=cell_id_array_t2[overlapped_idx_t2]
            cell_N_to_extract_0=len(overlapped_cell_id_t2_subspace)*len(cell_id_array_t1)/clonal_cells_t2
            if use_fixed_clonesize_t1:
                cell_N_to_extract=ave_clone_size_t1
            else:
                cell_N_to_extract=round_number_probabilistically(cell_N_to_extract_0)

            available_cell_id_t1_subspace=np.array(remaining_ids_t1)
            clone_annot_new,sel_cell_id_t1=infer_initial_states_of_a_clone(current_clone_id,overlapped_cell_id_t2_subspace,available_cell_id_t1_subspace,cell_N_to_extract,cell_id_array_t1,cell_id_array_t2,clone_annot_new,map_temp)

            # remove selected id from the list
            for kk in sel_cell_id_t1:
                remaining_ids_t1.remove(kk)

            if (len(remaining_ids_t1)==0) and ((j+1)<clone_N1): 
                logg.hint(f'Early break; current clone id: {j+1}')
                break
                
        ########### end: update clones

        cell_id_array_t1_new=np.nonzero((clone_annot_new.sum(1)>0) & (time_index_t1))[0]
        cell_id_array_t2_new=np.nonzero((clone_annot_new.sum(1)>0) & (time_index_t2))[0]


        #clone_annot_new=np.zeros(clone_annot_sort.shape,dtype=bool)
        #clone_annot_new[:,sort_clone_id]=clone_annot_new

        clone_annot_new=ssp.csr_matrix(clone_annot_new)
        adata.obsm['X_clone']=clone_annot_new
        adata.uns['multiTime_cell_id_t1']=[cell_id_array_t1_new] # For CoSpar, clonally-related states
        adata.uns['multiTime_cell_id_t2']=[cell_id_array_t2_new]
        adata.uns['clonal_cell_id_t1']=cell_id_array_t1_new # for prepare the similarity matrix with same cell states
        adata.uns['clonal_cell_id_t2']=cell_id_array_t2_new
        adata.uns['proportion']=[1]

        #pdb.set_trace()
        infer_Tmap_from_multitime_clones_private(adata,smooth_array=smooth_array,neighbor_N=CoSpar_KNN,noise_threshold=noise_threshold,
            normalization_mode=normalization_mode,save_subset=True,use_full_Smatrix=use_full_Smatrix,
            trunca_threshold=trunca_threshold,compute_new_Smatrix=compute_new)

        ## update, for the next iteration
        map_temp=adata.uns['transition_map']
        #clone_annot_sort=clone_annot_new.copy()



def infer_Tmap_from_one_time_clones_private_v01(adata,initialized_map,Clone_update_iter_N=1,
    normalization_mode=1,noise_threshold=0.2,CoSpar_KNN=20,use_full_Smatrix=False,
    smooth_array=[15,10,5],trunca_threshold=0.001,compute_new=True,sort_clone=1):
    """
    Infer Tmap from clones with a single time point

    This is the same as :func:`.infer_Tmap_from_one_time_clones_private`, at 
    use_fixed_clonesize_t1=True, except that it works for 
    mutually exclusive clones from static DNA barcoding.

    It is the same as v0, but add the option for sort_clone, np.argsort 
    can return strange results if there are clones of the same size 
    (the ordering of clone id is no longer deterministic). We tested that
    for clonal data.  
    """
    logg.info("This is version v0-1 for joint optimization, with actual sorting choice, but not use_fixed_clonesize_t1")

    cell_id_array_t1=adata.uns['Tmap_cell_id_t1']
    cell_id_array_t2=adata.uns['Tmap_cell_id_t2']
    data_des=adata.uns['data_des'][0]
    data_path=settings.data_path
    clone_annot=adata.obsm['X_clone']
    if not ssp.issparse(clone_annot): clone_annot=ssp.csr_matrix(clone_annot) 

    time_info=np.array(adata.obs['time_info'])
    time_index_t1=time_info==(time_info[cell_id_array_t1[0]])
    time_index_t2=time_info==(time_info[cell_id_array_t2[0]])

    if not ssp.issparse(initialized_map):
        map_temp=ssp.csr_matrix(initialized_map)
    else:
        map_temp=initialized_map

    clone_annot_temp=clone_annot.copy()
    clone_N1=clone_annot.shape[1]


   ### select the early states using the grouped distribution of a clone
    ### clones are not overlapping, and all early states should be attached to clones at the end

    # we sort clones according to their sizes. The order of cells are not affected. So, it should not affect downstream analysis
    # small clones tend to be the ones that are barcoded/mutated later, while large clones tend to be early mutations...
    clone_size_t2_temp=clone_annot_temp[cell_id_array_t2].sum(0).A.flatten()

    if sort_clone==1:
        logg.info("Sort clones by size (small to large)")
            
        sort_clone_id=np.argsort(clone_size_t2_temp,kind='stable')
        clone_size_t2=clone_size_t2_temp[sort_clone_id]
        clone_annot_sort=clone_annot_temp[:,sort_clone_id]
        
    elif sort_clone==-1:
        
        logg.info("Sort clones by size (large to small)")
            
        sort_clone_id=np.argsort(clone_size_t2_temp,kind='stable')[::-1]
        clone_size_t2=clone_size_t2_temp[sort_clone_id]
        clone_annot_sort=clone_annot_temp[:,sort_clone_id]
        
    else:
        
        logg.info("Do not order clones by size ")
        clone_size_t2=clone_size_t2_temp
        clone_annot_sort=clone_annot_temp

    ### select the early states using the grouped distribution of a clone
    ### clones are not overlapping, and all early states should be attached to clones at the end

    ave_clone_size_t1=int(np.ceil(len(cell_id_array_t1)/clone_N1));

    for x0 in range(Clone_update_iter_N):

        ########## begin: update clones
        remaining_ids_t1=list(np.arange(len(cell_id_array_t1),dtype=int))

        clone_annot_new=np.zeros(clone_annot_sort.shape,dtype=bool)
        clone_annot_new[cell_id_array_t2]=clone_annot_sort[cell_id_array_t2].A.astype(bool)
        for j in range(clone_N1):
            if (j%50==0):
                #pdb.set_trace()
                logg.hint(f"Inferring early clonal states: current clone id {j}")

            # add back the known clonal states at t2
            #pdb.set_trace()
            temp_t2_idx=clone_annot_sort[cell_id_array_t2][:,j].A.flatten()>0
            #clone_annot_new[cell_id_array_t2,j]=temp_t2_idx
            
            # infer the earlier clonal states for each clone
            ### select the early states using the grouped distribution of a clone
            ### clones are not overlapping, and all early states should be attached to clones at the end
            sorted_id_array=np.argsort(map_temp[remaining_ids_t1][:,temp_t2_idx].sum(1).A.flatten(),kind='stable')[::-1]
            sel_id_t1=sorted_id_array[:ave_clone_size_t1]
            temp_t1_idx=np.zeros(len(cell_id_array_t1),dtype=bool)
            temp_t1_idx[np.array(remaining_ids_t1)[sel_id_t1]]=True
            clone_annot_new[cell_id_array_t1,j]=temp_t1_idx
            for kk in np.array(remaining_ids_t1)[sel_id_t1]:
                remaining_ids_t1.remove(kk)
            
            if (len(remaining_ids_t1)==0) and ((j+1)<clone_N1): 
                logg.hint(f'Early break; current clone id: {j+1}')
                break
        ########### end: update clones

        cell_id_array_t1_new=np.nonzero((clone_annot_new.sum(1)>0) & (time_index_t1))[0]
        cell_id_array_t2_new=np.nonzero((clone_annot_new.sum(1)>0) & (time_index_t2))[0]


        clone_annot_new=ssp.csr_matrix(clone_annot_new)
        adata.obsm['X_clone']=clone_annot_new
        adata.uns['multiTime_cell_id_t1']=[cell_id_array_t1_new] # For CoSpar, clonally-related states
        adata.uns['multiTime_cell_id_t2']=[cell_id_array_t2_new]
        adata.uns['clonal_cell_id_t1']=cell_id_array_t1_new # for prepare the similarity matrix with same cell states
        adata.uns['clonal_cell_id_t2']=cell_id_array_t2_new
        adata.uns['proportion']=[1]

        infer_Tmap_from_multitime_clones_private(adata,smooth_array=smooth_array,neighbor_N=CoSpar_KNN,noise_threshold=noise_threshold,
            normalization_mode=normalization_mode,save_subset=True,use_full_Smatrix=use_full_Smatrix,
            trunca_threshold=trunca_threshold,compute_new_Smatrix=compute_new)

        ## update, for the next iteration
        map_temp=adata.uns['transition_map']
        #clone_annot_sort=clone_annot_new.copy()


def infer_Tmap_from_one_time_clones_private_v0(adata,initialized_map,Clone_update_iter_N=1,
    normalization_mode=1,noise_threshold=0.2,CoSpar_KNN=20,use_full_Smatrix=False,
    smooth_array=[15,10,5],trunca_threshold=0.001,compute_new=True):
    """
    Infer Tmap from clones with a single time point

    This is the same as :func:`.infer_Tmap_from_one_time_clones_private`, at 
    use_fixed_clonesize_t1=True and sort_clone=0, except that it works for 
    mutually exclusive clones from static DNA barcoding.
    """
    logg.info("This is version v0 for joint optimization")

    cell_id_array_t1=adata.uns['Tmap_cell_id_t1']
    cell_id_array_t2=adata.uns['Tmap_cell_id_t2']
    data_des=adata.uns['data_des'][0]
    data_path=settings.data_path
    clone_annot=adata.obsm['X_clone']
    if not ssp.issparse(clone_annot): clone_annot=ssp.csr_matrix(clone_annot) 

    time_info=np.array(adata.obs['time_info'])
    time_index_t1=time_info==(time_info[cell_id_array_t1[0]])
    time_index_t2=time_info==(time_info[cell_id_array_t2[0]])

    if not ssp.issparse(initialized_map):
        map_temp=ssp.csr_matrix(initialized_map)
    else:
        map_temp=initialized_map

    cell_id_array_t1_temp=cell_id_array_t1.copy()
    cell_id_array_t2_temp=cell_id_array_t2.copy()
    clone_annot_temp=clone_annot.copy()
    clone_N1=clone_annot.shape[1]


    ### select the early states using the grouped distribution of a clone
    ### clones are not overlapping, and all early states should be attached to clones at the end

    ave_clone_size_t1=int(np.ceil(len(cell_id_array_t1)/clone_N1));

    for x0 in range(Clone_update_iter_N):

        ########## begin: update clones
        remaining_ids_t1=list(np.arange(len(cell_id_array_t1),dtype=int))

        clone_annot_new=np.zeros(clone_annot_temp.shape,dtype=bool)
        clone_annot_new[cell_id_array_t2]=clone_annot_temp[cell_id_array_t2].A.astype(bool)
        for j in range(clone_N1):
            if (j%50==0):
                #pdb.set_trace()
                logg.hint(f"Inferring early clonal states: current clone id {j}")

            # add back the known clonal states at t2
            #pdb.set_trace()
            temp_t2_idx=clone_annot_temp[cell_id_array_t2_temp][:,j].A.flatten()>0
            clone_annot_new[cell_id_array_t2_temp,j]=temp_t2_idx
            
            # infer the earlier clonal states for each clone
            ### select the early states using the grouped distribution of a clone
            ### clones are not overlapping, and all early states should be attached to clones at the end
            sorted_id_array=np.argsort(map_temp[remaining_ids_t1][:,temp_t2_idx].sum(1).A.flatten(),kind='stable')[::-1]
            sel_id_t1=sorted_id_array[:ave_clone_size_t1]
            temp_t1_idx=np.zeros(len(cell_id_array_t1_temp),dtype=bool)
            temp_t1_idx[np.array(remaining_ids_t1)[sel_id_t1]]=True
            clone_annot_new[cell_id_array_t1_temp,j]=temp_t1_idx
            for kk in np.array(remaining_ids_t1)[sel_id_t1]:
                remaining_ids_t1.remove(kk)
            
            if (len(remaining_ids_t1)==0) and ((j+1)<clone_N1): 
                logg.hint(f'Early break; current clone id: {j+1}')
                break
        ########### end: update clones

        cell_id_array_t1_new=np.nonzero((clone_annot_new.sum(1)>0) & (time_index_t1))[0]
        cell_id_array_t2_new=np.nonzero((clone_annot_new.sum(1)>0) & (time_index_t2))[0]


        clone_annot_new=ssp.csr_matrix(clone_annot_new)
        adata.obsm['X_clone']=clone_annot_new
        adata.uns['multiTime_cell_id_t1']=[cell_id_array_t1_new] # For CoSpar, clonally-related states
        adata.uns['multiTime_cell_id_t2']=[cell_id_array_t2_new]
        adata.uns['clonal_cell_id_t1']=cell_id_array_t1_new # for prepare the similarity matrix with same cell states
        adata.uns['clonal_cell_id_t2']=cell_id_array_t2_new
        adata.uns['proportion']=[1]

        infer_Tmap_from_multitime_clones_private(adata,smooth_array=smooth_array,neighbor_N=CoSpar_KNN,noise_threshold=noise_threshold,
            normalization_mode=normalization_mode,save_subset=True,use_full_Smatrix=use_full_Smatrix,
            trunca_threshold=trunca_threshold,compute_new_Smatrix=compute_new)

        ## update, for the next iteration
        map_temp=adata.uns['transition_map']
        clone_annot_temp=clone_annot_new.copy()


def infer_Tmap_from_one_time_clones(adata_orig,initial_time_points,clonal_time_point,
    initialize_method='OT',OT_epsilon=0.02,OT_dis_KNN=5,OT_cost='SPD',
    HighVar_gene_pctl=85,Clone_update_iter_N=1,normalization_mode=1,
    noise_threshold=0.2,CoSpar_KNN=20,use_full_Smatrix=False,smooth_array=[15,10,5],
    trunca_threshold=0.001,compute_new=False,
    use_fixed_clonesize_t1=False,sort_clone=1,save_subset=True):
    """
    Infer transition map from clones with a single time point

    We iteratively infer transition map between each of the initial 
    time points ['day_1','day_2',...,] and the time point with clonal 
    observation. Given the two time points, after initializing the map 
    by either OT method or HighVar method, we jointly infer the likely 
    initial clonal cells and the transition map between cell states 
    in these two time points.  

    **Summary**
        
    * Parameters relevant for cell state selection:  initial_time_points, 
      clonal_time_point, use_full_Smatrix.

    * Choose the initialization method, and set the corresponding parameters. 

        * 'OT': tend to be more accurate, but not reliable 
          under batch effect. Key parameters: `OT_epsilon, OT_dis_KNN`. 
    
        * 'HighVar':  is robust to batch effect, but not as accurate.
          Key parameter: `HighVar_gene_pctl`.

    * Key parameters relevant for CoSpar itself: `smooth_array, normalization_mode, 
      CoSpar_KNN, noise_threshold, Clone_update_iter_N`.

    Parameters
    ----------
    adata_orig: :class:`~anndata.AnnData` object
        assumed to be preprocessed, can have multiple time points.
    initial_time_points: `list` 
        List of initial time points to be included for the transition map. 
        Like ['day_1','day_2']. Entries consistent with adata.obs['time_info']. 
    clonal_time_point: `str` 
        The time point with clonal observation. Its value should be 
        consistent with adata.obs['time_info']. 
    initialize_method: `str`, optional (default 'OT') 
        Method to initialize the transition map from state information. 
        Choice: {'OT', 'HighVar'}.
    OT_epsilon: `float`, optional (default: 0.02)  
        The entropic regularization, >0, a larger one increases 
        uncertainty of the transition. Relevant when `initialize_method='OT'`.
    OT_dis_KNN: `int`, optional (default: 5)
        Number of nearest neighbors to construct the KNN graph for
        computing the shortest path distance. Relevant when `initialize_method='OT'`. 
    OT_cost: `str`, optional (default: `SPD`), options {'GED','SPD'}
        The cost metric. We provide gene expression distance (GED), and also
        shortest path distance (SPD). GED is much faster, but SPD is more accurate.
        However, cospar is robust to the initialization. 
    HighVar_gene_pctl: `int`, optional (default: 85)
        Genes wht a variability percentile higher than this threshold are marked as 
        highly variable genes for dimension reduction. Range: [0,100]. 
        Relevant when `initialize_method='HighVar'`.
    Clone_update_iter_N: `int`, optional (default: 1)
        Number of iteration for the joint optimization
    normalization_mode: `int`, optional (default: 1)
        Method for normalization. Choice: [0,1]
        0, single-cell normalization
        1, Clone normalization
    smooth_array: `list`, optional (default: [15,10,5])
        List of smooth rounds at each iteration. 
        The n-th entry determines the smooth round for the Smatrix 
        at the n-th iteration. Its length determins the number of
        iteration.  
    CoSpar_KNN: `int`, optional (default: 20)
        the number of neighbors for KNN graph used for computing the similarity matrix.
    trunca_threshold: `float`, optional (default: 0.001)
        We set entries to zero in the computed similarity matrix that 
        are smaller than this threshold. This is to promote the Smatrix sparsity, which
        leads to faster computation, and smaller file size. 
        This threshld should be small, but not too small. 
    noise_threshold: `float`, optional (default: 0.1)
        noise threshold to remove noises in the updated transition map,
        in the range [0,1]
    save_subset: `bool`, optional (default: True)
        If true, save only Smatrix at smooth round [5,10,15,...].
        Else, save Smatrix at each round. 
    use_full_Smatrix: `bool`, optional (default: False)
        use the Smatrix as defined by all cells, whether they are clonally 
        barcoded or not. We sub-sample cell states relevant for downstream 
        analysis from this full Smatrix. This may refine the Smatrix. 
        But will also increase the computation time significantly.
    use_fixed_clonesize_t1: `bool`, optional (default: False)
        If true, fix the number of initial states as the same for all clones
    sort_clone: `int`, optional (default: 1)
        The order to infer initial states for each clone: {1,-1,others}
        1, sort clones by size from small to large
        -1,sort clones by size from large to small
        others, do not sort. 
    compute_new: `bool`, optional (default: False)
        If True, compute everthing (ShortestPathDis,OT_map etc.) from scratch, 
        whether it was computed and saved before or not. Regarding the Smatrix, it is 
        recomputed only when `use_full_Smatrix=False`.

    Returns
    -------
    adata: :class:`~anndata.AnnData` object
        Update adata.obsm['X_clone'] and adata.uns['transition_map'],
        as well as adata.uns['OT_transition_map'] or 
        adata.uns['intraclone_transition_map'], depending on the initialization.
    """


    for xx in initial_time_points:
        if xx not in list(set(adata_orig.obs['time_info'])):
            logg.error(f"the 'initial_time_points' are not valid. Please select from {list(set(adata_orig.obs['time_info']))}")
            return adata_orig

    with_clonal_info=(clonal_time_point in adata_orig.uns['clonal_time_points'])
    if not with_clonal_info:
        logg.warn(f"'clonal_time_point' do not contain clonal information. Please set clonal_time_point to be one of {adata_orig.uns['clonal_time_points']}")
        #logg.info("Consider run ----cs.tmap.CoSpar_NoClonalInfo------")
        logg.warn("Keep running but without clonal information")
        #return adata_orig

    sp_idx=np.zeros(adata_orig.shape[0],dtype=bool)
    time_info_orig=np.array(adata_orig.obs['time_info'])
    all_time_points=initial_time_points+[clonal_time_point]
    label='t'
    for xx in all_time_points:
        id_array=np.nonzero(time_info_orig==xx)[0]
        sp_idx[id_array]=True
        label=label+'*'+str(xx)

    adata=sc.AnnData(adata_orig.X[sp_idx]);
    adata.var_names=adata_orig.var_names
    adata.obsm['X_pca']=adata_orig.obsm['X_pca'][sp_idx]
    adata.obsm['X_umap']=adata_orig.obsm['X_umap'][sp_idx]
    adata.obs['state_info']=pd.Categorical(adata_orig.obs['state_info'][sp_idx])
    adata.obs['time_info']=pd.Categorical(adata_orig.obs['time_info'][sp_idx])
    
    data_des_0=adata_orig.uns['data_des'][0]
    data_des=data_des_0+f'_OneTimeClone_{label}'
    adata.uns['data_des']=[data_des]
    


    clone_annot_orig=adata_orig.obsm['X_clone']        
    clone_annot=clone_annot_orig[sp_idx]
    adata.obsm['X_clone']=clone_annot

    time_info=np.array(adata.obs['time_info'])
    time_index_t2=time_info==clonal_time_point
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
    ini_transition_map=np.zeros((len(Tmap_cell_id_t1),len(Tmap_cell_id_t2)))


    for yy in initial_time_points:
        
        logg.info("-------------------------------New Start--------------------------------------------------")
        logg.info(f"Current time point: {yy}")

        adata_temp=infer_Tmap_from_one_time_clones_twoTime(adata_orig,selected_two_time_points=[yy,clonal_time_point],
            initialize_method=initialize_method,OT_epsilon=OT_epsilon,OT_dis_KNN=OT_dis_KNN,
            OT_cost=OT_cost,HighVar_gene_pctl=HighVar_gene_pctl,
            Clone_update_iter_N=Clone_update_iter_N,normalization_mode=normalization_mode,
            noise_threshold=noise_threshold,CoSpar_KNN=CoSpar_KNN,use_full_Smatrix=use_full_Smatrix,smooth_array=smooth_array,
            trunca_threshold=trunca_threshold,compute_new=compute_new,
            use_fixed_clonesize_t1=use_fixed_clonesize_t1,sort_clone=sort_clone,save_subset=save_subset)

        temp_id_t1=np.nonzero(time_info==yy)[0]
        sp_id_t1=hf.converting_id_from_fullSpace_to_subSpace(temp_id_t1,Tmap_cell_id_t1)[0]
        
        if with_clonal_info:
            transition_map_temp=adata_temp.uns['transition_map'].A
            transition_map[sp_id_t1,:]=transition_map_temp

        if initialize_method=='OT':
            transition_map_ini_temp=adata_temp.uns['OT_transition_map']
        else:
            transition_map_ini_temp=adata_temp.uns['HighVar_transition_map']

        ini_transition_map[sp_id_t1,:]=transition_map_ini_temp.A

    if with_clonal_info:
        adata.uns['transition_map']=ssp.csr_matrix(transition_map)
    
    if initialize_method=='OT':
        adata.uns['OT_transition_map']=ssp.csr_matrix(ini_transition_map)
    else:
        adata.uns['HighVar_transition_map']=ssp.csr_matrix(ini_transition_map)


    return adata


def infer_Tmap_from_one_time_clones_twoTime(adata_orig,selected_two_time_points=['1','2'],
    initialize_method='OT',OT_epsilon=0.02,OT_dis_KNN=5,OT_cost='SPD',HighVar_gene_pctl=80,
    Clone_update_iter_N=1,normalization_mode=1,noise_threshold=0.2,CoSpar_KNN=20,
    use_full_Smatrix=False,smooth_array=[15,10,5],
    trunca_threshold=0.001,compute_new=True,use_fixed_clonesize_t1=False,
    sort_clone=1,save_subset=True):
    """
    Infer transition map from clones with a single time point

    It is the same as :func:`.infer_Tmap_from_one_time_clones`, except that
    it assumes that the input adata_orig has only two time points. 
    """

    time_info_orig=np.array(adata_orig.obs['time_info'])
    sort_time_point=np.sort(list(set(time_info_orig)))
    N_valid_time=np.sum(np.in1d(sort_time_point,selected_two_time_points))
    if (N_valid_time!=2): 
        logg.error(f"Must select only two time points among the list {sort_time_point}")
        #The second time point in this list (not necessarily later time point) is assumed to have clonal data.")
    else:
        ####################################
        
        logg.info("-----------Pre-processing and sub-sampling cells------------")
        # select cells from the two time points, and sub-sampling, create the new adata object with these cell states
        sp_idx=(time_info_orig==selected_two_time_points[0]) | (time_info_orig==selected_two_time_points[1])
  
        adata=sc.AnnData(adata_orig.X[sp_idx]);
        adata.var_names=adata_orig.var_names
        adata.obsm['X_pca']=adata_orig.obsm['X_pca'][sp_idx]
        adata.obsm['X_umap']=adata_orig.obsm['X_umap'][sp_idx]
        adata.obs['state_info']=pd.Categorical(adata_orig.obs['state_info'][sp_idx])
        adata.obs['time_info']=pd.Categorical(adata_orig.obs['time_info'][sp_idx])
        
        data_des_0=adata_orig.uns['data_des'][0]
        data_des=data_des_0+f'_OneTimeClone_t*{selected_two_time_points[0]}*{selected_two_time_points[1]}'
        adata.uns['data_des']=[data_des]
        


        clone_annot_orig=adata_orig.obsm['X_clone']        
        barcode_id=np.nonzero(clone_annot_orig[sp_idx].A.sum(0).flatten()>0)[0]
        clone_annot=clone_annot_orig[sp_idx][:,barcode_id]
        adata.obsm['X_clone']=clone_annot

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
        if use_full_Smatrix: 

            temp_str='0'+str(trunca_threshold)[2:]
            round_of_smooth=np.max(smooth_array)

            similarity_file_name=f'{data_path}/Similarity_matrix_with_all_cell_states_kNN{CoSpar_KNN}_Truncate{temp_str}_v0_fullsimilarity{use_full_Smatrix}'
            if not (os.path.exists(similarity_file_name+f'_SM{round_of_smooth}.npz') and (not compute_new)):
                similarity_matrix_full=generate_similarity_matrix(adata_orig,similarity_file_name,round_of_smooth=round_of_smooth,
                            neighbor_N=CoSpar_KNN,truncation_threshold=trunca_threshold,save_subset=save_subset,compute_new_Smatrix=compute_new)

        

        if initialize_method=='OT':
            
            logg.info("----------------")
            logg.info("Step 1: Use OT method for initialization")

            compute_custom_OT_transition_map(adata,OT_epsilon=OT_epsilon,OT_cost=OT_cost,OT_dis_KNN=OT_dis_KNN,compute_new=compute_new)
            OT_transition_map=adata.uns['OT_transition_map']
            initialized_map=OT_transition_map

            
        else:
            
            logg.info("----------------")
            logg.info("Step 1: Use highly variable genes to construct pseudo-clones, and apply CoSpar to generate initialized map!")

            t=time.time()
            Tmap_from_highly_variable_genes(adata,min_counts=3,min_cells=3,min_gene_vscore_pctl=HighVar_gene_pctl,noise_threshold=noise_threshold,neighbor_N=CoSpar_KNN,
                normalization_mode=normalization_mode,use_full_Smatrix=use_full_Smatrix,smooth_array=smooth_array,trunca_threshold=trunca_threshold,
                compute_new_Smatrix=compute_new)

            HighVar_transition_map=adata.uns['HighVar_transition_map']
            initialized_map=HighVar_transition_map

            
            logg.info(f"Finishing computing transport map from highly variable genes, used time {time.time()-t}")


        ########### Jointly optimize the transition map and the initial clonal states
        if selected_two_time_points[1] in adata_orig.uns['clonal_time_points']:
        
            logg.info("----------------")
            logg.info("Step 2: Jointly optimize the transition map and the initial clonal states!")

            t=time.time()

            infer_Tmap_from_one_time_clones_private(adata,initialized_map,Clone_update_iter_N=Clone_update_iter_N,normalization_mode=normalization_mode,noise_threshold=noise_threshold,
                CoSpar_KNN=CoSpar_KNN,use_full_Smatrix=use_full_Smatrix,smooth_array=smooth_array,trunca_threshold=trunca_threshold,
                compute_new=compute_new,use_fixed_clonesize_t1=use_fixed_clonesize_t1,sort_clone=sort_clone)


            
            logg.info(f"Finishing computing transport map from CoSpar using inferred clonal data, used time {time.time()-t}")
        else:
            logg.warn("No clonal information available. Skip the joint optimization of clone and scRNAseq data")


        return adata

def infer_naive_Tmap(adata):
    """
    Compute transition map using only the lineage information

    We simply average transitions across all clones, assuming that
    the intra-clone transition is uniform within the same clone. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        It should have been preprocessed by :func:`.select_time_points`

    Returns
    -------
    Update `adata` with the attributes adata.uns['naive_transition_map']
    """

    cell_id_t2=adata.uns['Tmap_cell_id_t2']
    cell_id_t1=adata.uns['Tmap_cell_id_t1']
    clone_annot=adata.obsm['X_clone']

    naive_map=clone_annot[cell_id_t1]*clone_annot[cell_id_t2].T
    naive_map=naive_map.astype(int)
    adata.uns['naive_transition_map']=ssp.csr_matrix(naive_map)

def infer_weinreb_Tmap(adata):
    """
    Compute transition map using only the lineage information

    Find uni-potent clones, then compute the transition map by simply 
    averaging across all clonal transitions as in :func:`.infer_naive_Tmap`.
    The intra-clone transition is uniform within the same clone. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        It should have been preprocessed by :func:`.select_time_points`

    Returns
    -------
    Update `adata` with the attributes adata.uns['weinreb_transition_map']
    """

    logg.info("This method works when there are only time points and all datasets")
    cell_id_t2=adata.uns['Tmap_cell_id_t2']
    cell_id_t1=adata.uns['Tmap_cell_id_t1']
    clone_annot=adata.obsm['X_clone']
    state_annote=np.array(adata.obs['state_info'])

    fate_array=list(set(state_annote))

    # if include_undiff:
    #     fate_array=['Ccr7_DC',  'Mast', 'Meg', 'pDC', 'Eos', 'Lymphoid', 'Erythroid', 'Baso', 'Neutrophil', 'Monocyte','undiff','Neu_Mon']
    # else:
    #     fate_array=['Ccr7_DC',  'Mast', 'Meg', 'pDC', 'Eos', 'Lymphoid', 'Erythroid', 'Baso', 'Neutrophil', 'Monocyte','Neu_Mon']
    potential_vector_clone, fate_entropy_clone=hf.compute_state_potential(clone_annot[cell_id_t2].T,state_annote[cell_id_t2],fate_array,fate_count=True)


    sel_unipotent_clone_id=np.array(list(set(np.nonzero(fate_entropy_clone==1)[0])))
    clone_annot_unipotent=clone_annot[:,sel_unipotent_clone_id]
    weinreb_map=clone_annot_unipotent[cell_id_t1]*clone_annot_unipotent[cell_id_t2].T
    weinreb_map=weinreb_map.astype(int)
    logg.info(f"Used clone fraction {len(sel_unipotent_clone_id)/clone_annot.shape[1]}")
    adata.uns['weinreb_transition_map']=ssp.csr_matrix(weinreb_map)
