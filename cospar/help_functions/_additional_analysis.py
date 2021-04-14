import numpy as np
import scipy
import scipy.stats
from scipy import stats
import scipy.sparse as ssp
import pandas as pd
from .. import settings
from .. import logging as logg
import time
import ot.bregman as otb
from ._help_functions_CoSpar import *
from matplotlib import pyplot as plt
import seaborn as sns
from .. import plotting as pl


"""
This is not a necessary component of CoSpar. It requires additional packages. 
When publishing, you can change the __ini__ setting to remove this part. 
Making it too fat increase the work for maintenance. 
"""


def Wasserstein_distance_private(prob_t0,prob_t1,full_cost_matrix,
    OT_epsilon=0.05,OT_stopThr=10**(-8),OT_max_iter=1000):
    """
    Compute symmetric Wasserstein distance between two distributions.

    Parameters
    ----------
    prob_t0: `np.array`, (n_1,)
        Distribution on initial state space.
    prob_t1: `np.array`, (n_2,)
         Distribution on later state space
    full_cost_matrix: `np.array`, shape (n_1, n_2)
        A cost matrix to map all initial states to all later states. This is a full matrix.
    OT_epsilon: `float`, optional (default: 0.05)
        Entropic regularization parameter to compute the optional 
        transport map from target to ref. 
    OT_stopThr: `float`, optional (default: 10**(-8))
        The stop thresholding for computing the transport map. 
    OT_max_iter: `float`, optional (default: 1000)
        The maximum number of iteration for computing the transport map. 

    Returns
    ------- 
    A vector for [forward_distance, backward_distance, the average]
    """
 
    # normalized distribution
    prob_t0=np.array(prob_t0)
    prob_t1=np.array(prob_t1)
    sp_id_t0=np.nonzero(prob_t0>0)[0]
    sp_id_t1=np.nonzero(prob_t1>0)[0]

    resol=10**(-10)
    input_t0=prob_t0[sp_id_t0]/(resol+np.sum(prob_t0[sp_id_t0]));
    input_t1=prob_t1[sp_id_t1]/(resol+np.sum(prob_t1[sp_id_t1]));

    logg.info("Compute forward transition map")
    sp_cost_matrix_t0t1=full_cost_matrix[sp_id_t0][:,sp_id_t1]
    OT_transition_map_t0t1=otb.sinkhorn_stabilized(input_t0,input_t1,sp_cost_matrix_t0t1,OT_epsilon,numItermax=OT_max_iter,stopThr=OT_stopThr)

    # if more than 10% of the prediction is less than 50% accurate, then we declare it a failure
    flag_1=np.sum(OT_transition_map_t0t1.sum(1)<0.5*input_t0)>0.1*len(input_t0)
    flag_2=np.sum(OT_transition_map_t0t1.sum(1)>2*input_t0)>0.1*len(input_t0)

    if not (flag_1 or flag_2):
        Wass_dis=np.sum(OT_transition_map_t0t1*sp_cost_matrix_t0t1)
        return Wass_dis
    else:
        logg.error("Forward transition map construction failed")
        logg.info("Compute backward transition map instead")
        sp_cost_matrix_t1t0=full_cost_matrix[sp_id_t1][:,sp_id_t0]
        OT_transition_map_t1t0=otb.sinkhorn_stabilized(input_t1,input_t0,sp_cost_matrix_t1t0,OT_epsilon,numItermax=OT_max_iter,stopThr=OT_stopThr)

        flag_3=np.sum(OT_transition_map_t1t0.sum(1)<0.5*input_t1)>0.1*len(input_t1)
        flag_4=np.sum(OT_transition_map_t1t0.sum(1)>2*input_t1)>0.1*len(input_t1)
        
        Wass_dis=np.sum(OT_transition_map_t1t0*sp_cost_matrix_t1t0)
        
        if not (flag_3 or flag_4):
            return Wass_dis
        else:
            logg.error("Backward transition map construction failed")
            return None



def Wasserstein_distance(adata,group_A,group_B,OT_dis_KNN=5,OT_epsilon=0.05,compute_new=False,show_groups=True):
    """
    Compute Wasserstein between two populations

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    OT_epsilon: `float`, optional (default: 0.02)  
        The entropic regularization, >0. A larger value increases 
        uncertainty of the transition. 
    OT_dis_KNN: `int`, optional (default: 5)
        Number of nearest neighbors to construct the KNN graph for
        computing the shortest path distance. 
    OT_epsilon: `float`, optional (default: 0.05)
        Entropic regularization parameter to compute the optional 
        transport map from target to ref.  
    compute_new: `bool`, optional (default: False)
        If True, compute OT_map and also the shortest path distance from scratch, 
        whether it was computed and saved before or not.
    show_groups: `bool`, optional (default: True)
        Plot each group, and overlay them on top of each other.

    Returns
    -------
    A vector for [forward_distance, backward_distance, the average]
    """

    data_des=adata.uns['data_des'][-1]
    data_path=settings.data_path
    figure_path=settings.figure_path
    group_A=np.array(group_A).astype(float)
    group_B=np.array(group_B).astype(float)


    if (len(group_A)!=len(group_B)) or (len(group_A)!=adata.shape[0]):
        logg.error("Size mismatch between group_A, group_B, and adata.shape[0].")
        return None
    else:
        if (np.sum(group_A>0)==0) or  (np.sum(group_B>0)==0):
            logg.error("No cells selected.")
            return None   

        if show_groups:
            X_emb=adata.obsm['X_emb']
            x_emb=X_emb[:,0]
            y_emb=X_emb[:,1]

            fig_width=settings.fig_width; fig_height=settings.fig_height; point_size=settings.fig_point_size
            fig=plt.figure(figsize=(3*fig_width,fig_height));
            ax=plt.subplot(1,3,1)
            pl.customized_embedding(x_emb,y_emb,group_A,
                 point_size=point_size,set_lim=False,ax=ax,title='Group A')
                        
            ax=plt.subplot(1,3,2)
            pl.customized_embedding(x_emb,y_emb,group_B,
                 point_size=point_size,set_lim=False,ax=ax,title='Group B')


            resol=10**(-10)
            group_A=group_A/(resol+np.max(group_A))
            group_B=group_B/(resol+np.max(group_B))
            vector_array=(group_A-group_B)
            ax=plt.subplot(1,3,3)
            new_idx=np.argsort(abs(vector_array))
            pl.customized_embedding(x_emb[new_idx],y_emb[new_idx],
                                vector_array[new_idx],title='Overlay',
                                point_size=point_size,set_lim=False,ax=ax,color_map=plt.cm.bwr,order_points=False)

            plt.tight_layout()
            fig.savefig(f'{figure_path}/{data_des}_Wass_dis_group_overlay.{settings.file_format_figs}')



        SPD_file_name=f'{data_path}/{data_des}_ShortestPathDistanceMatrix_KNN{OT_dis_KNN}.npy'
        if os.path.exists(SPD_file_name) and (not compute_new):

            logg.info("Load pre-computed shortest path distance matrix")
            OT_cost_matrix=np.load(SPD_file_name)

        else:

            logg.info("Compute new shortest path distance matrix")
            t=time.time()       
            ShortPath_dis=compute_shortest_path_distance(adata,num_neighbors_target=OT_dis_KNN,mode='distances',method='others',normalize=False)
            
            # we do not want normalization here
            OT_cost_matrix=ShortPath_dis 

            np.save(SPD_file_name,OT_cost_matrix) # This is not a sparse matrix at all. 

            logg.info(f"Finishing computing shortest-path distance, used time {time.time()-t}")

        Wass_dis=Wasserstein_distance_private(group_A,group_B,OT_cost_matrix,OT_epsilon=OT_epsilon)

        return Wass_dis


########################

# Miscellaneous analysis

########################


def assess_fate_prediction_by_correlation_v0(adata,ground_truth,selected_time_points,plot_style='boxplot',figure_index='',show_groups=True):
    """
    Assess biary fate prediction by correlation 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        It should be run through cs.pl.binary_fate_bias
    ground_truth: `np.array`
        A vector of expected fate bias for each cell in the 
        full space corresponding to adata_orig. We expect bias to fate A has value (0,1],
        and bias towards fate B has value [-1,0). Cells with value zeros are 
        discarded before computing correlation. 
    selected_time_points: `list`
        A list of selected time points for making the comparison.
    plot_style: `string`
        Plot used to visualize the results. It can be {'boxplot','scatter'}. 
    figure_index: `string` optional (default: '')
        String index for annotate filename for saved figures. Used to distinuigh plots from different conditions. 
    show_groups: `bool`, optional (default: True)
        Plot each group.

    Returns
    -------
    correlation with ground truth at selected time points.
    """

    data_des=adata.uns['data_des'][-1]
    if 'binary_fate_bias' not in adata.uns.keys():
        logg.error("Binary fate bias not computed yet! Please run cs.pl.binary_fate_bias first!")
        return None

    else:
        fig_width=settings.fig_width; fig_height=settings.fig_height; point_size=settings.fig_point_size
        time_info=np.array(adata.obs['time_info'])
        sp_idx_time=selecting_cells_by_time_points(time_info,selected_time_points)
        
        ground_truth_sp=np.array(ground_truth)[adata.uns['sp_idx']]
        binary_fate_bias=adata.uns['binary_fate_bias'][1]
        sel_index= (abs(ground_truth_sp)>0) & sp_idx_time
        ground_truth_sp=(1+ground_truth_sp)/2 # transform the value to 0 and 1.

        if np.sum(sel_index)==0:
            logg.error("No cells selected.")
            return None
        else:
            corr=np.corrcoef(ground_truth_sp[sel_index],binary_fate_bias[sel_index])[0,1]
            if np.isnan(corr):
                logg.error("Correlation is NaN.")
                return None   
            else:
                corr=round(100*corr)/100

                if show_groups:
                    X_emb=adata.obsm['X_emb']
                    x_emb=X_emb[:,0]
                    y_emb=X_emb[:,1]

                    fig_width=settings.fig_width; fig_height=settings.fig_height; point_size=settings.fig_point_size
                    fig=plt.figure(figsize=(2*fig_width,fig_height));
                    ax=plt.subplot(1,2,1)
                    new_idx=np.argsort(abs(ground_truth_sp-0.5))
                    pl.customized_embedding(x_emb[new_idx],y_emb[new_idx],ground_truth_sp[new_idx],
                         point_size=point_size,set_lim=False,color_map=plt.cm.bwr,ax=ax,title='Group A',order_points=False)
                                
                    ax=plt.subplot(1,2,2)
                    new_idx=np.argsort(abs(binary_fate_bias-0.5))
                    pl.customized_embedding(x_emb[new_idx],y_emb[new_idx],binary_fate_bias[new_idx],
                         point_size=point_size,set_lim=False,color_map=plt.cm.bwr,ax=ax,title='Group B',order_points=False)
                
                if plot_style=='boxplot':
                    data_frame=pd.DataFrame({'Ref':ground_truth_sp[sel_index],'Prediction':binary_fate_bias[sel_index]})
                    fig=plt.figure(figsize=(fig_width,fig_height))
                    ax=plt.subplot(1,1,1)
                    sns.violinplot(x="Ref", y="Prediction", data=data_frame,ax=ax,color='red')
                    sns.set(style="white")
                    #ax.set_ylim([-0.5,20])
                    ax.set_xlabel('Reference fate bias')
                    ax.set_ylabel('Predicted fate bias')
                    plt.tight_layout()
                    ax.set_title(f"Corr={corr}, {figure_index}")
                    fig.savefig(f'{settings.figure_path}/{data_des}_{figure_index}_reference_prediction_box.{settings.file_format_figs}')

                if plot_style=='scatter':
                    fig=plt.figure(figsize=(fig_width,fig_height))
                    ax=plt.subplot(1,1,1)
                    ax.plot(ground_truth_sp[sel_index],binary_fate_bias[sel_index],'*r')
                    ax.set_xlabel('Reference fate bias')
                    ax.set_ylabel('Predicted fate bias')
                    plt.tight_layout()
                    ax.set_title(f"Corr={corr}, {figure_index}")
                    fig.savefig(f'{settings.figure_path}/{data_des}_{figure_index}_reference_prediction_scatter.{settings.file_format_figs}')

                return corr


def assess_fate_prediction_by_correlation(adata,expect_vector,predict_vecotr,selected_time_points,plot_style='scatter',figure_index='',show_groups=True,mask=None,remove_neutral_ref=True,background=False,vmax=1,vmin=0):
    """
    Assess biary fate prediction by correlation 

    The expect_vector and predict_vecotr are of the same length, in the range (0,1). 
    We only use cells with non-neutral bias (which is 0.5) to compute the fate correlation. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        It should be run through pl.prediction
    expect_vector: `np.array`
        A vector of expected fate bias for each cell. The bias range is (0,1),
        with 0.5 being neutral. The neutral cells are discarded before computing correlation. 
    predict_vector: `np.array`
        A vector of predicted fate bias for each cell. The bias range is (0,1),
        with 0.5 being neutral. 
    selected_time_points: `list`
        A list of selected time points for making the comparison.
    plot_style: `string`, optional (default: 'scatter')
        Plot used to visualize the results. It can be {'boxplot','scatter'}. 
    figure_index: `string` optional (default: '')
        String index for annotate filename for saved figures. Used to distinuigh plots from different conditions. 
    show_groups: `bool`, optional (default: True)
        Plot each group.
    mask: `np.array`, optional (default: None)
        A boolean array to define which cells are used for computing correlation.
    remove_neutral_ref: `bool`, optional (default: True)
        Remove neutral reference states before computing the correlation.
    background: `bool`, optional (default: False)
        Show background at given time points.
    vmax: `float`, optional (default: 1)
        Maximum value to plot. 
    vmin: `float`, optional (default: 0)
        Minimum value to plot.

    Returns
    -------
    correlation between expect_vector and predict_vector at selected time points.
    """


    # Copy the vector to avoid change the original vector. 
    reference=expect_vector.copy()
    prediction=predict_vecotr.copy()
    data_des=adata.uns['data_des'][-1]
    if (len(reference)!=len(prediction)) or (len(reference)!=adata.shape[0]):
        logg.error("Size mismatch between reference, prediction, and adata.shape[0].")
        return None
    else:        
        fig_width=settings.fig_width; fig_height=settings.fig_height; point_size=settings.fig_point_size
        time_info=np.array(adata.obs['time_info'])
        sp_idx_time=selecting_cells_by_time_points(time_info,selected_time_points)
        
        if (mask is not None) and (len(mask)==len(prediction)):
            mask=np.array(mask)
            reference[~mask]=0.5
            prediction[~mask]=0.5
        
        if remove_neutral_ref:
            logg.info("Remove neutral states in the reference before computing correlation.")
            sel_index= (abs(reference-0.5)>0) & sp_idx_time
        else:
            sel_index= sp_idx_time

        if np.sum(sel_index)==0:
            logg.error("No cells selected.")
            return None
        else:
            
            reference_sp=reference[sel_index]
            prediction_sp=prediction[sel_index]
            corr=np.corrcoef(reference_sp,prediction_sp)[0,1]
            if np.isnan(corr):
                logg.error("Correlation is NaN.")
                return None   
            else:
                corr=round(100*corr)/100

                if show_groups:
                    X_emb=adata.obsm['X_emb']
                    x_emb=X_emb[:,0]
                    y_emb=X_emb[:,1]

                   
                    fig=plt.figure(figsize=(fig_width,fig_height));
                    ax=plt.subplot(1,1,1)
                    vector_temp=reference_sp
                    new_idx=np.argsort(abs(vector_temp-0.5))
                    if background:

                        if mask is not None:
                            sel_index_1= mask & sp_idx_time
                        else:
                            sel_index_1= sp_idx_time

                        pl.customized_embedding(x_emb[sel_index_1],y_emb[sel_index_1],np.ones(np.sum(sel_index_1)),
                             point_size=point_size,set_lim=False,ax=ax,order_points=False)

                    pl.customized_embedding(x_emb[sel_index][new_idx],y_emb[sel_index][new_idx],vector_temp[new_idx],
                         point_size=point_size,set_lim=False,color_map=plt.cm.bwr,ax=ax,title='Reference',order_points=False,vmax=vmax,vmin=vmin)
                    fig.savefig(f'{settings.figure_path}/{data_des}_{figure_index}_reference.{settings.file_format_figs}')

                    fig=plt.figure(figsize=(fig_width,fig_height));
                    ax=plt.subplot(1,1,1)
                    vector_temp=prediction_sp
                    new_idx=np.argsort(abs(vector_temp-0.5))
                    if background:

                        if mask is not None:
                            sel_index_1= mask & sp_idx_time
                        else:
                            sel_index_1= sp_idx_time

                        pl.customized_embedding(x_emb[sel_index_1],y_emb[sel_index_1],np.ones(np.sum(sel_index_1)),
                             point_size=point_size,set_lim=False,ax=ax,order_points=False)

                    pl.customized_embedding(x_emb[sel_index][new_idx],y_emb[sel_index][new_idx],vector_temp[new_idx],
                         point_size=point_size,set_lim=False,color_map=plt.cm.bwr,ax=ax,title='Prediction',order_points=False,vmax=vmax,vmin=vmin)
                    fig.savefig(f'{settings.figure_path}/{data_des}_{figure_index}_prediction.{settings.file_format_figs}')

                if plot_style=='boxplot':
                    reference_sp[reference_sp<0.01]=0
                    reference_sp[reference_sp>0.99]=1

                    data_frame=pd.DataFrame({'Ref':reference_sp,'Prediction':prediction_sp})
                    fig=plt.figure(figsize=(fig_width,fig_height))
                    ax=plt.subplot(1,1,1)
                    sns.violinplot(x="Ref", y="Prediction", data=data_frame,ax=ax,color='red')
                    sns.set(style="white")
                    #ax.set_ylim([-0.5,20])
                    ax.set_xlabel('Reference fate bias')
                    ax.set_ylabel('Predicted fate bias')
                    plt.tight_layout()
                    ax.set_title(f"Corr={corr}, {figure_index}")
                    fig.savefig(f'{settings.figure_path}/{data_des}_{figure_index}_reference_prediction_box.{settings.file_format_figs}')

                if plot_style=='scatter':
                    fig=plt.figure(figsize=(fig_width,fig_height))
                    ax=plt.subplot(1,1,1)
                    ax.plot(reference_sp,prediction_sp,'*r')
                    ax.set_xlabel('Reference fate bias')
                    ax.set_ylabel('Predicted fate bias')
                    plt.tight_layout()
                    ax.set_title(f"Corr={corr}, {figure_index}")
                    fig.savefig(f'{settings.figure_path}/{data_des}_{figure_index}_reference_prediction_scatter.{settings.file_format_figs}')

                return corr




####### plot heat maps for genes
def heatmap_v1(figure_path, data_matrix, variable_names_x,variable_names_y,int_seed=10,
    data_des='',log_transform=False,color_map=plt.cm.Reds,vmin=None,vmax=None,fig_width=4,fig_height=6,
    color_bar=True):
    """
    Plot ordered heat map of data_matrix matrix.

    Parameters
    ----------
    figure_path: `str`
        path to save figures
    data_matrix: `np.array`
        A matrix whose columns should match variable_names 
    variable_names: `list`
        List of variable names
    color_bar_label: `str`, optional (default: 'cov')
        Color bar label
    data_des: `str`, optional (default: '')
        String to distinguish different saved objects.
    int_seed: `int`, optional (default: 10)
        Seed to initialize the plt.figure object (to avoid 
        plotting on existing object).
    log_transform: `bool`, optional (default: False)
        If true, perform a log transform. This is needed when the data 
        matrix has entries varying by several order of magnitude. 
    """

    #o = get_hierch_order(data_matrix)
    #o1 = get_hierch_order(data_matrix.T)
    
    plt.figure(int_seed)
    
    if log_transform:
        plt.imshow(np.log(data_matrix+1)/np.log(10), aspect='auto',cmap=color_map, vmin=vmin,vmax=vmax)
    else:
        plt.imshow(data_matrix, aspect='auto',cmap=color_map, vmax=vmax,vmin=vmin)
        
       
    variable_names_x=list(variable_names_x)
    variable_names_y=list(variable_names_y)
    if variable_names_x=='':
        plt.xticks([])
    else:
        plt.xticks(np.arange(data_matrix.shape[1])+.4, variable_names_x, rotation=70, ha='right')
        
    if variable_names_y=='':
        plt.yticks([])
    else:
        plt.yticks(np.arange(data_matrix.shape[0]), variable_names_y, rotation=0, ha='right')

    if color_bar:
        cbar = plt.colorbar()
        cbar.set_label('Z-Score', rotation=270, labelpad=20)
    plt.gcf().set_size_inches((fig_width,fig_height))
    plt.tight_layout()
    plt.savefig(figure_path+f'/{data_des}_data_matrix.{settings.file_format_figs}')



def gene_expression_heat_map(adata, state_info, gene_list,selected_fates,rename_selected_fates=None,color_bar=False,method='zscore',fig_width=6,fig_height=3,horizontal='True',log_transform=False,vmin=None,vmax=None):
    """
    Plot heatmap of gene expression within given clusters.
    
    The gene expression can be the relative value or zscore, depending on method {'zscore','Relative'}
    """

    
    
    mega_cluster_list,valid_fate_list,fate_array_flat,sel_index_list=analyze_selected_fates(selected_fates,state_info)
    gene_full=np.array(adata.var_names)
    gene_list=np.array(gene_list)
    sel_idx=np.in1d(gene_full,gene_list)
    valid_sel_idx=np.in1d(gene_list,gene_full)
    
    if np.sum(valid_sel_idx)>0:
        cleaned_gene_list=gene_list[valid_sel_idx]
        if np.sum(valid_sel_idx)<len(gene_list):
            invalid_gene_list=gene_list[~valid_sel_idx]
            print(f"These are invalid gene names: {invalid_gene_list}")
    else:
        print("No valid genes selected.")
    gene_expression_matrix=np.zeros((len(mega_cluster_list),len(cleaned_gene_list)))
    
    X=adata.X
    resol=10**(-10)
    
    if method=='zscore':
        logg.info("Using zscore (range: [-2,2], or [-1,1]")
    else:
        logg.info("Using relative gene expression. Range [0,1]")

    for k,temp in enumerate(cleaned_gene_list):
        temp_id=np.nonzero(gene_full==temp)[0][0]
        temp_vector=np.zeros(len(sel_index_list))
        for j,temp_idx in enumerate(sel_index_list):
            temp_vector[j]=np.mean(X[temp_idx,temp_id])
        
        if method=='zscore':
            z_score=stats.zscore(temp_vector)
            gene_expression_matrix[:,k]=z_score
        else:
            temp_vector=(temp_vector+resol)/(resol+np.sum(temp_vector))
            gene_expression_matrix[:,k]=temp_vector
        
        
    if (rename_selected_fates is None) or (len(rename_selected_fates) != len(mega_cluster_list)):
        rename_selected_fates=mega_cluster_list
        
    if horizontal:
        heatmap_v1(settings.figure_path, gene_expression_matrix, cleaned_gene_list,rename_selected_fates,int_seed=10,
        data_des='',log_transform=False,color_map=plt.cm.coolwarm,fig_width=fig_width,fig_height=fig_height,
        color_bar=color_bar,vmin=vmin,vmax=vmax)
    else:
        heatmap_v1(settings.figure_path, gene_expression_matrix.T,rename_selected_fates, cleaned_gene_list,int_seed=10,
        data_des='',log_transform=False,color_map=plt.cm.coolwarm,fig_width=fig_height,fig_height=fig_width,
        color_bar=color_bar,vmin=vmin,vmax=vmax)
    
    return gene_expression_matrix
    



# ####### plot heat maps for genes
# def heatmap_v1(figure_path, data_matrix, variable_names_x,variable_names_y,int_seed=10,
#     data_des='',log_transform=False,color_map=plt.cm.Reds,vmin=None,vmax=None,fig_width=4,fig_height=6,
#     color_bar=True):
#     """
#     Plot ordered heat map of data_matrix matrix.

#     Parameters
#     ----------
#     figure_path: `str`
#         path to save figures
#     data_matrix: `np.array`
#         A matrix whose columns should match variable_names 
#     variable_names: `list`
#         List of variable names
#     color_bar_label: `str`, optional (default: 'cov')
#         Color bar label
#     data_des: `str`, optional (default: '')
#         String to distinguish different saved objects.
#     int_seed: `int`, optional (default: 10)
#         Seed to initialize the plt.figure object (to avoid 
#         plotting on existing object).
#     log_transform: `bool`, optional (default: False)
#         If true, perform a log transform. This is needed when the data 
#         matrix has entries varying by several order of magnitude. 
#     """

#     #o = get_hierch_order(data_matrix)
#     #o1 = get_hierch_order(data_matrix.T)
    
#     plt.figure(int_seed)
    
#     if log_transform:
#         plt.imshow(np.log(data_matrix+1)/np.log(10), aspect='auto',cmap=color_map, vmin=vmin,vmax=vmax)
#     else:
#         plt.imshow(data_matrix, aspect='auto',cmap=color_map, vmax=vmax,vmin=vmin)
        
       
#     variable_names_x=list(variable_names_x)
#     variable_names_y=list(variable_names_y)
#     if variable_names_x=='':
#         plt.xticks([])
#     else:
#         plt.xticks(np.arange(data_matrix.shape[1])+.4, variable_names_x, rotation=70, ha='right')
        
#     if variable_names_y=='':
#         plt.yticks([])
#     else:
#         plt.yticks(np.arange(data_matrix.shape[0]), variable_names_y, rotation=0, ha='right')

#     if color_bar:
#         cbar = plt.colorbar()
#         cbar.set_label('Number of barcodes (log10)', rotation=270, labelpad=20)
#     plt.gcf().set_size_inches((fig_width,fig_height))
#     plt.tight_layout()
#     plt.savefig(figure_path+f'/{data_des}_data_matrix.{settings.file_format_figs}')



# def gene_expression_heat_map(adata, state_info, gene_list,selected_fates,rename_selected_fates=None,color_bar=False,fig_width=6,fig_height=3,horizontal='True',log_transform=False,vmin=None,vmax=None):
#     """
#     Cacluate the gene expression Z-score of each gene within given clusters
#     """


#     mega_cluster_list,valid_fate_list,fate_array_flat,sel_index_list=analyze_selected_fates(selected_fates,state_info)
#     gene_full=np.array(adata.var_names)
#     gene_list=np.array(gene_list)
#     sel_idx=np.in1d(gene_full,gene_list)
#     valid_sel_idx=np.in1d(gene_list,gene_full)
    
#     if np.sum(valid_sel_idx)>0:
#         cleaned_gene_list=gene_list[valid_sel_idx]
#         if np.sum(valid_sel_idx)<len(gene_list):
#             invalid_gene_list=gene_list[~valid_sel_idx]
#             print(f"These are invalid gene names: {invalid_gene_list}")
#     else:
#         print("No valid genes selected.")
#     gene_expression_matrix=np.zeros((len(mega_cluster_list),len(cleaned_gene_list)))
    
#     X=adata.X
#     resol=10**(-10)
    
#     for k,temp in enumerate(cleaned_gene_list):
#         temp_id=np.nonzero(gene_full==temp)[0][0]
#         temp_vector=np.zeros(len(sel_index_list))
#         for j,temp_idx in enumerate(sel_index_list):
#             temp_vector[j]=np.mean(X[temp_idx,temp_id])
        
#         z_score=stats.zscore(temp_vector)
#         #temp_vector=(temp_vector+resol)/(resol+np.sum(temp_vector))
#         #gene_expression_matrix[:,k]=temp_vector
#         gene_expression_matrix[:,k]=z_score
        
        
#     if (rename_selected_fates is None) or (len(rename_selected_fates) != len(mega_cluster_list)):
#         rename_selected_fates=mega_cluster_list
        
#     if horizontal:
#         heatmap_v1(settings.figure_path, gene_expression_matrix, cleaned_gene_list,rename_selected_fates,int_seed=10,
#         data_des='',log_transform=False,color_map=plt.cm.coolwarm,fig_width=fig_width,fig_height=fig_height,
#         color_bar=color_bar,vmin=vmin,vmax=vmax)
#     else:
#         heatmap_v1(settings.figure_path, gene_expression_matrix.T,rename_selected_fates, cleaned_gene_list,int_seed=10,
#         data_des='',log_transform=False,color_map=plt.cm.coolwarm,fig_width=fig_height,fig_height=fig_width,
#         color_bar=color_bar,vmin=vmin,vmax=vmax)
    
#     return gene_expression_matrix
    

