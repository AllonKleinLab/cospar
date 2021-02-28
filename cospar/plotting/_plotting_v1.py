import numpy as np
import time
from plotnine import *  
from sklearn import manifold
import pandas as pd
import scanpy as sc
import os
import scipy.sparse as ssp
from .. import help_functions as hf
from matplotlib import pyplot as plt
from .. import settings
from .. import logging as logg

####################

## General

####################
    

def darken_cmap(cmap, scale_factor):
    """
    Generate a gradient color map for plotting.
    """

    cdat = np.zeros((cmap.N, 4))
    for ii in range(cdat.shape[0]):
        curcol = cmap(ii)
        cdat[ii, 0] = curcol[0] * scale_factor
        cdat[ii, 1] = curcol[1] * scale_factor
        cdat[ii, 2] = curcol[2] * scale_factor
        cdat[ii, 3] = 1
    cmap = cmap.from_list(cmap.N, cdat)
    return cmap

def start_subplot_figure(n_subplots, n_columns=5, fig_width=14, row_height=3):
    """
    Generate a figure object with given subplot specification.
    """

    n_rows = int(np.ceil(n_subplots / float(n_columns)))
    fig = plt.figure(figsize = (fig_width, n_rows * row_height))
    return fig, n_rows, n_columns

def plot_one_gene(E, gene_list, gene_to_plot, x, y, normalize=False, 
    ax=None, order_points=True, col_range=(0,100), buffer_pct=0.03, 
    point_size=1, color_map=None, smooth_operator=None):
    """
    Plot the expression of a list of genes on an embedding.

    Parameters
    ----------
    E: `sp.sparse`
        Cell-by-gene expression matrix
    gene_list: `list`
        Full list of gene names corresponding to E
    gene_to_plot, `list`
        List of genes to be plotted. 
    x: `np.array`
        x coordinate of the embedding
    y: `np.array`
        y coordinate of the embedding
    color_map: {plt.cm.Reds,plt.cm.Blues,...}, (default: None)
    ax: `axis`, optional (default: None)
        An external ax object can be passed here.
    order_points: `bool`, optional (default: True)
        Order points to plot by the gene expression 
    col_range: `tuple`, optional (default: (0,100))
        The color range to plot. The range should be within [0,100]
    buffer_pct: `float`, optional (default: 0.03)
        Extra space for the plot box frame
    point_size: `int`, optional (default: 1)
        Size of the data point
    smooth_operator: `np.array`, optional (default: None)
        A smooth matrix to be applied to the subsect of gene expression matrix. 

    Returns
    -------
    pp: the figure object
    """

    if color_map is None:
        color_map = darken_cmap(plt.cm.Reds,.9)
    if ax is None:
        fig,ax=plt.subplots()
        
    if normalize:
        E = tot_counts_norm(E, target_mean=1e6)[0]
    
    k = list(gene_list).index(gene_to_plot)
    coldat = E[:,k].A
    
    if smooth_operator is None:
        coldat = coldat.squeeze()
    else:
        coldat = np.dot(smooth_operator, coldat).squeeze()
    
    if order_points:
        o = np.argsort(coldat)
    else:
        o = np.arange(len(coldat))
        
    vmin = np.percentile(coldat, col_range[0])
    vmax = np.percentile(coldat, col_range[1])
    if vmax==vmin:
        vmax = coldat.max()
        
    pp = ax.scatter(x[o], y[o], c=coldat[o], s=point_size, cmap=color_map,
               vmin=vmin, vmax=vmax)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(x.min()-x.ptp()*buffer_pct, x.max()+x.ptp()*buffer_pct)
    ax.set_ylim(y.min()-y.ptp()*buffer_pct, y.max()+y.ptp()*buffer_pct)
    
    return pp


def embedding(adata,basis='X_emb',color=None):
    """
    Scatter plot for user-specified embedding basis.

    We imported :func:`~scanpy.pl.embedding` for this purpose.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    basis: `str`, optional (default: 'X_emb')
        The embedding to use for the plot.
    color: `str, list of str, or None` (default: None)
        Keys for annotations of observations/cells or variables/genes, 
        e.g., 'state_info', 'time_info',['Gata1','Gata2']
    """

    sc.pl.embedding(adata,basis=basis,color=color)


def customized_embedding(x, y, vector, normalize=False, title=None, ax=None, 
    order_points=True, set_ticks=False, col_range=(0, 100), buffer_pct=0.03, point_size=1, 
    color_map=None, smooth_operator=None,set_lim=True,
    vmax=np.nan,vmin=np.nan,color_bar=False):
    """
    Plot a vector on an embedding.

    Parameters
    ----------
    x: `np.array`
        x coordinate of the embedding
    y: `np.array`
        y coordinate of the embedding
    vector: `np.array`
        A vector to be plotted.
    color_map: {plt.cm.Reds,plt.cm.Blues,...}, (default: None)
    ax: `axis`, optional (default: None)
        An external ax object can be passed here.
    order_points: `bool`, optional (default: True)
        Order points to plot by the gene expression 
    col_range: `tuple`, optional (default: (0,100))
        The color range to plot. The range should be within [0,100]
    buffer_pct: `float`, optional (default: 0.03)
        Extra space for the plot box frame
    point_size: `int`, optional (default: 1)
        Size of the data point
    smooth_operator: `np.array`, optional (default: None)
        A smooth matrix to be applied to the subsect of gene expression matrix. 
    set_lim: `bool`, optional (default: True)
        Set the plot range (x_limit, and y_limit) automatically.
    vmax: `float`, optional (default: np.nan)
        Maximum color range (saturation). 
        All values above this will be set as vmax.
    vmin: `float`, optional (default: np.nan)
        The minimum color range, all values below this will be set to be vmin.
    color_bar: `bool`, optional (default, False)
        If True, plot the color bar. 
    set_ticks: `bool`, optional (default, False)
        If False, remove figure ticks.   
    """

    if color_map is None:
        color_map = darken_cmap(plt.cm.Reds, .9)
    if ax is None:
        fig, ax = plt.subplots()

    coldat = vector.astype(float)

    if smooth_operator is None:
        coldat = coldat.squeeze()
    else:
        coldat = np.dot(smooth_operator, coldat).squeeze()

    if order_points:
        o = np.argsort(coldat)
    else:
        o = np.arange(len(coldat))
    if np.isnan(vmin):
        vmin = np.percentile(coldat, col_range[0])
    if np.isnan(vmax):
        vmax = np.percentile(coldat, col_range[1])
    if vmax == vmin:
        vmax = coldat.max()

    pp = ax.scatter(x[o], y[o], c=coldat[o], s=point_size, cmap=color_map,
                    vmin=vmin, vmax=vmax)

    if not set_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    if set_lim==True:
        ax.set_xlim(x.min() - x.ptp() * buffer_pct, x.max() + x.ptp() * buffer_pct)
        ax.set_ylim(y.min() - y.ptp() * buffer_pct, y.max() + y.ptp() * buffer_pct)

    if title is not None:
        ax.set_title(title)

    if color_bar:
        #plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Reds), ax=ax)
        plt.colorbar(plt.cm.ScalarMappable(cmap=color_map), ax=ax)

    # if savefig:
    #     fig.savefig(f'figure/customized_embedding_fig_{int(np.round(np.random.rand()*100))}.{settings.file_format_figs}')


def gene_expression_on_manifold(adata,selected_genes,savefig=False,selected_time_points=[],color_bar=False):
    """
    Plot gene expression on the state manifold.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_genes: `list` or 'str'
        List of genes to plot.
    savefig: `bool`, optional (default: False)
        Save the figure.
    selected_time_points: `list`, optional (default: all)
        A list of time points to further restrict the cell states to plot. 
        If map_backwards=True, plot initial states that are among these time points;
        otherwise, show later states that are among these time points.
    color_bar: `bool`, (default: False)
        If True, plot the color bar. 
    """


    fig_width=settings.fig_width; fig_height=settings.fig_height; point_size=settings.fig_point_size
        
    if type(selected_genes)==str:
        selected_genes=[selected_genes]

    x_emb=adata.obsm['X_emb'][:,0]
    y_emb=adata.obsm['X_emb'][:,1]
    figure_path=settings.figure_path

    time_info=np.array(adata.obs['time_info'])
    if (len(selected_time_points)>0):
        sp_idx=np.zeros(adata.shape[0],dtype=bool)
        for xx in selected_time_points:
            sp_id_temp=np.nonzero(time_info==xx)[0]
            sp_idx[sp_id_temp]=True
    else:
        sp_idx=np.ones(adata.shape[0],dtype=bool)
            
    for j in range(len(selected_genes)):
        genes_plot=[selected_genes[j]]

        gene_list=adata.var_names

        col=1
        row=1
        fig = plt.figure(figsize=(fig_width * col, fig_height * row))

        # Plot each gene's expression in a different subplot
        for iG,g in enumerate(genes_plot):
            ax = plt.subplot(row, col, iG+1)
            plot_one_gene(adata.X[sp_idx], gene_list, g, x_emb[sp_idx]-200, y_emb[sp_idx], ax=ax, col_range=(0, 99.8), point_size=point_size)
            ax.set_title(f'{g}')
            ax.axis('off')
    
        if color_bar:
            plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Reds), ax=ax)

        fig.tight_layout()

        if savefig:
            fig.savefig(f'{figure_path}/lung_marker_genes_{selected_genes[j]}.{settings.file_format_figs}')


####################

## Fate bias analysis

####################

def single_cell_transition(adata,selected_state_id_list,used_map_name='transition_map',map_backwards=True,savefig=False,initial_point_size=3,color_bar=True):
    """
    Plot transition probability from given initial cell states.

    If `map_backwards=True`, plot future state probability starting from given initial state;
    otherwise, plot the probability of source states where the current cell state comes from.  

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_state_id_list: `list`
        List of cell id's. Like [0,1,2].
    used_map_name: `str`
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map',...}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    map_backwards: `bool`, optional (default: True)
        If `map_backwards=True`, analyze transitions backward in time, 
        which plots initial cell states (rows of Tmap, at t1);
        otherwise, analyze forward transitions and show later cell 
        states (columns of Tmap, at t2).
    initial_point_size: `int`, optional (default: 3)
        Relative size of the data point for the selected cells.
    save_fig: `bool`, optional (default: False)
        If true, save figure to defined directory at settings.figure_path
    color_bar: `bool`, optional (default: True)
        Plot the color bar. 
    """

    hf.check_available_map(adata)
    fig_width=settings.fig_width; fig_height=settings.fig_height; point_size=settings.fig_point_size
    #set_up_plotting()


    if used_map_name not in adata.uns['available_map']:
        logg.error(f"used_map_name should be among {adata.uns['available_map']}")

    else:
        state_annote=adata.obs['state_info']
        x_emb=adata.obsm['X_emb'][:,0]
        y_emb=adata.obsm['X_emb'][:,1]
        data_des=adata.uns['data_des'][-1]
        figure_path=settings.figure_path

        if not map_backwards:
            cell_id_t1=adata.uns['Tmap_cell_id_t1']
            cell_id_t2=adata.uns['Tmap_cell_id_t2']
            Tmap=adata.uns[used_map_name]

        else:
            cell_id_t2=adata.uns['Tmap_cell_id_t1']
            cell_id_t1=adata.uns['Tmap_cell_id_t2']
            Tmap=adata.uns[used_map_name].T

        selected_state_id_list=np.array(selected_state_id_list)
        full_id_list=np.arange(len(cell_id_t1))
        valid_idx=np.in1d(full_id_list,selected_state_id_list)
        if np.sum(valid_idx)<len(selected_state_id_list):
            logg.error(f"Valid id range is (0,{len(cell_id_t1)-1}). Please use a smaller ID!")
            selected_state_id_list=full_id_list[valid_idx]

        if len(selected_state_id_list)==0:
            logg.error("No valid states selected.")
        else:
            if ssp.issparse(Tmap): Tmap=Tmap.A
                    

            row = len(selected_state_id_list)
            col = 1
            fig = plt.figure(figsize=(fig_width * col, fig_height * row))

            for j, target_cell_ID in enumerate(selected_state_id_list):
                if j > 0:
                    disp_name = 0
                ax0 = plt.subplot(row, col, col * j + 1)

                if target_cell_ID<Tmap.shape[0]:
                    prob_vec=np.zeros(len(x_emb))
                    prob_vec[cell_id_t2]=Tmap[target_cell_ID, :]
                    customized_embedding(x_emb, y_emb, prob_vec, point_size=point_size, ax=ax0)
                    
                    ax0.plot(x_emb[cell_id_t1][target_cell_ID],y_emb[cell_id_t1][target_cell_ID],'*b',markersize=initial_point_size*point_size)

                    #ax0.set_title(f"t1 state (blue star) ({cell_id_t1[target_cell_ID]})")
                    if map_backwards:
                        ax0.set_title(f"ID ($t_2$): {target_cell_ID}")
                    else:
                        ax0.set_title(f"ID ($t_1$): {target_cell_ID}")
                    plt.rc('text', usetex=True)


            if color_bar:
                Clb=fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Reds), ax=ax0,label='Probability')


            plt.tight_layout()
            if savefig:
                fig.savefig(f"{figure_path}/plotting_transition_map_probability_{map_backwards}.{settings.file_format_figs}")
            plt.rc('text', usetex=False)


def fate_map(adata,selected_fates=[],used_map_name='transition_map',
    map_backwards=True,normalize_by_fate_size=False,selected_time_points=[],background=True, show_histogram=False,
    plot_target_state=True,auto_color_scale=True,color_bar=True,
    target_transparency=0.2,horizontal=False,figure_index=''):
    """
    Plot transition probability to given fate/ancestor clusters.

    If `map_backwards=True`, plot transition probability of early 
    states to given fate clusters (fate map); else, plot transition 
    probability of later states from given ancestor clusters (ancestor map).
    Figures are saved at the defined directory at settings.figure_path.

    If `normalize_by_fate_size=True`, we normalize the predicted 
    fate probability by the expected probability, i.e., the fraction of cells 
    within the targeted cluster at the corresponding time point. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_fates: `list`
        List of cluster ids consistent with adata.obs['state_info']. 
        It allows a nested structure. If so, we merge clusters within 
        each sub-list into a mega-fate cluster.
    used_map_name: `str`
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map',...}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    map_backwards: `bool`, optional (default: True)
        If `map_backwards=True`, analyze transitions backward in time, 
        which plots initial cell state (rows of Tmap, at t1) profiles towards 
        given fate clusters; otherwise, analyze forward transitions and show later cell 
        states (columns of Tmap, at t2) profiles starting from given initial clusters. 
    normalize_by_fate_size: `bool`, optional (default: False)
        If true, normalize the predicted fate probability by expected probability,
        defined as the fraction of cells inside the targeted cluster at 
        the corresponding time point.  
    selected_time_points: `list`, optional (default: all)
        A list of time points to further restrict the cell states to plot. 
        The default choice is not to constrain the cell states to show. 
    background: `bool`, optional (default: True)
        If true, plot all cell states (t1+t2) in grey as the background. 
    show_histogram: `bool`, optional (default: False)
        If true, show the distribution of inferred fate probability.
    plot_target_state: `bool`, optional (default: True)
        If true, highlight the target clusters as defined in selected_fates.
    color_bar: `bool`, optional (default: True)
        plot the color bar if True.
    target_transparency: `float`, optional (default: 0.2)
        It controls the transparency of the plotted target cell states, 
        for visual effect. Range: [0,1].
    horizontal: `bool`, optional (default: False)
        If true, plot the figure panels horizontally; else, vertically.
    figure_index: `str`, optional (default: '')
        String index for annotate filename for saved figures. Used to distinuigh plots from different conditions. 


    Returns
    -------
    Store a dictionary of results {"raw_fate_map","normalized_fate_map","expected_prob"} at adata.uns['fate_map_output']. 
    """

    hf.check_available_map(adata)
    fig_width=settings.fig_width; fig_height=settings.fig_height; point_size=settings.fig_point_size
    #set_up_plotting()

    if used_map_name not in adata.uns['available_map']:
        logg.error(f"used_map_name should be among {adata.uns['available_map']}")

    else:        
        state_annote=adata.obs['state_info']
        if map_backwards:
            cell_id_t1=adata.uns['Tmap_cell_id_t1']
            cell_id_t2=adata.uns['Tmap_cell_id_t2']

        else:
            cell_id_t2=adata.uns['Tmap_cell_id_t1']
            cell_id_t1=adata.uns['Tmap_cell_id_t2']


        time_info=np.array(adata.obs['time_info'])
        sp_idx=hf.selecting_cells_by_time_points(time_info[cell_id_t1],selected_time_points)


        x_emb=adata.obsm['X_emb'][:,0]
        y_emb=adata.obsm['X_emb'][:,1]
        data_des=adata.uns['data_des'][-1]
        figure_path=settings.figure_path



        fate_map_0,mega_cluster_list,relative_bias,expected_prob,valid_fate_list,sel_index_list=hf.compute_fate_map_and_intrinsic_bias(adata,selected_fates=selected_fates,used_map_name=used_map_name,map_backwards=map_backwards)

        if (len(mega_cluster_list)==0) or (np.sum(sp_idx)==0):
            logg.error("No cells selected. Computation aborted!")
        else:

            if normalize_by_fate_size:
                fate_map=relative_bias
            else:
                fate_map=fate_map_0

            ################### plot fate probability
            vector_array=[vector for vector in list(fate_map.T)]
            description=[fate for fate in mega_cluster_list]
            if horizontal:
                row =1; col =len(vector_array)
            else:
                row =len(vector_array); col =1

            fig = plt.figure(figsize=(fig_width * col, fig_height * row))
            for j in range(len(vector_array)):
                ax0 = plt.subplot(row, col, j + 1)
                
                if background:
                    customized_embedding(x_emb,y_emb,np.zeros(len(y_emb)),point_size=point_size,ax=ax0,title=description[j])            
                    if plot_target_state:
                        for zz in valid_fate_list[j]:
                            idx_2=state_annote==zz
                            ax0.plot(x_emb[idx_2],y_emb[idx_2],'.',color='cyan',markersize=point_size*1,alpha=target_transparency)
                else:
                    customized_embedding(x_emb[cell_id_t1],y_emb[cell_id_t1],np.zeros(len(y_emb[cell_id_t1])),point_size=point_size,ax=ax0,title=description[j])

                if auto_color_scale:
                    customized_embedding(x_emb[cell_id_t1][sp_idx],y_emb[cell_id_t1][sp_idx],vector_array[j][sp_idx],point_size=point_size,ax=ax0,title=description[j],set_lim=False)
                else:
                    customized_embedding(x_emb[cell_id_t1][sp_idx],y_emb[cell_id_t1][sp_idx],vector_array[j][sp_idx],point_size=point_size,ax=ax0,title=description[j],set_lim=False,vmax=1,vmin=0)
            
            if color_bar:
                fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Reds), ax=ax0,label='Fate probability')
          
            #yy=int(np.random.rand()*100)
            plt.tight_layout()
            fig.savefig(f'{figure_path}/{data_des}_fate_map_overview_{description[j]}.{settings.file_format_figs}')

            if show_histogram:
                fig = plt.figure(figsize=(fig_width * col, fig_height * row))
                for j in range(len(vector_array)):
                    ax = plt.subplot(row, col, j + 1)

                    temp_array=vector_array[j][sp_idx]
                    new_idx=np.argsort(abs(temp_array-0.5))
                    xxx=temp_array[new_idx]
                    ax = plt.subplot(row, col, j + 1)
                    ax.hist(xxx,50,color='#2ca02c',density=True)
                    ax.set_xlim([0,1])
                    ax.set_xlabel('Relative fate bias')
                    ax.set_ylabel('Density')
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.set_title(f'{description[j]}, Ave.: {int(np.mean(xxx)*100)/100}')
                    plt.tight_layout()
                    fig.savefig(f'{figure_path}/{data_des}_intrinsic_fate_bias_BW{map_backwards}_histogram.{settings.file_format_figs}')

            ## save data to adata
            adata.uns['fate_map_output']={"raw_fate_map":fate_map[sp_idx,:],"normalized_fate_map":relative_bias[sp_idx,:],"expected_prob":expected_prob}



def binary_fate_bias(adata,selected_fates=[],used_map_name='transition_map',
    map_backwards=True,normalize_by_fate_size=False,selected_time_points=[],sum_fate_prob_thresh=0,
    plot_target_state=False,color_bar=True,show_histogram=True,
    target_transparency=0.2,figure_index=''):
    """
    Plot fate bias to given two fate/ancestor clusters (A, B).

    Fate bias is a `scalar` between (0,1) at each state, defined as 
    competition between two fate clusters. It is computed from the 
    fate probability Prob(X) towards cluster X.
    Specifically, the bias of a state between fate A and B is

    * Prob(A)/[Prob(A)+Prob(B)]

    If `normalize_by_fate_size=True`, before computing the bias, we first 
    normalize the predicted fate probability Prob(X) by the 
    expected probability, i.e., the fraction of cells within the targeted cluster X at
    the corresponding time point. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_fates: `list`
        List of cluster ids consistent with adata.obs['state_info']. 
        It allows a nested structure. If so, we merge clusters within 
        each sub-list into a mega-fate cluster.
    used_map_name: `str`
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map',...}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    map_backwards: `bool`, optional (default: True)
        If `map_backwards=True`, analyze transitions backward in time, 
        which plots initial cell state (rows of Tmap, at t1) profiles towards 
        given fate clusters; otherwise, analyze forward transitions and show later cell 
        states (columns of Tmap, at t2) profiles starting from given initial clusters. 
    normalize_by_fate_size: `bool`, optional (default: False)
        If true, normalize the predicted fate probability by expected probability,
        defined as the fraction of cells inside the targeted cluster at 
        the corresponding time point. 
    selected_time_points: `list`, optional (default: all)
        A list of time points to further restrict the cell states to plot. 
        The default choice is not to constrain the cell states to show. 
    sum_fate_prob_thresh: `float`, optional (default: 0)
        The fate bias of a state is plotted only when it has a cumulative fate 
        probability to the combined cluster (A+B) larger than this threshold,
        i.e., P(i->A)+P(i+>B) >  sum_fate_prob_thresh. 
    plot_target_state: `bool`, optional (default: True)
        If true, highlight the target clusters as defined in selected_fates.
    color_bar: `bool`, optional (default: True)
        plot the color bar if True.
    show_histogram: `bool`, optional (default: True)
        If true, show the distribution of inferred fate probability.
    target_transparency: `float`, optional (default: 0.2)
        It controls the transparency of the plotted target cell states, 
        for visual effect. Range: [0,1].
    figure_index: `str`, optional (default: '')
        String index for annotate filename for saved figures. Used to distinuigh plots from different conditions. 

    Returns
    -------
    The results are stored at adata.uns['binary_fate_bias']
    """

    hf.check_available_map(adata)
    fig_width=settings.fig_width; fig_height=settings.fig_height; point_size=settings.fig_point_size
    #set_up_plotting()
    if used_map_name not in adata.uns['available_map']:
        logg.error(f"used_map_name should be among {adata.uns['available_map']}")

    else:
        state_annote=adata.obs['state_info']
        if map_backwards:
            cell_id_t1=adata.uns['Tmap_cell_id_t1']
            cell_id_t2=adata.uns['Tmap_cell_id_t2']

        else:
            cell_id_t2=adata.uns['Tmap_cell_id_t1']
            cell_id_t1=adata.uns['Tmap_cell_id_t2']

        x_emb=adata.obsm['X_emb'][:,0]
        y_emb=adata.obsm['X_emb'][:,1]
        data_des=adata.uns['data_des'][-1]
        figure_path=settings.figure_path


        ## select time points
        time_info=np.array(adata.obs['time_info'])
        sp_idx=hf.selecting_cells_by_time_points(time_info[cell_id_t1],selected_time_points)

        cell_id_t1_sp=cell_id_t1[sp_idx]
            

        if len(selected_fates)!=2: 
            logg.error(f"Must have only two fates")
        else:
            fate_map,mega_cluster_list,relative_bias,expected_prob,valid_fate_list,sel_index_list=hf.compute_fate_map_and_intrinsic_bias(adata,selected_fates=selected_fates,used_map_name=used_map_name,map_backwards=map_backwards)

            if (len(mega_cluster_list)!=2) or (np.sum(sp_idx)==0):
                logg.error(f"Do not have valid fates or time points. Computation aborted!")
            else:
                resol=10**(-10)

                fig=plt.figure(figsize=(fig_width,fig_height))
                ax=plt.subplot(1,1,1)

                if normalize_by_fate_size:
                    potential_vector_temp=relative_bias[sp_idx,:]+resol
                else:
                    potential_vector_temp=fate_map[sp_idx,:]+resol

                diff=potential_vector_temp[:,0]#-potential_vector_temp[:,1]
                tot=potential_vector_temp.sum(1)

                valid_idx=tot>sum_fate_prob_thresh # default 0.5
                vector_array=np.zeros(np.sum(valid_idx))
                vector_array=diff[valid_idx]/(tot[valid_idx])
                #vector_array=2*potential_vector_temp[valid_idx,8]/tot[valid_idx]-1
                #vector_array=potential_vector_temp[:,8]/potential_vector_temp[:,9]

                #customized_embedding(x_emb[cell_id_t1][sp_idx],y_emb[cell_id_t1][sp_idx],np.zeros(len(y_emb[cell_id_t1][sp_idx])),point_size=point_size,ax=ax)
                if plot_target_state:
                    customized_embedding(x_emb,y_emb,np.zeros(len(y_emb)),point_size=point_size,ax=ax)
         
                    for zz in valid_fate_list[0]:
                        idx_2=state_annote[cell_id_t2]==zz
                        ax.plot(x_emb[cell_id_t2[idx_2]],y_emb[cell_id_t2[idx_2]],'.',color='red',markersize=point_size*2,alpha=target_transparency)
                    for zz in valid_fate_list[1]:
                        idx_2=state_annote[cell_id_t2]==zz
                        ax.plot(x_emb[cell_id_t2[idx_2]],y_emb[cell_id_t2[idx_2]],'.',color='blue',markersize=point_size*2,alpha=target_transparency)

                        
                else:
                    customized_embedding(x_emb[cell_id_t1_sp],y_emb[cell_id_t1_sp],np.zeros(len(y_emb[cell_id_t1_sp])),point_size=point_size,ax=ax)
                #customized_embedding(x_emb[cell_id_t2],y_emb[cell_id_t2],np.zeros(len(y_emb[cell_id_t2])),point_size=point_size,ax=ax)

                new_idx=np.argsort(abs(vector_array-0.5))
                customized_embedding(x_emb[cell_id_t1_sp][valid_idx][new_idx],y_emb[cell_id_t1_sp][valid_idx][new_idx],
                                    vector_array[new_idx],vmax=1,vmin=0,
                                    point_size=point_size,set_lim=False,ax=ax,color_map=plt.cm.bwr,order_points=False)

        #         # remove un-wanted time points
        #         if len(cell_id_t1[~sp_idx])>0:
        #             customized_embedding(x_emb[cell_id_t1[~sp_idx]],y_emb[cell_id_t1[~sp_idx]],np.zeros(len(y_emb[cell_id_t1[~sp_idx]])),
        #                         point_size=point_size,set_lim=False,ax=ax,color_map=plt.cm.bwr,order_points=False)

                if color_bar:
                    Clb=fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.bwr), ax=ax,label='Fate bias')
                    Clb.ax.set_title(f'{mega_cluster_list[0]}')

                plt.tight_layout()
                fig.savefig(f'{figure_path}/{data_des}_binary_fate_bias_BW{map_backwards}{figure_index}.{settings.file_format_figs}')



                #adata.uns['binary_fate_bias']=vector_array
                vector_array_fullSpace=np.zeros(len(x_emb))+0.5
                vector_array_fullSpace[cell_id_t1_sp[valid_idx]]=vector_array
                adata.uns['binary_fate_bias']=[vector_array,vector_array_fullSpace]

                

                if show_histogram:
                    xxx=vector_array
                    fig=plt.figure(figsize=(fig_width,fig_height));ax=plt.subplot(1,1,1)
                    ax.hist(xxx,50,color='#2ca02c',density=True)
                    ax.set_xlim([0,1])
                    ax.set_xlabel('Fate bias')
                    ax.set_ylabel('Density')
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.set_title(f'Average: {int(np.mean(xxx)*100)/100}')
                    plt.tight_layout()
                    fig.savefig(f'{figure_path}/{data_des}_binary_fate_bias_BW{map_backwards}_histogram{figure_index}.{settings.file_format_figs}')



def fate_coupling_from_Tmap(adata,selected_fates=[],used_map_name='transition_map',selected_time_points=[],normalize_fate_map=False,color_bar=True,coupling_normalization='SW',rename_selected_fates=[]):
    """
    Plot fate coupling determined by the transition map.

    We use the fate map of cell states at t1 to compute the fate coupling.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_fates: `list`
        List of cluster ids consistent with adata.obs['state_info']. 
        It allows a nested structure. If so, we merge clusters within 
        each sub-list into a mega-fate cluster.
    used_map_name: `str`
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map',...}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    selected_time_points: `list`, optional (default: all)
        A list of time points to further restrict the cell states to plot. 
        The default choice is not to constrain the cell states to show. 
    normalize_fate_map: `bool`, optional (default: False)
        If true, normalize fate map before computing the fate coupling. 
    color_bar: `bool`, optional (default: True)
        Plot the color bar.
    coupling_normalization: `str`, optional (default: 'SW')
        Method to normalize the coupling matrix: {'SW','Weinreb'}.
    rename_selected_fates: `list`, optional (default: [])
        Provide new names in substitution of names in selected_fates.
        For this to be effective, the new name list needs to have names 
        in exact correspondence to those in the old list. 
    """

    hf.check_available_map(adata)
    fig_width=settings.fig_width; fig_height=settings.fig_height; point_size=settings.fig_point_size
    #set_up_plotting()
    
    map_backwards=True
    
    if used_map_name not in adata.uns['available_map']:
        logg.error(f"used_map_name should be among {adata.uns['available_map']}")

    else:        
        state_annote=adata.obs['state_info']
        if map_backwards:
            cell_id_t1=adata.uns['Tmap_cell_id_t1']
            cell_id_t2=adata.uns['Tmap_cell_id_t2']

        else:
            cell_id_t2=adata.uns['Tmap_cell_id_t1']
            cell_id_t1=adata.uns['Tmap_cell_id_t2']


        time_info=np.array(adata.obs['time_info'])
        sp_idx=hf.selecting_cells_by_time_points(time_info[cell_id_t1],selected_time_points)


        x_emb=adata.obsm['X_emb'][:,0]
        y_emb=adata.obsm['X_emb'][:,1]
        data_des=adata.uns['data_des'][-1]
        data_des=f'{data_des}_Tmap_fate_coupling'
        figure_path=settings.figure_path



        fate_map,mega_cluster_list,relative_bias,expected_prob,valid_fate_list,sel_index_list=hf.compute_fate_map_and_intrinsic_bias(adata,selected_fates=selected_fates,used_map_name=used_map_name,map_backwards=map_backwards)

        if (len(mega_cluster_list)==0) or (np.sum(sp_idx)==0):
            logg.error("No cells selected. Computation aborted!")
        else:
            # normalize the map to enhance the fate choice difference among selected clusters
            if normalize_fate_map and (fate_map.shape[1]>1):
                resol=10**-10 
                fate_map=hf.sparse_rowwise_multiply(fate_map,1/(resol+np.sum(fate_map,1)))
                #fate_entropy_temp=fate_entropy_array[x0]
            
    
            if len(rename_selected_fates)!=len(mega_cluster_list):
                rename_selected_fates=mega_cluster_list

            X_ICSLAM = hf.get_normalized_covariance(fate_map[sp_idx],method=coupling_normalization)
            heatmap(figure_path, X_ICSLAM, rename_selected_fates,color_bar_label='Coupling',color_bar=color_bar,data_des=data_des)


####################

## DDE analysis

####################


def differential_genes(adata,plot_groups=True,gene_N=100,plot_gene_N=3,
    savefig=False):
    """
    Perform differential gene expression analysis and plot top DGE genes.

    It is assumed that the selected cell states are stored at 
    adata.obs['cell_group_A'] and adata.obs['cell_group_B'].
    You can run this function after 
    :func:`.dynamic_trajectory_from_binary_fate_bias`, which performs
    cell state selection.

    We use Wilcoxon rank-sum test to calculate P values, followed by
    Benjamini-Hochberg correction. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Need to contain gene expression matrix, and DGE cell groups A, B. 
    plot_groups: `bool`, optional (default: True)
        If true, plot the selected ancestor states for A, B
    gene_N: `int`, optional (default: 100)
        Number of top differentially expressed genes to selected.
    plot_gene_N: `int`, optional (default: 5)
        Number of top DGE genes to plot
    savefig: `bool`, optional (default: False)
        Save all plots.

    Returns
    -------
    diff_gene_A: `pd.DataFrame`
        Genes differentially expressed in cell state group A, ranked
        by the ratio of mean expressions between 
        the two groups, with the top being more differentially expressed.
    diff_gene_B: `pd.DataFrame`
        Genes differentially expressed in cell state group B, ranked
        by the ratio of mean expressions between 
        the two groups, with the top being more differentially expressed.
    """

    fig_width=settings.fig_width; fig_height=settings.fig_height; point_size=settings.fig_point_size
    diff_gene_A=[]
    diff_gene_B=[]
    if ('cell_group_A' not in adata.obs.keys()) or ('cell_group_B' not in adata.obs.keys()): 
        logg.error("Cell population A or B not defined yet. Please run upstream methods to define the population\n"
            "like: cs.pl.dynamic_trajectory_from_binary_fate_bias.")
    else:
        idx_for_group_A=adata.obs['cell_group_A']
        idx_for_group_B=adata.obs['cell_group_B']
        #hf.check_available_map(adata)
        #set_up_plotting()
        if (np.sum(idx_for_group_A)==0) or (np.sum(idx_for_group_B)==0):
            logg.error("Group A or B has zero selected cell states. Could be that the cluser name is wrong; Or, the selection is too stringent. Consider use a smaller 'bias_threshold'")

        else:

            dge=hf.get_dge_SW(adata,idx_for_group_B,idx_for_group_A)

            dge=dge.sort_values(by='ratio',ascending=True)
            diff_gene_A=dge[:gene_N]
            #diff_gene_A=diff_gene_A_0[dge[:gene_N]['pv']<0.05]

            dge=dge.sort_values(by='ratio',ascending=False)
            diff_gene_B=dge[:gene_N]
            #diff_gene_B=diff_gene_B_0[dge[:gene_N]['pv']<0.05]

            x_emb=adata.obsm['X_emb'][:,0]
            y_emb=adata.obsm['X_emb'][:,1]
            figure_path=settings.figure_path
            
            if plot_groups:

                fig,nrow,ncol = start_subplot_figure(2, row_height=4, n_columns=2, fig_width=8)
                ax = plt.subplot(nrow, ncol, 1)
                customized_embedding(x_emb,y_emb,idx_for_group_A,ax=ax,point_size=point_size)
                ax.set_title(f'Group A')
                ax.axis('off')
                ax = plt.subplot(nrow, ncol, 2)
                customized_embedding(x_emb,y_emb,idx_for_group_B,ax=ax,point_size=point_size)
                ax.set_title(f'Group B')
                ax.axis('off')
                
                plt.tight_layout()
                if savefig:
                    fig.savefig(f'{figure_path}/dge_analysis_groups.{settings.file_format_figs}')
                
            #logg.error("Plot differentially-expressed genes for group A")
            if plot_gene_N>0:
                #logg.error(f"Plot the top {plot_gene_N} genes that are differentially expressed on group A")
                fig,nrow,ncol = start_subplot_figure(plot_gene_N, row_height=2.5, n_columns=5, fig_width=16)
                for j in range(plot_gene_N):
                    ax = plt.subplot(nrow, ncol, j+1)

                    #pdb.set_trace()
                    gene_name=np.array(diff_gene_A['gene'])[j]
                    customized_embedding(x_emb,y_emb,adata.obs_vector(gene_name),ax=ax,point_size=point_size)
                    ax.set_title(f'{gene_name}')
                    ax.axis('off')
                plt.tight_layout()
                if savefig:
                    fig.savefig(f'{figure_path}/dge_analysis_groups_A_genes.{settings.file_format_figs}')
                
                #logg.error("Plot differentially-expressed genes for group B")
                #logg.error(f"Plot the top {plot_gene_N} genes that are differentially expressed on group B")
                fig,nrow,ncol = start_subplot_figure(plot_gene_N, row_height=2.5, n_columns=5, fig_width=16)
                for j in range(plot_gene_N):
                    ax = plt.subplot(nrow, ncol, j+1)
                    gene_name=np.array(diff_gene_B['gene'])[j]
                    customized_embedding(x_emb,y_emb,adata.obs_vector(gene_name),ax=ax,point_size=point_size)
                    ax.set_title(f'{gene_name}')
                    ax.axis('off')
                plt.tight_layout()
                if savefig:
                    fig.savefig(f'{figure_path}/dge_analysis_groups_B_genes.{settings.file_format_figs}')
            
                # logg.error('--------------Differentially expressed genes for group A --------------')
                # logg.error(diff_gene_A)
                
                # logg.error('--------------Differentially expressed genes for group B --------------')
                # logg.error(diff_gene_B)
        
    return diff_gene_A,diff_gene_B



def differential_genes_for_given_fates(adata,selected_fates=[],selected_time_points=[],
    plot_groups=True,gene_N=100,plot_gene_N=3,savefig=False):
    """
    Find and plot DGE genes between different clusters.

    We assume that one or two fate clusters are selected. When only
    one fate is provided, we perform DGE analysis for states in this cluster 
    and the remaining states. We use Wilcoxon rank-sum test to calculate P values, 
    followed by Benjamini-Hochberg correction. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    selected_fates: `list`
        List of cluster_ids consistent with adata.obs['state_info']
    selected_time_points: `list`, optional (default: all)
        A list of time points to further restrict the cell states to plot. 
        If map_backwards=True, plot initial states that are among these time points;
        else, plot later states that are among these time points.
    plot_groups: `bool`, optional (default: True)
        If true, plot the selected ancestor states for A, B
    gene_N: `int`, optional (default: 100)
        Number of top differentially expressed genes to selected.
    plot_gene_N: `int`, optional (default: 5)
        Number of top DGE genes to plot
    savefig: `bool`, optional (default: False)
        Save all plots.

    Returns
    -------
    diff_gene_A: `pd.DataFrame`
        Genes differentially expressed in cell state group A, ranked
        by the ratio of mean expressions between 
        the two groups, with the top being more differentially expressed.
    diff_gene_B: `pd.DataFrame`
        Genes differentially expressed in cell state group B, ranked
        by the ratio of mean expressions between 
        the two groups, with the top being more differentially expressed.
    """

    diff_gene_A=[]
    diff_gene_B=[]
    fig_width=settings.fig_width; fig_height=settings.fig_height; point_size=settings.fig_point_size


    time_info=np.array(adata.obs['time_info'])
    sp_idx=hf.selecting_cells_by_time_points(time_info,selected_time_points)
    if np.sum(sp_idx)==0:
        logg.error("No states selected. Abort computation.")
        return diff_gene_A,diff_gene_B
    else:
        adata_1=adata[np.nonzero(sp_idx)[0]]


    state_annot_0=np.array(adata_1.obs['state_info'])
    if (len(selected_fates)==0) or (len(selected_fates)>2) or (np.sum(sp_idx)==0):
        logg.error(f"There should be only one or two fate clusters; or no cell states selected. Abort computation!")
    else:

        mega_cluster_list,valid_fate_list,fate_array_flat,sel_index_list=hf.analyze_selected_fates(selected_fates,state_annot_0)
        if len(mega_cluster_list)==0:
            logg.error("No cells selected. Computation aborted!")
        else:
            idx_for_group_A=sel_index_list[0]

            if len(mega_cluster_list)==2:
                idx_for_group_B=sel_index_list[1]
            else:
                idx_for_group_B=~idx_for_group_A

            group_A_idx_full=np.zeros(adata.shape[0],dtype=bool)
            group_A_idx_full[np.nonzero(sp_idx)[0]]=idx_for_group_A
            group_B_idx_full=np.zeros(adata.shape[0],dtype=bool)
            group_B_idx_full[np.nonzero(sp_idx)[0]]=idx_for_group_B
            adata.obs['cell_group_A']=group_A_idx_full
            adata.obs['cell_group_B']=group_B_idx_full
            #adata.uns['DGE_analysis']=[adata_1,idx_for_group_A,idx_for_group_B]

            diff_gene_A,diff_gene_B=differential_genes(adata,plot_groups=plot_groups,gene_N=gene_N,plot_gene_N=plot_gene_N,savefig=savefig)
                
    return diff_gene_A,diff_gene_B


######################

## Dynamic trajectory

######################

def dynamic_trajectory_from_binary_fate_bias(adata,selected_fates=[],used_map_name='transition_map',
    map_backwards=True,normalize_by_fate_size=False,selected_time_points=[],
    bias_threshold=0.1,sum_fate_prob_thresh=0,avoid_target_states=False,
    plot_ancestor=True,savefig=False,plot_target_state=True,target_transparency=0.2):
    """
    Identify trajectories towards/from two given clusters.

    Fate bias is a `scalar` between (0,1) at each state, defined as competition between 
    two fate clusters, as in :func:`.binary_fate_bias`. Given the fate probability 
    Prob(X) towards cluster X, the selected ancestor population satisfies:

       * Prob(A)+Prob(B)>sum_fate_prob_thresh; 

       * for A: Bias>0.5+bias_threshold

       * for B: bias<0.5+bias_threshold

    If `normalize_by_fate_size=True`, before population selection, we first 
    normalize the predicted fate probability Prob(X) by the 
    expected probability, i.e., the fraction of cells within the targeted cluster X at
    the corresponding time point. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_fates: `list`
        List of cluster ids consistent with adata.obs['state_info']. 
        It allows a nested structure. 
    used_map_name: `str`
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map',...}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    map_backwards: `bool`, optional (default: True)
        If `map_backwards=True`, analyze transitions backward in time, 
        which plots initial cell state (rows of Tmap, at t1) profiles towards 
        given fate clusters; otherwise, analyze forward transitions and show later cell 
        states (columns of Tmap, at t2) profiles starting from given initial clusters. 
    normalize_by_fate_size: `bool`, optional (default: False)
        If true, normalize the predicted fate probability by expected probability,
        defined as the fraction of cells inside the targeted cluster at 
        the corresponding time point. 
    selected_time_points: `list`, optional (default: all)
        A list of time points to further restrict the cell states to plot. 
        The default choice is not to constrain the cell states to show. 
    bias_threshold: `float`, optional (default: 0), range: (0,0.5)
        The threshold for selecting ancestor population. 
    sum_fate_prob_thresh: `float`, optional (default: 0), range: (0,1)
        Minimum cumulative probability towards joint cluster (A,B) 
        to qualify for ancestor selection.
    savefig: `bool`, optional (default: False)
        Save all plots.
    avoid_target_states: `bool`, optional (default: False)
        If true, avoid selecting cells at the target cluster (A, or B) as 
        ancestor population.
    plot_target_state: `bool`, optional (default: True)
        If true, highlight the target clusters as defined in selected_fates.
    target_transparency: `float`, optional (default: 0.2)
        Transparency parameter for plotting. 

    Returns
    -------
    Store the inferred ancestor states in adata.uns['cell_group_A'] and adata.uns['cell_group_B']

    Combine ancestor states and target states into adata.uns['dynamic_trajectory'] for each fate. 
    """

    diff_gene_A=[]
    diff_gene_B=[]
    hf.check_available_map(adata)
    fig_width=settings.fig_width; fig_height=settings.fig_height; point_size=settings.fig_point_size

    if used_map_name not in adata.uns['available_map']:
        logg.error(f"used_map_name should be among {adata.uns['available_map']}")


    else:
        if map_backwards:
            cell_id_t1=adata.uns['Tmap_cell_id_t1']
            cell_id_t2=adata.uns['Tmap_cell_id_t2']

        else:
            cell_id_t2=adata.uns['Tmap_cell_id_t1']
            cell_id_t1=adata.uns['Tmap_cell_id_t2']

        figure_path=settings.figure_path
        state_annote_t1=np.array(adata.obs['state_info'][cell_id_t1])

        if  (len(selected_fates)!=2):
            logg.error(f" Must provide exactly two fates.")

        else:
            ## select time points
            time_info=np.array(adata.obs['time_info'])
            sp_idx=hf.selecting_cells_by_time_points(time_info[cell_id_t1],selected_time_points)

            #if 'fate_map' not in adata.uns.keys():
            fate_map,mega_cluster_list,relative_bias,expected_prob,valid_fate_list,sel_index_list=hf.compute_fate_map_and_intrinsic_bias(adata,selected_fates=selected_fates,used_map_name=used_map_name,map_backwards=map_backwards)

            if (len(mega_cluster_list)!=2) or (np.sum(sp_idx)==0):
                logg.error(f"Do not have valid fates or time points. Computation aborted!")
            else:

                resol=10**(-10)
                if normalize_by_fate_size:
                    potential_vector_temp=relative_bias+resol
                else:
                    potential_vector_temp=fate_map+resol

                # if len(mega_cluster_list)==1:
                #     idx_for_group_A=potential_vector_temp[:,0]>bias_threshold
                #     idx_for_group_B=~idx_for_group_A
                #     #valid_idx=np.ones(len(cell_id_t1),dtype=bool)

                #     ### remove states already exist in the selected fate cluster 
                #     if avoid_target_states:
                #         for zz in valid_fate_list[0]:
                #             id_A_t1=np.nonzero(state_annote_t1==zz)[0]
                #             idx_for_group_A[id_A_t1]=False

                # else:
                    
                diff=potential_vector_temp[:,0]#-potential_vector_temp[:,1]
                tot=potential_vector_temp.sum(1)

                valid_idx=tot>sum_fate_prob_thresh # default 0
                valid_id=np.nonzero(valid_idx)[0]
                vector_array=np.zeros(np.sum(valid_idx))
                vector_array=diff[valid_idx]/(tot[valid_idx])

                idx_for_group_A=np.zeros(len(tot),dtype=bool)
                idx_for_group_B=np.zeros(len(tot),dtype=bool)
                idx_for_group_A[valid_id]=vector_array>(0.5+bias_threshold)
                idx_for_group_B[valid_id]=vector_array<(0.5-bias_threshold)


                ### remove states already exist in the selected fate cluster 
                if avoid_target_states:
                    for zz in valid_fate_list[0]:
                        id_A_t1=np.nonzero(state_annote_t1==zz)[0]
                        idx_for_group_A[id_A_t1]=False

                    for zz in valid_fate_list[1]:
                        id_B_t1=np.nonzero(state_annote_t1==zz)[0]
                        idx_for_group_B[id_B_t1]=False


                group_A_idx_full=np.zeros(adata.shape[0],dtype=bool)
                group_A_idx_full[cell_id_t1[sp_idx]]=idx_for_group_A[sp_idx]
                group_B_idx_full=np.zeros(adata.shape[0],dtype=bool)
                group_B_idx_full[cell_id_t1[sp_idx]]=idx_for_group_B[sp_idx]
                adata.obs['cell_group_A']=group_A_idx_full
                adata.obs['cell_group_B']=group_B_idx_full                


                if plot_ancestor:
                    x_emb=adata.obsm['X_emb'][:,0]
                    y_emb=adata.obsm['X_emb'][:,1]
                    state_annote=adata.obs['state_info']

                    fig,nrow,ncol = start_subplot_figure(2, row_height=4, n_columns=2, fig_width=8)
                    ax = plt.subplot(nrow, ncol, 1)
                    customized_embedding(x_emb,y_emb,group_A_idx_full,ax=ax,point_size=point_size)
                    if plot_target_state:
                        for zz in valid_fate_list[0]:
                            idx_2=state_annote==zz
                            ax.plot(x_emb[idx_2],y_emb[idx_2],'.',color='cyan',markersize=point_size*1,alpha=target_transparency)
                    ax.set_title(f'Group A')
                    ax.axis('off')
                    
                    ax = plt.subplot(nrow, ncol, 2)
                    customized_embedding(x_emb,y_emb,group_B_idx_full,ax=ax,point_size=point_size)
                    if plot_target_state:
                        for zz in valid_fate_list[1]:
                            idx_2=state_annote==zz
                            ax.plot(x_emb[idx_2],y_emb[idx_2],'.',color='cyan',markersize=point_size*1,alpha=target_transparency)
                    ax.set_title(f'Group B')
                    ax.axis('off')
                    
                    plt.tight_layout()
                    if savefig:
                        fig.savefig(f'{figure_path}/ancestor_state_groups.{settings.file_format_figs}')


                #diff_gene_A,diff_gene_B=differential_genes(adata,plot_groups=plot_groups,gene_N=gene_N,plot_gene_N=plot_gene_N,savefig=savefig,point_size=point_size)
        
                if 'dynamic_trajectory' not in adata.uns.keys():
                    adata.uns['dynamic_trajectory']={}

                # store the trajectory
                temp_list=[group_A_idx_full,group_B_idx_full]
                for j, fate_name in enumerate(mega_cluster_list):
                    selected_idx=sel_index_list[j]
                    #fate_name,selected_idx=flexible_selecting_cells(adata,selected_fate)
                    combined_prob=temp_list[j].astype(int)+selected_idx.astype(int)

                    if map_backwards:
                        adata.uns['dynamic_trajectory'][fate_name]={'map_backwards':combined_prob} # include both the targeted fate cluster and the inferred earlier states
                    else:
                        adata.uns['dynamic_trajectory'][fate_name]={'map_forward':combined_prob} 



def dynamic_trajectory_via_iterative_mapping(adata,selected_fate,used_map_name='transition_map',
    map_backwards=True,map_threshold=0.1,plot_separately=False,
    apply_time_constaint=False,color_bar=True):
    """
    Infer trajectory towards/from a cluster

    If map_backwards=True, infer the trajectory backward in time. 
    Using inferred transition map, the inference is applied recursively. It
    starts with the cell states for the selected fate and uses the selected 
    map to infer the immediate ancestor states. Then, using these putative 
    ancestor states as the secondary input, it finds their immediate ancestors 
    again. This goes on until all time points are exhausted.

    If map_backwards=False, infer the trajectory forward in time. 

    It only works for transition map from multi-time clones.
    
    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_fate: `str`, or `list`
        Targeted cluster of the trajectory, as consistent with adata.obs['state_info']
        When it is a list, the listed clusters are combined into a single fate cluster. 
    used_map_name: `str`
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map',...}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    map_backwards: `bool`, optional (default: True)
        If `map_backwards=True`, analyze transitions backward in time, 
        which plots initial cell state (rows of Tmap, at t1) profiles towards 
        given fate clusters; otherwise, analyze forward transitions and show later cell 
        states (columns of Tmap, at t2) profiles starting from given initial clusters. 
    plot_separately: `bool`, optional (default: False)
        Plot the inferred trajecotry separately for each time point.
    map_threshold: `float`, optional (default: 0.1)
        Relative threshold in the range [0,1] for truncating the fate map 
        towards the cluster. Only states above the threshold will be selected.
    apply_time_constaint: `bool`, optional (default: False)
        If true, in each iteration of finding the immediate ancestor states, select cell states
        at the corresponding time point.  
    color_bar: `bool`, optional (default: True)

    Returns
    -------
    Results are stored at adata.uns['dynamic_trajectory'] 
    """        

    # We always use the probabilistic map, which is more realiable. Otherwise, the result is very sensitive to thresholding
    #transition_map=adata.uns['transition_map']
    #demultiplexed_map=adata.uns['demultiplexed_map']
    hf.check_available_map(adata)
    fig_width=settings.fig_width; fig_height=settings.fig_height; point_size=settings.fig_point_size

    if used_map_name not in adata.uns['available_map']:
        logg.error(f"used_map_name should be among {adata.uns['available_map']}")

    else:

        state_annote_0=np.array(adata.obs['state_info'])
        if map_backwards:
            cell_id_t1=adata.uns['Tmap_cell_id_t1']
            cell_id_t2=adata.uns['Tmap_cell_id_t2']

        else:
            cell_id_t2=adata.uns['Tmap_cell_id_t1']
            cell_id_t1=adata.uns['Tmap_cell_id_t2']

        time_info=np.array(adata.obs['time_info'])
        x_emb=adata.obsm['X_emb'][:,0]
        y_emb=adata.obsm['X_emb'][:,1]
        data_des=adata.uns['data_des'][-1]
        figure_path=settings.figure_path


        ##### we normalize the map in advance to avoid normalization later in mapout_trajectories
        used_map_0=adata.uns[used_map_name]
        resol=10**(-10)
        used_map_0=hf.sparse_rowwise_multiply(used_map_0,1/(resol+np.sum(used_map_0,1).A.flatten()))

        if map_backwards:
            used_map=used_map_0
        else:
            used_map=used_map_0.T

        #fate_name,selected_idx=flexible_selecting_cells(adata,selected_fate)
        mega_cluster_list,valid_fate_list,fate_array_flat,sel_index_list=hf.analyze_selected_fates([selected_fate],adata.obs['state_info'])
        if len(mega_cluster_list)==0:
            logg.error("No cells selected. Computation aborted!")
        else:
            fate_name=mega_cluster_list[0]
            selected_idx=sel_index_list[0]


            sort_time_info=np.sort(list(set(time_info)))[::-1]


            prob_0r=selected_idx
            if apply_time_constaint:
                prob_0r=prob_0r*(time_info==sort_time_info[0])

            prob_0r_temp=prob_0r>0
            prob_0r_0=prob_0r_temp.copy()
            prob_array=[]

            #used_map=hf.sparse_column_multiply(used_map,1/(resol+used_map.sum(0)))
            for j,t_0 in enumerate(sort_time_info[1:]):
                prob_1r_full=np.zeros(len(x_emb))

                prob_1r_full[cell_id_t1]=hf.mapout_trajectories(used_map,prob_0r_temp,threshold=map_threshold,cell_id_t1=cell_id_t1,cell_id_t2=cell_id_t2)

                ## thresholding the result 
                prob_1r_full=prob_1r_full*(prob_1r_full>map_threshold*np.max(prob_1r_full))

                if apply_time_constaint:
                    prob_1r_full=prob_1r_full*(time_info==t_0)

                # rescale
                prob_1r_full=prob_1r_full/(np.max(prob_1r_full)+resol)

                prob_array.append(prob_1r_full)
                prob_0r_temp=prob_1r_full


            cumu_prob=np.array(prob_array).sum(0)
            ### plot the results
            if plot_separately:
                col=len(sort_time_info);
                row=1
                fig = plt.figure(figsize=(fig_width * col, fig_height * row))
                ax0=plt.subplot(row,col,1)
                if apply_time_constaint:
                    customized_embedding(x_emb,y_emb,prob_0r_0,ax=ax0,point_size=point_size,title=f"Initial: t={sort_time_info[0]}");
                    for k,t_0 in enumerate(sort_time_info[1:]):
                        ax1=plt.subplot(row,col,2+k)
                        customized_embedding(x_emb,y_emb,prob_array[k],ax=ax1,point_size=point_size,title=f"t={t_0}")

                else:
                    customized_embedding(x_emb,y_emb,prob_0r_0,ax=ax0,point_size=point_size,title=f"Initial");
                    for k,t_0 in enumerate(sort_time_info[1:]):
                        ax1=plt.subplot(row,col,2+k)
                        customized_embedding(x_emb,y_emb,prob_array[k],ax=ax1,point_size=point_size,title=f"{k+1}-th mapping")

                fig.savefig(f'{figure_path}/{data_des}_predicting_fate_trajectory_separate_BW{map_backwards}.{settings.file_format_figs}')  
            else:

                col=2; row=1
                fig = plt.figure(figsize=(fig_width * col, fig_height * row))
                ax0=plt.subplot(row,col,1)
                customized_embedding(x_emb,y_emb,prob_0r_0,ax=ax0,point_size=point_size,title=f"Initial: t={sort_time_info[0]}");

                ax1=plt.subplot(row,col,2)
                customized_embedding(x_emb,y_emb,cumu_prob,ax=ax1,point_size=point_size,title=f"All time")

                fig.savefig(f'{figure_path}/{data_des}_predicting_fate_trajectory_allTime_BW{map_backwards}.{settings.file_format_figs}')

            if color_bar:
                fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Reds), ax=ax1,label='Fate Probability')

            if 'dynamic_trajectory' not in adata.uns.keys():
                adata.uns['dynamic_trajectory']={}

            combined_prob=cumu_prob+prob_0r
            if map_backwards:
                adata.uns['dynamic_trajectory'][fate_name]={'map_backwards':combined_prob} # include both the targeted fate cluster and the inferred earlier states
            else:
                adata.uns['dynamic_trajectory'][fate_name]={'map_forward':combined_prob} 


def gene_expression_dynamics(adata,selected_fate,gene_name_list,traj_threshold=0.1,
    map_backwards=True,invert_PseudoTime=False,include_target_states=True,
    compute_new=True,gene_exp_percentile=99,n_neighbors=8,
    plot_raw_data=False,stat_smooth_method='loess'):
    """
    Plot gene trend along the inferred dynamic trajectory.

    We assume that the dynamic trajecotry at given specification is already
    available at adata.uns['dynamic_trajectory'], which can be created via
    :func:`.dynamic_trajectory_via_iterative_mapping` or
    :func:`.dynamic_trajectory_from_binary_fate_bias`.

    Using the states that belong to the trajectory, it computes the pseudo time 
    for these states and shows expression dynamics of selected genes along 
    this pseudo time. 

    Specifically, we first construct KNN graph, compute spectral embedding,
    and take the first component as the pseudo time. To create dynamics for a
    selected gene, we re-weight the expression of this gene at each cell by its
    probability belonging to the trajectory, and rescale the expression at selected
    percentile value. Finally, we fit a curve to the data points. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_fate: `str`, or `list`
        targeted cluster of the trajectory, as consistent with adata.obs['state_info']
        When it is a list, the listed clusters are combined into a single fate cluster. 
    gene_name_list: `list`
        List of genes to plot on the dynamic trajectory. 
    traj_threshold: `float`, optional (default: 0.1), range: (0,1)
        Relative threshold, used to thresholding the inferred dynamic trajecotry to select states. 
    map_backwards: `bool`, optional (default: True)
        If `map_backwards=True`, use the backward trajectory computed before; 
        otherwise, use the forward trajectory computed before.
    invert_PseudoTime: `bool`, optional (default: False)
        If true, invert the pseudotime: 1-pseuotime. This is useful when the direction
        of pseudo time does not agree with intuition.
    include_target_states: `bool`, optional (default: True)
        If true, include the target states to the dynamic trajectory.
    compute_new: `bool`, optional (default: True)
        If true, compute everyting from stratch (as we save computed pseudotime)
    gene_exp_percentile: `int`, optional (default: 99)
        Plot gene expression below this percentile.
    n_neighbors: `int`, optional (default: 8)
        Number of nearest neighbors for constructing KNN graph.
    plot_raw_data: `bool`, optional (default: False)
        Plot the raw gene expression values of each cell along the pseudotime. 
    stat_smooth_method: `str`, optional (default: 'loess')
        Smooth method used in the ggplot. Current available choices are:
        'auto' (Use loess if (n<1000), glm otherwise),
        'lm' (Linear Model),
        'wls' (Linear Model),
        'rlm' (Robust Linear Model),
        'glm' (Generalized linear Model),
        'gls' (Generalized Least Squares),
        'lowess' (Locally Weighted Regression (simple)),
        'loess' (Locally Weighted Regression),
        'mavg' (Moving Average),
        'gpr' (Gaussian Process Regressor)}.

    Returns
    -------
    An adata object with only selected cell states. It can be used for dynamic inference with other packages. 
    """
    
    
    hf.check_available_map(adata)
    fig_width=settings.fig_width; fig_height=settings.fig_height; point_size=settings.fig_point_size

    if len(adata.uns['available_map'])==0:
        logg.error(f"There is no transition map available yet")

    else:

        state_annote_0=np.array(adata.obs['state_info'])
        time_info=np.array(adata.obs['time_info'])

        if map_backwards:
            cell_id_t1=adata.uns['Tmap_cell_id_t1']
            cell_id_t2=adata.uns['Tmap_cell_id_t2']

        else:
            cell_id_t2=adata.uns['Tmap_cell_id_t1']
            cell_id_t1=adata.uns['Tmap_cell_id_t2']

        time_index_t1=np.zeros(len(time_info),dtype=bool)
        time_index_t2=np.zeros(len(time_info),dtype=bool)
        time_index_t1[cell_id_t1]=True
        time_index_t2[cell_id_t2]=True
        

        mega_cluster_list,valid_fate_list,fate_array_flat,sel_index_list=hf.analyze_selected_fates([selected_fate],adata.obs['state_info'])
        if len(mega_cluster_list)==0:
            logg.error("No cells selected. Computation aborted!")
            return adata
        else:
            fate_name=mega_cluster_list[0]
            target_idx=sel_index_list[0]
                
            x_emb=adata.obsm['X_emb'][:,0]
            y_emb=adata.obsm['X_emb'][:,1]
            data_des=adata.uns['data_des'][-1]
            data_path=settings.data_path
            figure_path=settings.figure_path
            file_name=f'{data_path}/{data_des}_fate_trajectory_pseudoTime_{fate_name}_{map_backwards}.npy'


            if map_backwards:
                map_choice='map_backwards'
            else:
                map_choice='map_forward'

            if ('dynamic_trajectory' not in adata.uns.keys()) or (fate_name not in adata.uns['dynamic_trajectory'].keys()) or (map_choice not in adata.uns['dynamic_trajectory'][fate_name].keys()):
                logg.error(f"The target fate trajectory for {fate_name} at map_backwards={map_backwards} have not been inferred yet.\n" 
                    "Please run one of the two dynamic trajectory inference methods first.\n"
                    "like: cs.pl.dynamic_trajectory_from_binary_fate_bias, with the corresponding selected_fate and map_backwards choice.")
                
            else:
                prob_0=np.array(adata.uns['dynamic_trajectory'][fate_name][map_choice])
                
                if not include_target_states:
                    sel_cell_idx=(prob_0>traj_threshold*np.max(prob_0)) & time_index_t1
                else:
                    sel_cell_idx=prob_0>traj_threshold*np.max(prob_0)
                    
                #logg.error(sel_cell_idx)
                sel_cell_id=np.nonzero(sel_cell_idx)[0]


                #file_name=f"data/Pseudotime_{which_branch}_t2.npy"
                if os.path.exists(file_name) and (not compute_new):
                    logg.info("Load pre-computed pseudotime")
                    PseudoTime=np.load(file_name)
                else:
                    
                    from sklearn import manifold
                    data_matrix=adata.obsm['X_pca'][sel_cell_idx]
                    method=manifold.SpectralEmbedding(n_components=1,n_neighbors=n_neighbors)
                    PseudoTime = method.fit_transform(data_matrix)
                    np.save(file_name,PseudoTime)
                    #logg.info("Run time:",time.time()-t)


                PseudoTime=PseudoTime-np.min(PseudoTime)
                PseudoTime=(PseudoTime/np.max(PseudoTime)).flatten()
                
                ## re-order the pseudoTime such that the target fate has the pseudo time 1.
                if invert_PseudoTime:
                    # target_fate_id=np.nonzero(target_idx)[0]
                    # convert_fate_id=hf.converting_id_from_fullSpace_to_subSpace(target_fate_id,sel_cell_id)[0]
                    #if np.mean(PseudoTime[convert_fate_id])<0.5: PseudoTime=1-PseudoTime
                    PseudoTime=1-PseudoTime
                
                #pdb.set_trace()
                if np.sum((PseudoTime>0.25)& (PseudoTime<0.75))==0: # the cell states do not form a contiuum. Plot raw data instead
                    logg.error("The selected cell states do not form a connected graph. Cannot form a continuum of pseudoTime. Only plot the raw data")
                    plot_raw_data=True

                ## plot the pseudotime ordering
                fig = plt.figure(figsize=(fig_width *2,fig_height))
                ax=plt.subplot(1,2,1)
                customized_embedding(x_emb,y_emb,sel_cell_idx,ax=ax,title='Selected cells',point_size=point_size)
                ax1=plt.subplot(1,2,2)
                customized_embedding(x_emb[sel_cell_idx],y_emb[sel_cell_idx],PseudoTime,ax=ax1,title='Pseudo Time',point_size=point_size)
                #customized_embedding(x_emb[final_id],y_emb[final_id],PseudoTime,ax=ax1,title='Pseudo time')
                Clb=fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Reds), ax=ax1,label='Pseudo time')
                fig.savefig(f'{figure_path}/{data_des}_fate_trajectory_pseudoTime_{fate_name}_{map_backwards}.{settings.file_format_figs}')

                temp_dict={'PseudoTime':PseudoTime}
                for gene_name in gene_name_list:
                    yy_max=np.percentile(adata.obs_vector(gene_name),gene_exp_percentile) # global blackground
                    yy=np.array(adata.obs_vector(gene_name)[sel_cell_idx])
                    rescaled_yy=yy*prob_0[sel_cell_idx]/yy_max # rescaled by global background
                    temp_dict[gene_name]=rescaled_yy
                
                
                data2=pd.DataFrame(temp_dict)
                data2_melt=pd.melt(data2,id_vars=['PseudoTime'],value_vars=gene_name_list)
                gplot=ggplot(data=data2_melt,mapping=aes(x="PseudoTime", y='value',color='variable')) + \
                (geom_point() if plot_raw_data else stat_smooth(method=stat_smooth_method)) +\
                theme_classic()+\
                labs(x="Pseudo time",
                     y="Normalized gene expression",
                      color="Gene name")

                gplot.save(f'{figure_path}/{data_des}_fate_trajectory_pseutoTime_gene_expression_{fate_name}_{map_backwards}.{settings.file_format_figs}',width=fig_width, height=fig_height,verbose=False)
                gplot.draw()

                return adata[sel_cell_idx]


##################

# Clone related #

##################


def clones_on_manifold(adata,selected_clone_list=[0],clone_point_size=12,
    color_list=['red','blue','purple','green','cyan','black'],selected_time_points=[],title=True):
    """
    Plot clones on top of state embedding.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    selected_clone_list: `list`
        List of selected clone ID's.
    clone_point_size: `int`, optional (default: 12)
        Relative size of the data point belonging to a clone, 
        as compared to other background points.
    color_list: `list`, optional (default: ['red','blue','purple','green','cyan','black'])
        The list of color that defines color at respective time points. 
    selected_time_points: `list`, optional (default: all)
        Select time points to show corresponding states. If set to be [], use all states. 
    title: `bool`, optional (default: True)
        If ture, show the clone id as panel title. 
    """

    fig_width=settings.fig_width; fig_height=settings.fig_height; point_size=settings.fig_point_size
    x_emb=adata.obsm['X_emb'][:,0]
    y_emb=adata.obsm['X_emb'][:,1]
    data_des=adata.uns['data_des'][-1]
    #data_path=settings.data_path
    figure_path=settings.figure_path
    clone_annot=adata.obsm['X_clone']
    time_info=np.array(adata.obs['time_info'])

    # use only valid time points
    sp_idx=hf.selecting_cells_by_time_points(time_info,selected_time_points)
    selected_time_points=np.sort(list(set(time_info[sp_idx])))

    selected_clone_list=np.array(selected_clone_list)
    full_id_list=np.arange(clone_annot.shape[1])
    valid_idx=np.in1d(full_id_list,selected_clone_list)
    if np.sum(valid_idx)<len(selected_clone_list):
        logg.error(f"Valid id range is (0,{clone_annot.shape[1]-1}). Please use a smaller ID!")
        selected_clone_list=full_id_list[valid_idx]

    if len(selected_clone_list)==0:
        logg.error("No valid states selected.")
    else:
        # using all data
        for my_id in selected_clone_list:
            fig = plt.figure(figsize=(fig_width, fig_height))
            ax=plt.subplot(1,1,1)
            idx_t=np.zeros(len(time_info),dtype=bool)
            for j, xx in enumerate(selected_time_points):
                idx_t0=time_info==selected_time_points[j]
                idx_t=idx_t0 | idx_t
            
            customized_embedding(x_emb[idx_t],y_emb[idx_t],np.zeros(len(y_emb[idx_t])),ax=ax,point_size=point_size)
            for j, xx in enumerate(selected_time_points):
                idx_t=time_info==selected_time_points[j]
                idx_clone=clone_annot[:,my_id].A.flatten()>0
                idx=idx_t & idx_clone
                ax.plot(x_emb[idx],y_emb[idx],'.',color=color_list[j%len(color_list)],markersize=clone_point_size*point_size,markeredgecolor='white',markeredgewidth=point_size)

                if title:
                    ax.set_title(f'ID: {my_id}')

            fig.savefig(f'{figure_path}/{data_des}_different_clones_{my_id}.{settings.file_format_figs}')



def clonal_fate_bias(adata,selected_fate='',clone_size_thresh=3,
    N_resampling=1000,compute_new=True,show_histogram=True):
    """
    Plot clonal fate bias towards a cluster.

    This is just -log(P-value), where P-value is for the observation 
    cell fraction of a clone in the targeted cluster as compared to 
    randomized clones, where the randomized sampling produces clones 
    of the same size as the targeted clone. The computed results will 
    be saved at the directory settings.data_path.

    This function can be time-consuming. The time cost scales linearly
    with the number of resampling and the number of clones. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    selected_fate: `str`
        The targeted fate cluster, from adata.obs['state_info'].
    clone_size_thresh: `int`, optional (default: 3)
        Clones with size >= this threshold will be highlighted in 
        the plot in red.
    N_resampling: `int`, optional (default: 1000)
        The number of randomized sampling for assessing the P-value of a clone. 
    compute_new: `bool`, optional (default: True)
        Compute from scratch, regardless of existing saved files. 
    show_histogram: `bool`, optional (default: True)
        If true, show the distribution of inferred fate probability.

    Returns
    -------
    fate_bias: `np.array`
        Computed clonal fate bias.
    sort_idx: `np.array`
        Corresponding clone id list. 
    """

    fig_width=settings.fig_width; fig_height=settings.fig_height; point_size=settings.fig_point_size
    state_annote_0=adata.obs['state_info']
    data_des=adata.uns['data_des'][-1]
    clonal_matrix=adata.obsm['X_clone']
    data_path=settings.data_path
    data_des=adata.uns['data_des'][-1]
    figure_path=settings.figure_path
    state_list=list(set(state_annote_0))


    valid_clone_idx=(clonal_matrix.sum(0).A.flatten()>=2)
    valid_clone_id=np.nonzero(valid_clone_idx)[0]
    sel_cell_idx=(clonal_matrix[:,valid_clone_idx].sum(1).A>0).flatten()
    sel_cell_id=np.nonzero(sel_cell_idx)[0]
    clone_N=np.sum(valid_clone_idx)
    cell_N=len(sel_cell_id)

    ## annotate the state_annot
    clonal_matrix_new=clonal_matrix[sel_cell_id][:,valid_clone_id]
    state_annote_new=state_annote_0[sel_cell_id]
    hit_target=np.zeros(len(sel_cell_id),dtype=bool)

    mega_cluster_list,valid_fate_list,fate_array_flat,sel_index_list=hf.analyze_selected_fates([selected_fate],state_annote_new)
    if len(mega_cluster_list)==0:
        logg.error("No cells selected. Computation aborted!")
        return None, None 
    else:
        fate_name=mega_cluster_list[0]
        target_idx=sel_index_list[0]

        file_name=f'{data_path}/{data_des}_clonal_fate_bias_{N_resampling}_{fate_name}.npz'

        if (not os.path.exists(file_name)) or compute_new:

            ## target clone
            target_ratio_array=np.zeros(clone_N)


            null_ratio_array=np.zeros((clone_N,N_resampling))
            P_value_up=np.zeros(clone_N)
            P_value_down=np.zeros(clone_N)
            P_value=np.zeros(clone_N)
            P_value_rsp=np.zeros((clone_N,N_resampling))

            for m in range(clone_N):
                if m%50==0:
                    logg.info(f"Current clone id: {m}")
                target_cell_idx=(clonal_matrix_new[:,m].sum(1).A>0).flatten()
                target_clone_size=np.sum(target_cell_idx) 

                if target_clone_size>0:
                    target_ratio=np.sum(target_idx[target_cell_idx])/target_clone_size
                    target_ratio_array[m]=target_ratio
                    #N_resampling=int(np.floor(cell_N/target_clone_size))


                    sel_cell_id_copy=list(np.arange(cell_N))

                    for j in range(N_resampling):
                        temp_id=np.random.choice(sel_cell_id_copy,size=target_clone_size,replace=False)
                        null_ratio_array[m,j]=np.sum(target_idx[temp_id])/target_clone_size


                    ## reprogamming clone
                    P_value_up[m]=np.sum(null_ratio_array[m]>=target_ratio)/N_resampling
                    P_value_down[m]=np.sum(null_ratio_array[m]<=target_ratio)/N_resampling
                    P_value[m]=np.min([P_value_up[m],P_value_down[m]])

                    for j1,zz in enumerate(null_ratio_array[m]):
                        P_value_up_rsp=np.sum(null_ratio_array[m]>=zz)/N_resampling
                        P_value_down_rsp=np.sum(null_ratio_array[m]<=zz)/N_resampling
                        P_value_rsp[m,j1]=np.min([P_value_up_rsp,P_value_down_rsp])            


            np.savez(file_name,P_value_rsp=P_value_rsp,P_value=P_value)

        else:
            saved_data=np.load(file_name,allow_pickle=True)
            P_value_rsp=saved_data['P_value_rsp']
            P_value=saved_data['P_value']
            #target_ratio_array=saved_data['target_ratio_array']


        ####### Plotting
        clone_size_array=clonal_matrix_new.sum(0).A.flatten()

        resol=1/N_resampling
        P_value_rsp_new=P_value_rsp.reshape((clone_N*N_resampling,))
        sort_idx=np.argsort(P_value_rsp_new)
        P_value_rsp_new=P_value_rsp_new[sort_idx]+resol
        sel_idx=((np.arange(clone_N)+1)*len(sort_idx)/clone_N).astype(int)-1
        fate_bias_rsp=-np.log10(P_value_rsp_new[sel_idx])

        sort_idx=np.argsort(P_value)
        P_value=P_value[sort_idx]+resol
        fate_bias=-np.log10(P_value)
        idx=clone_size_array[sort_idx]>=clone_size_thresh


        fig=plt.figure(figsize=(fig_width,fig_height));ax=plt.subplot(1,1,1)
        ax.plot(np.arange(len(fate_bias))[~idx],fate_bias[~idx],'.',color='blue',markersize=5,label=f'Size $<$ {int(clone_size_thresh)}')#,markeredgecolor='black',markeredgewidth=0.2)
        ax.plot(np.arange(len(fate_bias))[idx],fate_bias[idx],'.',color='red',markersize=5,label=f'Size $\ge$ {int(clone_size_thresh)}')#,markeredgecolor='black',markeredgewidth=0.2)
        ax.plot(np.arange(len(fate_bias_rsp)),fate_bias_rsp,'.',color='grey',markersize=5,label='Randomized')#,markeredgecolor='black',markeredgewidth=0.2)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel('Clone rank')
        plt.rc('text', usetex=True)
        #ax.set_ylabel('Fate bias ($-\\log_{10}P_{value}$)')
        ax.set_ylabel('Clonal fate bias')
        ax.legend()
        #ax.set_xlim([0,0.8])
        fig.tight_layout()
        fig.savefig(f'{figure_path}/{data_des}_clonal_fate_bias.{settings.file_format_figs}')
        plt.rc('text', usetex=False)
        #plt.show()


        if show_histogram:
            target_fraction_array=(clonal_matrix_new.T*target_idx)/clone_size_array
            fig=plt.figure(figsize=(fig_width,fig_height));ax=plt.subplot(1,1,1)
            ax.hist(target_fraction_array,color='#2ca02c',density=True)
            ax.set_xlim([0,1])
            ax.set_xlabel('Clonal fraction in selected fates')
            ax.set_ylabel('Density')
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_title(f'Average: {int(np.mean(target_fraction_array)*100)/100};   Expect: {int(np.mean(target_idx)*100)/100}')
            fig.savefig(f'{figure_path}/{data_des}_observed_clonal_fraction.{settings.file_format_figs}')

        return fate_bias,sort_idx



def heatmap(figure_path, X, variable_names,color_bar_label='cov',data_des='',color_bar=True):
    """
    Plot heat map of a two-dimensional matrix X.

    Parameters
    ----------
    figure_path: `str`
        path to save figures
    X: `np.array`
        The two-dimensional matrix to plot
    variable_names: `list`
        List of variable names
    color_bar_label: `str`, optional (default: 'cov')
        Color bar label
    data_des: `str`, optional (default: '')
        String to distinguish different saved objects.
    color_bar: `bool`, optional (default: True)
        If true, plot the color bar.
    """

    fig_width=settings.fig_width; fig_height=settings.fig_height; point_size=settings.fig_point_size
    vmax = (np.percentile(X-np.diag(np.diag(X)),95) + np.percentile(X-np.diag(np.diag(X)),98))/2
    #vmax=np.max(X)
    plt.imshow(X, vmax=vmax)
    plt.xticks(np.arange(X.shape[0])+.4, variable_names, ha='right', rotation=60);
    plt.yticks(np.arange(X.shape[0])+.4, variable_names);
    if color_bar:
        cbar = plt.colorbar()
        cbar.set_label(color_bar_label, rotation=270, labelpad=20)
        plt.gcf().set_size_inches((6,4))
    else:
        plt.gcf().set_size_inches((4,4))
    plt.tight_layout()
    plt.savefig(figure_path+f'/{data_des}_heat_map.{settings.file_format_figs}')



def ordered_heatmap(figure_path, data_matrix, variable_names,int_seed=10,
    data_des='',log_transform=False):
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

    o = hf.get_hierch_order(data_matrix)
    plt.figure(int_seed)
    
    if log_transform:
        plt.imshow(data_matrix[o,:], aspect='auto',cmap=plt.cm.Reds, vmax=1)
    else:
        plt.imshow(np.log(data_matrix[o,:]+1)/np.log(10), aspect='auto',cmap=plt.cm.Reds, vmin=0,vmax=1)
            
    plt.xticks(np.arange(data_matrix.shape[1])+.4, variable_names, rotation=70, ha='right')
    plt.yticks([])
    cbar = plt.colorbar()
    cbar.set_label('Number of barcodes (log10)', rotation=270, labelpad=20)
    plt.gcf().set_size_inches((4,6))
    plt.tight_layout()
    plt.savefig(figure_path+f'/{data_des}_data_matrix.{settings.file_format_figs}')



def barcode_heatmap(adata,selected_time_point,selected_fates=[],color_bar=True,rename_selected_fates=[]):
    """
    Plot barcode heatmap among different fate clusters.

    We select one time point with clonal measurement and show the 
    heatmap of barcodes among selected fate clusters. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    selected_time_point: `str`
        Time point to select the cell states.
    selected_fates: `list`, optional (default: all)
        List of fate clusters to use. If set to be [], use all.
    color_bar: `bool`, optional (default: True)
        Plot color bar. 
    rename_selected_fates: `list`, optional (default: [])
        Provide new names in substitution of names in selected_fates.
        For this to be effective, the new name list needs to have names 
        in exact correspondence to those in the old list. 
    """

    time_info=np.array(adata.obs['time_info'])
    sp_idx=hf.selecting_cells_by_time_points(time_info,[selected_time_point])
    clone_annot=adata[sp_idx].obsm['X_clone']
    state_annote=adata[sp_idx].obs['state_info']

    if (np.sum(sp_idx)==0):
        logg.error("No cells selected. Computation aborted!")
    else:
        mega_cluster_list,valid_fate_list,fate_array_flat,sel_index_list=hf.analyze_selected_fates(selected_fates,state_annote)
        if (len(mega_cluster_list)==0):
            logg.error("No cells selected. Computation aborted!")
        else:
            x_emb=adata.obsm['X_emb'][:,0]
            y_emb=adata.obsm['X_emb'][:,1]
            data_des=adata.uns['data_des'][-1]
            data_des=f'{data_des}_clonal'
            figure_path=settings.figure_path

            coarse_clone_annot=np.zeros((len(mega_cluster_list),clone_annot.shape[1]))
            for j, idx in enumerate(sel_index_list):
                coarse_clone_annot[j,:]=clone_annot[idx].sum(0)

            if len(rename_selected_fates)!=len(mega_cluster_list):
                rename_selected_fates=mega_cluster_list


            ordered_heatmap(figure_path, coarse_clone_annot.T, rename_selected_fates,data_des=data_des)




def fate_coupling_from_clones(adata,selected_time_point,selected_fates=[],color_bar=True,rename_selected_fates=[]):
    """
    Plot fate coupling based on clonal information.

    We select one time point with clonal measurement and show the normalized 
    clonal covariance among these fates.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    selected_time_point: `str`
        Time point to select the cell states.
    selected_fates: `list`, optional (default: all)
        List of fate clusters to use. If set to be [], use all.
    color_bar: `bool`, optional (default: True)
        Plot color bar. 
    rename_selected_fates: `list`, optional (default: [])
        Provide new names in substitution of names in selected_fates.
        For this to be effective, the new name list needs to have names 
        in exact correspondence to those in the old list. 
    """

    time_info=np.array(adata.obs['time_info'])
    sp_idx=hf.selecting_cells_by_time_points(time_info,[selected_time_point])
    clone_annot=adata[sp_idx].obsm['X_clone']
    state_annote=adata[sp_idx].obs['state_info']

    if (np.sum(sp_idx)==0):
        logg.error("No cells selected. Computation aborted!")
    else:
        mega_cluster_list,valid_fate_list,fate_array_flat,sel_index_list=hf.analyze_selected_fates(selected_fates,state_annote)
        if (len(mega_cluster_list)==0):
            logg.error("No cells selected. Computation aborted!")
        else:
            x_emb=adata.obsm['X_emb'][:,0]
            y_emb=adata.obsm['X_emb'][:,1]
            data_des=adata.uns['data_des'][-1]
            data_des=f'{data_des}_clonal_fate_coupling'
            figure_path=settings.figure_path

            coarse_clone_annot=np.zeros((len(mega_cluster_list),clone_annot.shape[1]))
            for j, idx in enumerate(sel_index_list):
                coarse_clone_annot[j,:]=clone_annot[idx].sum(0)

            if len(rename_selected_fates)!=len(mega_cluster_list):
                rename_selected_fates=mega_cluster_list

            X_weinreb = hf.get_normalized_covariance(coarse_clone_annot.T,method='Weinreb')
            heatmap(figure_path, X_weinreb, rename_selected_fates,color_bar_label='Coupling',color_bar=color_bar,data_des=data_des)




