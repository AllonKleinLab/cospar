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
import statsmodels.sandbox.stats.multicomp
import scipy.stats as stats
import matplotlib as mpl
from ete3 import Tree

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

    fig_height=settings.fig_height
    fig_width=settings.fig_width
    data_des=adata.uns['data_des'][-1]
    fig=plt.figure(figsize=(fig_width,fig_height));
    ax=plt.subplot(1,1,1)

    flag=False
    if color in adata.obs.keys(): flag=True
    if color in adata.var_names: flag=True

    if flag:
        sc.pl.embedding(adata,basis=basis,color=color,ax=ax)
        plt.tight_layout()
        fig.savefig(f'{settings.figure_path}/{data_des}_embedding.png', dpi=300)
    else:
        logg.error(f'Could not find key {color} in .var_names or .obs.columns.')

# This one show correct color bar range
def customized_embedding(x, y, vector, normalize=False, title=None, ax=None, 
    order_points=True, set_ticks=False, col_range=None, buffer_pct=0.03, point_size=1, 
    color_map=None, smooth_operator=None,set_lim=True,
    vmax=np.nan,vmin=np.nan,color_bar=False,color_bar_label=''):
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
    col_range: `tuple`, optional (default: None)
        The default setting is to plot the actual value of the vector. 
        If col_range is set within [0,100], it will plot the percentile of the values,
        and the color_bar will show range [0,1]. This re-scaling is useful for 
        visualizing gene expression. 
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
        if col_range is None:
            vmin=np.min(coldat)
        else:
            vmin = np.percentile(coldat, col_range[0])
        
    if np.isnan(vmax):
        if col_range is None:
            vmax=np.max(coldat)
        else:
            vmax = np.percentile(coldat, col_range[1])
        

    if vmax == vmin:
        vmax = coldat.max()
        
    #print(f'vmax: {vmax}; vmin: {vmin}, coldat: {coldat}')
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
        
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=color_map), ax=ax,label=color_bar_label)



def gene_expression_on_manifold(adata,selected_genes,savefig=False,selected_times=None,color_bar=False):
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
    selected_times: `list`, optional (default: all)
        A list of time points to further restrict the cell states to plot. 
        If map_backward=True, plot initial states that are among these time points;
        otherwise, show later states that are among these time points.
    color_bar: `bool`, (default: False)
        If True, plot the color bar. 
    """


    selected_genes=list(selected_genes)
    fig_width=settings.fig_width; fig_height=settings.fig_height; point_size=settings.fig_point_size
    if color_bar: fig_width=fig_width+0.5
        
    if type(selected_genes)==str:
        selected_genes=[selected_genes]

    x_emb=adata.obsm['X_emb'][:,0]
    y_emb=adata.obsm['X_emb'][:,1]
    figure_path=settings.figure_path

    time_info=np.array(adata.obs['time_info'])
    if selected_times is not None:
        sp_idx=np.zeros(adata.shape[0],dtype=bool)
        for xx in selected_times:
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
            cbar=plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Reds), ax=ax)
            cbar.set_label('Normalized expression')

        fig.tight_layout()

        if savefig:
            fig.savefig(f'{figure_path}/lung_marker_genes_{selected_genes[j]}.{settings.file_format_figs}')


####################

## Fate bias analysis

####################

def single_cell_transition(adata,selected_state_id_list,used_Tmap='transition_map',map_backward=True,savefig=False,initial_point_size=3,color_bar=True):
    """
    Plot transition probability from given initial cell states.

    If `map_backward=True`, plot the probability :math:`T_{ij}` over initial states :math:`i` 
    at a given later state :math:`j`. Otherwise, plot the probability :math:`T_{ij}` 
    over later states :math:`j` at a fixed initial state :math:`i`.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_state_id_list: `list`
        List of cell id's. Like [0,1,2].
    used_Tmap: `str`
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map',...}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    map_backward: `bool`, optional (default: True)
        If `map_backward=True`, plot the probability of source states where the current cell state comes from;
        otherwise, plot future state probability starting from given initial state.
    initial_point_size: `int`, optional (default: 3)
        Relative size of the data point for the selected cells.
    save_fig: `bool`, optional (default: False)
        If true, save figure to defined directory at settings.figure_path
    color_bar: `bool`, optional (default: True)
        Plot the color bar. 
    """

    hf.check_available_map(adata)
    fig_width=settings.fig_width; fig_height=settings.fig_height; point_size=settings.fig_point_size
    if color_bar: fig_width=fig_width+0.5
    #set_up_plotting()


    if used_Tmap not in adata.uns['available_map']:
        logg.error(f"used_Tmap should be among {adata.uns['available_map']}")

    else:
        state_annote=adata.obs['state_info']
        x_emb=adata.obsm['X_emb'][:,0]
        y_emb=adata.obsm['X_emb'][:,1]
        data_des=adata.uns['data_des'][-1]
        figure_path=settings.figure_path

        if not map_backward:
            cell_id_t1=adata.uns['Tmap_cell_id_t1']
            cell_id_t2=adata.uns['Tmap_cell_id_t2']
            Tmap=adata.uns[used_Tmap]

        else:
            cell_id_t2=adata.uns['Tmap_cell_id_t1']
            cell_id_t1=adata.uns['Tmap_cell_id_t2']
            Tmap=adata.uns[used_Tmap].T

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
                    prob_vec=prob_vec/np.max(prob_vec)
                    customized_embedding(x_emb, y_emb, prob_vec, point_size=point_size, ax=ax0,color_bar=color_bar,color_bar_label='Probability')
                    
                    ax0.plot(x_emb[cell_id_t1][target_cell_ID],y_emb[cell_id_t1][target_cell_ID],'*b',markersize=initial_point_size*point_size)

                    #ax0.set_title(f"t1 state (blue star) ({cell_id_t1[target_cell_ID]})")
                    if map_backward:
                        ax0.set_title(f"ID (t2): {target_cell_ID}")
                    else:
                        ax0.set_title(f"ID (t1): {target_cell_ID}")
                    #plt.rc('text', usetex=True)

            # if color_bar:
            #     Clb=fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Reds), ax=ax0,label='Probability')

            plt.tight_layout()
            if savefig:
                fig.savefig(f"{figure_path}/plotting_transition_map_probability_{map_backward}.{settings.file_format_figs}")
            #plt.rc('text', usetex=False)


def fate_map(adata,selected_fates=None,used_Tmap='transition_map',
    map_backward=True,method='norm-sum', selected_times=None,
    background=True, show_histogram=False,
    plot_target_state=True,auto_color_scale=True,color_bar=True,
    target_transparency=0.2,horizontal=False,figure_index=''):
    """
    Plot transition probability to given fate/ancestor clusters.

    Given a transition map :math:`T_{ij}`, we explore build
    the fate map :math:`P_i(\mathcal{C})` towards a set of states annotated with 
    fate :math:`\mathcal{C}` in the following ways. 

    Step 1: Map normalization: :math:`T_{ij}\leftarrow T_{ij}/\sum_j T_{ij}`. 

    Step 2: If `map_backward=False`, perform matrix transpose :math:`T_{ij} \leftarrow T_{ji}`.

    Step 3: aggregate fate probabiliteis within a given cluster :math:`\mathcal{C}`: 

    * method='sum': :math:`P_i(\mathcal{C})=\sum_{j\in \mathcal{C}} T_{ij}`.
      This gives the intuitive meaning of fate probability.

    * method='norm-sum': We normalize the map from 'sum' method within a cluster, i.e. 
      :math:`P_i(\mathcal{C})\leftarrow P_i(\mathcal{C})/\sum_j P_j(\mathcal{C})`. 
      This gives the probability that a fate cluster :math:`\mathcal{C}` originates 
      from an initial state :math:`i`.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_fates: `list`, optional (default: all fates)
        List of cluster ids consistent with adata.obs['state_info']. 
        It allows a nested list, where we merge clusters within 
        each sub-list into a mega-fate cluster.
    used_Tmap: `str`, optional (default: 'transition_map')
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map',...}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    map_backward: `bool`, optional (default: True)
        If `map_backward=True`, show fate properties of initial cell states :math:`i`; 
        otherwise, show progenitor properties of later cell states :math:`j`.
        This is used for building the fate map :math:`P_i(\mathcal{C})`. See :func:`.fate_map`.
    method: `str`, optional (default: 'norm-sum')
        Method to obtain the fate probability map :math:`P_i(\mathcal{C})` towards a set 
        of states annotated with fate :math:`\mathcal{C}`. Available options: 
        {'sum', 'norm-sum'}. See :func:`.fate_map`.
    selected_times: `list`, optional (default: all)
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
    adata.uns['fate_map']: `pd.DataFrame`
        The fate map output is attached to the adata object as a dictionary
        {cell_id, fate_probability}. 
    """

    hf.check_available_map(adata)
    fig_width=settings.fig_width; fig_height=settings.fig_height; point_size=settings.fig_point_size
    if color_bar: fig_width=fig_width+0.5
    #set_up_plotting()

    if used_Tmap not in adata.uns['available_map']:
        logg.error(f"used_Tmap should be among {adata.uns['available_map']}")

    else:        
        state_annote=adata.obs['state_info']
        if map_backward:
            cell_id_t1=adata.uns['Tmap_cell_id_t1']
            cell_id_t2=adata.uns['Tmap_cell_id_t2']

        else:
            cell_id_t2=adata.uns['Tmap_cell_id_t1']
            cell_id_t1=adata.uns['Tmap_cell_id_t2']


        time_info=np.array(adata.obs['time_info'])
        sp_idx=hf.selecting_cells_by_time_points(time_info[cell_id_t1],selected_times)


        x_emb=adata.obsm['X_emb'][:,0]
        y_emb=adata.obsm['X_emb'][:,1]
        data_des=adata.uns['data_des'][-1]
        figure_path=settings.figure_path


        if method=='norm-sum':
            color_bar_label='Progenitor prob.'
        else:
            color_bar_label='Fate probability'



        fate_map,mega_cluster_list,relative_bias,expected_prob,valid_fate_list,sel_index_list,fate_entropy=hf.compute_fate_probability_map(adata,
            selected_fates=selected_fates,used_Tmap=used_Tmap,map_backward=map_backward,method=method,fate_count=False)

        if (len(mega_cluster_list)==0) or (np.sum(sp_idx)==0):
            logg.error("No cells selected. Computation aborted!")
        else:

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
                    customized_embedding(x_emb[cell_id_t1][sp_idx],y_emb[cell_id_t1][sp_idx],vector_array[j][sp_idx],
                        point_size=point_size,ax=ax0,title=description[j],set_lim=False,color_bar=color_bar,color_bar_label=color_bar_label)
                else:
                    customized_embedding(x_emb[cell_id_t1][sp_idx],y_emb[cell_id_t1][sp_idx],vector_array[j][sp_idx],
                        point_size=point_size,ax=ax0,title=description[j],set_lim=False,vmax=1,vmin=0,color_bar=color_bar,color_bar_label=color_bar_label)
            
            plt.tight_layout()
            fig.savefig(f'{figure_path}/{data_des}_fate_map_overview_{description[j]}{figure_index}.{settings.file_format_figs}')

            if show_histogram:
                fig = plt.figure(figsize=(fig_width * col, fig_height * row))
                for j in range(len(vector_array)):
                    temp_array=vector_array[j][sp_idx]
                    new_idx=np.argsort(abs(temp_array-0.5))
                    xxx=temp_array[new_idx]
                    ax = plt.subplot(row, col, j + 1)
                    ax.hist(xxx,50,color='#2ca02c',density=True)
                    ax.set_xlim([0,1])
                    ax.set_xlabel(color_bar_label)
                    ax.set_ylabel('Density')
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.set_title(f'{description[j]}, Ave.: {int(np.mean(xxx)*100)/100}')
                plt.tight_layout()
                fig.savefig(f'{figure_path}/{data_des}_intrinsic_fate_bias_BW{map_backward}_histogram{figure_index}.{settings.file_format_figs}')

            ## save data to adata
            fate_map_dictionary={'cell_id':cell_id_t1[sp_idx]}
            for j,fate in enumerate(mega_cluster_list):
                fate_map_dictionary[fate]=fate_map[sp_idx,j]
            adata.uns['fate_map']=pd.DataFrame(fate_map_dictionary)


def fate_potency(adata,selected_fates=None,used_Tmap='transition_map',
    map_backward=True,method='norm-sum',fate_count=False,
    selected_times=None,background=True, 
    auto_color_scale=True,color_bar=True,figure_index=''):
    """
    Plot fate potency of early cell states for a given set of fates.

    Given a fate map :math:`P_i(\mathcal{C})` towards a set of 
    fate clusters :math:`\{\mathcal{C}_1,\mathcal{C}_2,\mathcal{C}_3,...\}` 
    constructed as in :func:`.fate_map`, we estimate the fate potency of a
    state :math:`i` in the following two ways:

    * fate_count=True: count the number of possible fates (with non-zero fate probabilities) 
      at state :math:`i`, i.e., :math:`\sum_x H\Big(P_i(\mathcal{C}_x)\Big)`, 
      where :math:`H(y)=\{1` for y>0; 0 otherwise}.

    * fate_count=False: calculate the Shannon entropy of the fate probability 
      starting at state :math:`i`, i.e., :math:`-\sum_x P_i(\mathcal{C}_x)\ln P_i(\mathcal{C}_x)`,


    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_fates: `list`, optional (default: all fates)
        List of cluster ids consistent with adata.obs['state_info']. 
        It allows a nested list, where we merge clusters within 
        each sub-list into a mega-fate cluster.
    used_Tmap: `str`, optional (default: 'transition_map')
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map',...}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    map_backward: `bool`, optional (default: True)
        If `map_backward=True`, show fate properties of initial cell states :math:`i`; 
        otherwise, show progenitor properties of later cell states :math:`j`.
        This is used for building the fate map :math:`P_i(\mathcal{C})`. See :func:`.fate_map`.
    method: `str`, optional (default: 'norm-sum')
        Method to obtain the fate probability map :math:`P_i(\mathcal{C})` towards a set 
        of states annotated with fate :math:`\mathcal{C}`. Available options: 
        {'sum', 'norm-sum'}. See :func:`.fate_map`.
    fate_count: `bool`, optional (default: False)
        Used to determine the method for computing the fate potential of a state.
        If ture, jus to count the number of possible fates; otherwise, use the Shannon entropy.
    selected_times: `list`, optional (default: all)
        A list of time points to further restrict the cell states to plot. 
        The default choice is not to constrain the cell states to show. 
    background: `bool`, optional (default: True)
        If true, plot all cell states (t1+t2) in grey as the background. 
    color_bar: `bool`, optional (default: True)
        plot the color bar if True.
    figure_index: `str`, optional (default: '')
        String index for annotate filename for saved figures. Used to distinuigh plots from different conditions. 

    Returns
    -------
    adata.uns['fate_potency']: `pd.DataFrame`
        The fate potency is attached to the adata object as a dictionary
        {cell_id, fate_potency}. 
    """

    hf.check_available_map(adata)
    fig_width=settings.fig_width; fig_height=settings.fig_height; point_size=settings.fig_point_size
    if color_bar: fig_width=fig_width+0.5
    #set_up_plotting()

    if used_Tmap not in adata.uns['available_map']:
        logg.error(f"used_Tmap should be among {adata.uns['available_map']}")

    else:        
        state_annote=adata.obs['state_info']
        if map_backward:
            cell_id_t1=adata.uns['Tmap_cell_id_t1']
            cell_id_t2=adata.uns['Tmap_cell_id_t2']

        else:
            cell_id_t2=adata.uns['Tmap_cell_id_t1']
            cell_id_t1=adata.uns['Tmap_cell_id_t2']


        time_info=np.array(adata.obs['time_info'])
        sp_idx=hf.selecting_cells_by_time_points(time_info[cell_id_t1],selected_times)


        x_emb=adata.obsm['X_emb'][:,0]
        y_emb=adata.obsm['X_emb'][:,1]
        data_des=adata.uns['data_des'][-1]
        figure_path=settings.figure_path


        if fate_count:
            colar_bar_label='Potency (fate number)'
        else:
            colar_bar_label='Potency (fate entropy)'


        fate_map,mega_cluster_list,relative_bias,expected_prob,valid_fate_list,sel_index_list,fate_entropy=hf.compute_fate_probability_map(adata,
            selected_fates=selected_fates,used_Tmap=used_Tmap,map_backward=map_backward,method=method,fate_count=fate_count)

        if (len(mega_cluster_list)==0) or (np.sum(sp_idx)==0):
            logg.error("No cells selected. Computation aborted!")
        else:


            fig = plt.figure(figsize=(fig_width, fig_height))
            ax0=plt.subplot(1,1,1)
                
            if background:
                customized_embedding(x_emb,y_emb,np.zeros(len(y_emb)),point_size=point_size,ax=ax0)            
            else:
                customized_embedding(x_emb[cell_id_t1],y_emb[cell_id_t1],np.zeros(len(y_emb[cell_id_t1])),point_size=point_size,ax=ax0)

            if auto_color_scale:
                customized_embedding(x_emb[cell_id_t1][sp_idx],y_emb[cell_id_t1][sp_idx],fate_entropy[sp_idx],
                    point_size=point_size,ax=ax0,set_lim=False,color_bar=color_bar,color_bar_label=colar_bar_label)
            else:
                customized_embedding(x_emb[cell_id_t1][sp_idx],y_emb[cell_id_t1][sp_idx],fate_entropy[sp_idx],
                    point_size=point_size,ax=ax0,set_lim=False,vmax=1,vmin=0,color_bar=color_bar,color_bar_label=colar_bar_label)
        
            plt.tight_layout()
            fig.savefig(f'{figure_path}/{data_des}_fate_potency{figure_index}.{settings.file_format_figs}')

            ## save data to adata
            adata.uns['fate_potency']=pd.DataFrame({'cell_id':cell_id_t1[sp_idx],'fate_potency':fate_entropy[sp_idx]})


def fate_bias(adata,selected_fates=None,used_Tmap='transition_map',
    map_backward=True,method='norm-sum',
    selected_times=None,sum_fate_prob_thresh=0.05,mask=None,
    plot_target_state=False,color_bar=True,show_histogram=True,pseudo_count=0,
    target_transparency=0.2,figure_index=''):
    """
    Plot fate bias to given two fate clusters (A, B).

    Given a fate map :math:`P_i` towards two fate clusters 
    :math:`\{\mathcal{A}, \mathcal{B}\}`, constructed according 
    to :func:`.fate_map`, we compute the fate bias of state :math:`i` as
    :math:`[P(\mathcal{A})+c_0]/[P(\mathcal{A})+P(\mathcal{B})+2c_0]`,
    where :math:`c_0=a * \max_{i,\mathcal{C}} P_i(\mathcal{C})`
    is a re-scaled pseudocount, with :math:`a` given by pseudo_count. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_fates: `list`
        List of cluster ids consistent with adata.obs['state_info']. 
        It allows a nested structure. If so, we merge clusters within 
        each sub-list into a mega-fate cluster.
    used_Tmap: `str`
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map',...}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    map_backward: `bool`, optional (default: True)
        If `map_backward=True`, show fate properties of initial cell states :math:`i`; 
        otherwise, show progenitor properties of later cell states :math:`j`.
        This is used for building the fate map :math:`P_i(\mathcal{C})`. See :func:`.fate_map`.
    method: `str`, optional (default: 'norm-sum')
        Method to obtain the fate probability map :math:`P_i(\mathcal{C})` towards a set 
        of states annotated with fate :math:`\mathcal{C}`. Available options: 
        {'sum', 'norm-sum'}. See :func:`.fate_map`.
    selected_times: `list`, optional (default: all)
        A list of time points to further restrict the cell states to plot. 
        The default choice is not to constrain the cell states to show. 
    sum_fate_prob_thresh: `float`, optional (default: 0.05)
        The fate bias of a state is plotted only when it has a cumulative fate 
        probability to the combined cluster (A+B) larger than this threshold,
        i.e., P(i->A)+P(i+>B) >  sum_fate_prob_thresh. 
    mask: `np.array`, optional (default: None)
        A boolean array for available cell states. It should has the length as adata.shape[0].
        Especially useful to constrain the states to show fate bias.
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
    pseudo_count: `float`, optional (default: 0)
        Pseudo count to compute the fate bias. See above.

    Returns
    -------
    adata.uns['fate_bias']: `pd.DataFrame`
        The fate bias is attached to the adata object as a dictionary
        {cell_id, fate_bias}. 
    """

    hf.check_available_map(adata)
    fig_width=settings.fig_width; fig_height=settings.fig_height; point_size=settings.fig_point_size
    if color_bar: fig_width=fig_width+0.5


    if method=='norm-sum':
        color_bar_label='Progenitor bias'
    else:
        color_bar_label='Fate bias'


    #set_up_plotting()
    if used_Tmap not in adata.uns['available_map']:
        logg.error(f"used_Tmap should be among {adata.uns['available_map']}")

    else:
        state_annote=adata.obs['state_info']
        if map_backward:
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
        sp_idx=hf.selecting_cells_by_time_points(time_info[cell_id_t1],selected_times)
        if mask is not None:
            if len(mask)==adata.shape[0]:
                mask=mask.astype(bool)
                sp_idx=sp_idx & (mask[cell_id_t1])
            else:
                logg.error('mask length does not match adata.shape[0]. Ignored mask.')

        cell_id_t1_sp=cell_id_t1[sp_idx]
            

        if len(selected_fates)!=2: 
            logg.error(f"Must have only two fates")
        else:
            fate_map,mega_cluster_list,relative_bias,expected_prob,valid_fate_list,sel_index_list,fate_entropy=hf.compute_fate_probability_map(adata,
                selected_fates=selected_fates,used_Tmap=used_Tmap,map_backward=map_backward,method=method)

            if (len(mega_cluster_list)!=2) or (np.sum(sp_idx)==0):
                logg.error(f"Do not have valid fates or time points. Computation aborted!")
            else:
                if pseudo_count==0:
                    pseudo_count=10**(-10)

                fig=plt.figure(figsize=(fig_width,fig_height))
                ax=plt.subplot(1,1,1)

                potential_vector_temp=fate_map[sp_idx,:]+pseudo_count*np.max(fate_map[sp_idx,:])
                valid_idx=fate_map[sp_idx,:].sum(1)>sum_fate_prob_thresh # default 0.5

                diff=potential_vector_temp[:,0]#-potential_vector_temp[:,1]
                tot=potential_vector_temp.sum(1)

                #valid_idx=tot>sum_fate_prob_thresh # default 0.5
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
                    Clb=fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.bwr), ax=ax,label=color_bar_label)
                    Clb.ax.set_title(f'{mega_cluster_list[0]}')

                plt.tight_layout()
                fig.savefig(f'{figure_path}/{data_des}_fate_bias_BW{map_backward}{figure_index}.{settings.file_format_figs}')



                # #adata.uns['fate_bias']=vector_array
                # vector_array_fullSpace=np.zeros(len(x_emb))+0.5
                # vector_array_fullSpace[cell_id_t1_sp[valid_idx]]=vector_array
                # adata.uns['fate_bias']=[vector_array,vector_array_fullSpace]

                ## save data to adata
                adata.uns['fate_bias']=pd.DataFrame({'cell_id':cell_id_t1_sp[valid_idx][new_idx],'fate_bias':vector_array[new_idx]})

                if show_histogram:
                    xxx=vector_array
                    fig=plt.figure(figsize=(fig_width,fig_height));ax=plt.subplot(1,1,1)
                    ax.hist(xxx,50,color='#2ca02c',density=True)
                    ax.set_xlim([0,1])
                    ax.set_xlabel(color_bar_label)
                    ax.set_ylabel('Density')
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.set_title(f'Average: {int(np.mean(xxx)*100)/100}')
                    plt.tight_layout()
                    fig.savefig(f'{figure_path}/{data_des}_fate_bias_BW{map_backward}_histogram{figure_index}.{settings.file_format_figs}')



def fate_coupling_from_Tmap(adata,selected_fates=None,used_Tmap='transition_map',
    selected_times=None,fate_map_method='sum',color_bar=True,
    method='SW',rename_fates=None,plot_heatmap=True):
    """
    Plot fate coupling determined by the transition map.

    We use the fate map :math:`P_i(\mathcal{C}_l)` towards a set of 
    fate clusters :math:`\{\mathcal{C}_l, l=0,1,2...\}` to compute the
    fate coupling :math:`Y_{ll'}`.

    * If method='SW': we first obtain :math:`Y_{ll'}=\sum_i P_i(\mathcal{C}_l)P_i(\mathcal{C}_{l'})`.
      Then, we normalize the the coupling: :math:`Y_{ll'}\leftarrow Y_{ll'}/\sqrt{Y_{ll}Y_{l'l'}}`.

    * If method='Weinreb', we calculate the normalized 
      covariance as in :func:`~cospar.hf.get_normalized_covariance`

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_fates: `list`, optional (default: all fates)
        List of cluster ids consistent with adata.obs['state_info']. 
        It allows a nested list, where we merge clusters within 
        each sub-list into a mega-fate cluster.
    used_Tmap: `str`, optional (default: 'transition_map')
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map',...}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    selected_times: `list`, optional (default: all)
        A list of time points to further restrict the cell states to plot. 
        The default choice is not to constrain the cell states to show. 
    fate_map_method: `str`, optional (default: 'sum')
        Method to obtain the fate probability map :math:`P_i(\mathcal{C})` towards a set 
        of states annotated with fate :math:`\mathcal{C}`. Available options: 
        {'sum', 'norm-sum'}. See :func:`.fate_map`.
    color_bar: `bool`, optional (default: True)
        Plot the color bar.
    method: `str`, optional (default: 'SW')
        Method to normalize the coupling matrix: {'SW', 'Weinreb'}.
    rename_fates: `list`, optional (default: [])
        Provide new names in substitution of names in selected_fates.
        For this to be effective, the new name list needs to have names 
        in exact correspondence to those in the old list. 
    plot_heatmap: `bool`, optional (default: True)
        If true, plot the heatmap for fate coupling.

    Returns
    -------
    X_coupling: `np.array`
        A inferred coupling matrix between selected fate clusters.
    """

    hf.check_available_map(adata)
    fig_width=settings.fig_width; fig_height=settings.fig_height; point_size=settings.fig_point_size
    if color_bar: fig_width=fig_width+0.5
    #set_up_plotting()
    
    map_backward=True
    
    if used_Tmap not in adata.uns['available_map']:
        logg.error(f"used_Tmap should be among {adata.uns['available_map']}")
        return None
    else:        
        state_annote=adata.obs['state_info']
        if map_backward:
            cell_id_t1=adata.uns['Tmap_cell_id_t1']
            cell_id_t2=adata.uns['Tmap_cell_id_t2']

        else:
            cell_id_t2=adata.uns['Tmap_cell_id_t1']
            cell_id_t1=adata.uns['Tmap_cell_id_t2']


        time_info=np.array(adata.obs['time_info'])
        sp_idx=hf.selecting_cells_by_time_points(time_info[cell_id_t1],selected_times)


        x_emb=adata.obsm['X_emb'][:,0]
        y_emb=adata.obsm['X_emb'][:,1]
        data_des=adata.uns['data_des'][-1]
        data_des=f'{data_des}_Tmap_fate_coupling'
        figure_path=settings.figure_path



        fate_map,mega_cluster_list,relative_bias,expected_prob,valid_fate_list,sel_index_list,fate_entropy=hf.compute_fate_probability_map(adata,
            selected_fates=selected_fates,used_Tmap=used_Tmap,map_backward=map_backward,method=fate_map_method)

        if (len(mega_cluster_list)==0) or (np.sum(sp_idx)==0):
            logg.error("No cells selected. Computation aborted!")
            return None
        else:

            if rename_fates is None: 
                rename_fates=mega_cluster_list

            if len(rename_fates)!=len(mega_cluster_list):
                logg.warn('rename_fates does not have the same length as selected_fates, thus not used.') 
                rename_fates=mega_cluster_list

            X_coupling = hf.get_normalized_covariance(fate_map[sp_idx],method=method)
            if plot_heatmap:
                heatmap(figure_path, X_coupling, rename_fates,color_bar_label='Fate coupling',color_bar=color_bar,data_des=data_des)

            return X_coupling

####################

## DDE analysis

####################



def differential_genes(adata,group_A_idx=None,group_B_idx=None,plot_groups=True,FDR_cutoff=0.05,plot_gene_N=3,savefig=False,sort_by='ratio'):
    """
    Perform differential gene expression analysis and plot top DGE genes.

    We use Wilcoxon rank-sum test to calculate P values, followed by
    Benjamini-Hochberg correction. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Need to contain gene expression matrix.
    group_A_idx: `np.array`, optional (default: None)
        A boolean array of the size adata.shape[0] for defining population A.
        If not specified, we set it to be adata.obs['cell_group_A']. 
    group_B_idx: `np.array`, optional (default: None)
        A boolean array of the size adata.shape[0] for defining population B.
        If not specified, we set it to be adata.obs['cell_group_A'].         
    plot_groups: `bool`, optional (default: True)
        If true, plot the selected ancestor states for A, B
    plot_gene_N: `int`, optional (default: 5)
        Number of top DGE genes to plot
    savefig: `bool`, optional (default: False)
        Save all plots.
    FDR_cutoff: `float`, optional (default: 0.05)
        Cut off for the corrected Pvalue of each gene. Only genes below this
        cutoff will be shown.
    sort_by: `float`, optional (default: 'ratio')
        The key to sort the differentially expressed genes. The key can be: 'ratio' or 'Qvalue'.

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

    if sort_by not in ['ratio','Qvalue']:
        logg.error(f"sort_by must be among {['ratio','Qvalue']}")
        return diff_gene_A, diff_gene_B

    if (group_A_idx is not None): 
        group_A_idx=np.array(group_A_idx)
        if len(group_A_idx)!=adata.shape[0]:
            logg.error('group_A_idx should be a boolean array of the size adata.shape[0].')
            return diff_gene_A, diff_gene_B

    if (group_B_idx is not None): 
        group_B_idx=np.array(group_B_idx)
        if len(group_B_idx)!=adata.shape[0]:
            logg.error('group_B_idx should be a boolean array of the size adata.shape[0].')
            return diff_gene_A, diff_gene_B

    if group_A_idx is None:
        if ('cell_group_A' not in adata.obs.keys()): 
            logg.error("Cell population A not defined yet. Please define it directly at group_A_idx or adata.obs['cell_group_A']")
            return diff_gene_A, diff_gene_B
        else:
            group_A_idx=adata.obs['cell_group_A']

    if group_B_idx is None:
        if ('cell_group_B' not in adata.obs.keys()): 
            logg.error("Cell population B not defined yet. Please define it directly at group_B_idx or adata.obs['cell_group_B']")
            return diff_gene_A, diff_gene_B
        else:
            group_B_idx=adata.obs['cell_group_B']


    #set_up_plotting()
    if (np.sum(group_A_idx)==0) or (np.sum(group_B_idx)==0):
        logg.error("Group A or B has zero selected cell states.")

    else:

        dge=hf.get_dge_SW(adata,group_B_idx,group_A_idx)

        dge=dge.sort_values(by=sort_by,ascending=True)
        diff_gene_A_0=dge
        diff_gene_A=diff_gene_A_0[(dge['Qvalue']<FDR_cutoff) & (dge['ratio']<0)]
        diff_gene_A=diff_gene_A.reset_index()

        dge=dge.sort_values(by=sort_by,ascending=False)
        diff_gene_B_0=dge
        diff_gene_B=diff_gene_B_0[(dge['Qvalue']<FDR_cutoff) & (dge['ratio']>0)]
        diff_gene_B=diff_gene_B.reset_index()

        x_emb=adata.obsm['X_emb'][:,0]
        y_emb=adata.obsm['X_emb'][:,1]
        figure_path=settings.figure_path
        
        if plot_groups:
            fig,nrow,ncol = start_subplot_figure(2, row_height=4, n_columns=2, fig_width=8)
            ax = plt.subplot(nrow, ncol, 1)
            customized_embedding(x_emb,y_emb,group_A_idx,ax=ax,point_size=point_size)
            ax.set_title(f'Group A')
            ax.axis('off')
            ax = plt.subplot(nrow, ncol, 2)
            customized_embedding(x_emb,y_emb,group_B_idx,ax=ax,point_size=point_size)
            ax.set_title(f'Group B')
            ax.axis('off')
            
            plt.tight_layout()
            if savefig:
                fig.savefig(f'{figure_path}/dge_analysis_groups.{settings.file_format_figs}')
            
        #logg.error("Plot differentially-expressed genes for group A")
        if plot_gene_N>0:

            #logg.error(f"Plot the top {plot_gene_N} genes that are differentially expressed on group A")
            if len(diff_gene_A['gene'])<plot_gene_N:
                plot_gene_N_A=len(diff_gene_A['gene'])
            else:
                plot_gene_N_A=plot_gene_N

            fig,nrow,ncol = start_subplot_figure(plot_gene_N_A, row_height=2.5, n_columns=5, fig_width=16)
            for j in range(plot_gene_N_A):
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
            if len(diff_gene_B['gene'])<plot_gene_N:
                plot_gene_N_B=len(diff_gene_B['gene'])
            else:
                plot_gene_N_B=plot_gene_N

            fig,nrow,ncol = start_subplot_figure(plot_gene_N_B, row_height=2.5, n_columns=5, fig_width=16)
            for j in range(plot_gene_N_B):
                ax = plt.subplot(nrow, ncol, j+1)
                gene_name=np.array(diff_gene_B['gene'])[j]
                customized_embedding(x_emb,y_emb,adata.obs_vector(gene_name),ax=ax,point_size=point_size)
                ax.set_title(f'{gene_name}')
                ax.axis('off')
            plt.tight_layout()
            if savefig:
                fig.savefig(f'{figure_path}/dge_analysis_groups_B_genes.{settings.file_format_figs}')

        
    return diff_gene_A,diff_gene_B




def differential_genes_for_given_fates(adata,selected_fates=None,selected_times=None,
    plot_groups=True,plot_gene_N=3,FDR_cutoff=0.05,savefig=False):
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
    selected_times: `list`, optional (default: all)
        A list of time points to further restrict the cell states to plot. 
        If map_backward=True, plot initial states that are among these time points;
        else, plot later states that are among these time points.
    plot_groups: `bool`, optional (default: True)
        If true, plot the selected ancestor states for A, B
    plot_gene_N: `int`, optional (default: 5)
        Number of top DGE genes to plot.
    savefig: `bool`, optional (default: False)
        Save all plots.
    FDR_cutoff: `float`, optional (default: 0.05)
        Cut off for the corrected Pvalue of each gene. Only genes below this
        cutoff will be shown.

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
    sp_idx=hf.selecting_cells_by_time_points(time_info,selected_times)
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

            diff_gene_A,diff_gene_B=differential_genes(adata,plot_groups=plot_groups,plot_gene_N=plot_gene_N,FDR_cutoff=FDR_cutoff,savefig=savefig)
                
    return diff_gene_A,diff_gene_B


######################

## Dynamic trajectory

######################

def dynamic_trajectory_from_fate_bias(adata,selected_fates=None,used_Tmap='transition_map',
    map_backward=True,method='norm-sum',selected_times=None,
    bias_threshold_A=0.5,bias_threshold_B=0.5,sum_fate_prob_thresh=0,pseudo_count=0,avoid_target_states=False,mask=None,
    plot_ancestor=True,savefig=False,plot_target_state=True,target_transparency=0.2):
    """
    Identify trajectories towards/from two given clusters.

    Given fate bias :math:`Q_i` for a state :math:`i` as defined in :func:`.fate_bias`, 
    the selected ancestor population satisfies:

       * :math:`P_i(\mathcal{A})+P_i(\mathcal{B})` > sum_fate_prob_thresh; 

       * Ancestor population for fate :math:`\mathcal{A}` satisfies :math:`Q_i` > bias_threshold_A

       * Ancestor population for fate :math:`\mathcal{B}` satisfies :math:`Q_i` < bias_threshold_B

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_fates: `list`
        List of cluster ids consistent with adata.obs['state_info']. 
        It allows a nested structure. 
    used_Tmap: `str`
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map',...}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    map_backward: `bool`, optional (default: True)
        If `map_backward=True`, show fate properties of initial cell states :math:`i`; 
        otherwise, show progenitor properties of later cell states :math:`j`.
        This is used for building the fate map :math:`P_i(\mathcal{C})`. See :func:`.fate_map`.
    method: `str`, optional (default: 'norm-sum')
        Method to obtain the fate probability map :math:`P_i(\mathcal{C})` towards a set 
        of states annotated with fate :math:`\mathcal{C}`. Available options: 
        {'sum', 'norm-sum'}. See :func:`.fate_map`.
    selected_times: `list`, optional (default: all)
        A list of time points to further restrict the cell states to plot. 
        The default choice is not to constrain the cell states to show. 
    bias_threshold_A: `float`, optional (default: 0), range: (0,1)
        The threshold for selecting ancestor population for fate A. 
    bias_threshold_B: `float`, optional (default: 0), range: (0,1)
        The threshold for selecting ancestor population for fate B.
    sum_fate_prob_thresh: `float`, optional (default: 0), range: (0,1)
        Minimum cumulative probability towards joint cluster (A,B) 
        to qualify for ancestor selection.
    pseudo_count: `float`, optional (default: 0)
        Pseudo count to compute the fate bias. The bias = (Pa+c0)/(Pa+Pb+2*c0), 
        where c0=pseudo_count*(maximum fate probability) is a rescaled pseudo count. 
    savefig: `bool`, optional (default: False)
        Save all plots.
    avoid_target_states: `bool`, optional (default: False)
        If true, avoid selecting cells at the target cluster (A, or B) as 
        ancestor population.
    mask: `np.array`, optional (default: None)
        A boolean array for available cell states. It should has the length as adata.shape[0].
        Especially useful to constrain the states to show fate bias.
    plot_ancestor: `bool`, optional (default: True)
        If true, plot the progenitor states that have been selected based on cell fate bias. 
    plot_target_state: `bool`, optional (default: True)
        If true, highlight the target clusters as defined in selected_fates.
    target_transparency: `float`, optional (default: 0.2)
        Transparency parameter for plotting. 

    Returns
    -------
    adata.obs['cell_group_A']: `np.array` of `bool`
        A boolean array for selected progenitor states towards fate :math:`\mathcal{A}`.
    adata.obs['cell_group_B']: `np.array` of `bool`
        A boolean array for selected progenitor states towards fate :math:`\mathcal{B}`.
    adata.obs[f'traj_{fate_name}']: `np.array`
        A binary array for indicating states belonging to a trajectory.
    """

    diff_gene_A=[]
    diff_gene_B=[]
    hf.check_available_map(adata)
    fig_width=settings.fig_width; fig_height=settings.fig_height; point_size=settings.fig_point_size

    if used_Tmap not in adata.uns['available_map']:
        logg.error(f"used_Tmap should be among {adata.uns['available_map']}")


    else:
        if map_backward:
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
            sp_idx=hf.selecting_cells_by_time_points(time_info[cell_id_t1],selected_times)
            if mask is not None:
                if len(mask)==adata.shape[0]:
                    mask=mask.astype(bool)
                    sp_idx=sp_idx & (mask[cell_id_t1])
                else:
                    logg.error('mask length does not match adata.shape[0]. Ignored mask.')

            #if 'fate_map' not in adata.uns.keys():
            fate_map,mega_cluster_list,relative_bias,expected_prob,valid_fate_list,sel_index_list,fate_entropy=hf.compute_fate_probability_map(adata,
                selected_fates=selected_fates,used_Tmap=used_Tmap,map_backward=map_backward,method=method)

            if (len(mega_cluster_list)!=2) or (np.sum(sp_idx)==0):
                logg.error(f"Do not have valid fates or time points. Computation aborted!")
            else:
                if pseudo_count==0:
                    pseudo_count=10**(-10)

                potential_vector_temp=fate_map+pseudo_count*np.max(fate_map)
                valid_idx=fate_map.sum(1)>sum_fate_prob_thresh # default 0.5
                    
                diff=potential_vector_temp[:,0]#-potential_vector_temp[:,1]
                tot=potential_vector_temp.sum(1)

                #valid_idx=tot>sum_fate_prob_thresh # default 0
                valid_id=np.nonzero(valid_idx)[0]
                vector_array=np.zeros(np.sum(valid_idx))
                vector_array=diff[valid_idx]/(tot[valid_idx])

                idx_for_group_A=np.zeros(len(tot),dtype=bool)
                idx_for_group_B=np.zeros(len(tot),dtype=bool)
                idx_for_group_A[valid_id]=vector_array>(bias_threshold_A)
                idx_for_group_B[valid_id]=vector_array<(bias_threshold_B)

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
        
                # store the trajectory
                temp_list=[group_A_idx_full,group_B_idx_full]
                for j, fate_name in enumerate(mega_cluster_list):
                    selected_idx=sel_index_list[j]
                    combined_prob_temp=temp_list[j].astype(int)+selected_idx.astype(int)
                    adata.obs[f'traj_{fate_name}']=combined_prob_temp



def dynamic_trajectory_via_iterative_mapping(adata,selected_fate=None,used_Tmap='transition_map',
    map_backward=True,map_threshold=0.1,plot_separately=False,
    apply_time_constaint=False,color_bar=True):
    """
    Infer trajectory towards/from a cluster

    If map_backward=True, infer the trajectory backward in time. 
    Using inferred transition map, the inference is applied recursively. It
    starts with the cell states for the selected fate and uses the selected 
    map to infer the immediate ancestor states. Then, using these putative 
    ancestor states as the secondary input, it finds their immediate ancestors 
    again. This goes on until all time points are exhausted.

    It only works for transition map from multi-time clones.
    
    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_fate: `str`, or `list`
        Targeted cluster of the trajectory, as consistent with adata.obs['state_info']
        When it is a list, the listed clusters are combined into a single fate cluster. 
    used_Tmap: `str`
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map',...}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    map_backward: `bool`, optional (default: True)
        If `map_backward=True`, show fate properties of initial cell states :math:`i`; 
        otherwise, show progenitor properties of later cell states :math:`j`.
        This is used for building the fate map :math:`P_i(\mathcal{C})`. See :func:`.fate_map`.
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
    adata.obs[f'traj_{fate_name}']: `np.array`
        The probability of each state to belong to a trajectory.
    """        

    # We always use the probabilistic map, which is more realiable. Otherwise, the result is very sensitive to thresholding
    #transition_map=adata.uns['transition_map']
    #demultiplexed_map=adata.uns['demultiplexed_map']
    hf.check_available_map(adata)
    fig_width=settings.fig_width; fig_height=settings.fig_height; point_size=settings.fig_point_size

    if used_Tmap not in adata.uns['available_map']:
        logg.error(f"used_Tmap should be among {adata.uns['available_map']}")

    else:

        state_annote_0=np.array(adata.obs['state_info'])
        if map_backward:
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
        used_map_0=adata.uns[used_Tmap]
        resol=10**(-10)
        used_map_0=hf.sparse_rowwise_multiply(used_map_0,1/(resol+np.sum(used_map_0,1).A.flatten()))

        if map_backward:
            used_map=used_map_0
        else:
            used_map=used_map_0.T

        if type(selected_fate)==str:
            selected_fate=[selected_fate]
        #fate_name,selected_idx=flexible_selecting_cells(adata,selected_fate)
        mega_cluster_list,valid_fate_list,fate_array_flat,sel_index_list=hf.analyze_selected_fates(selected_fate,adata.obs['state_info'])
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
                        customized_embedding(x_emb,y_emb,prob_array[k],ax=ax1,point_size=point_size,title=f"Iteration: {k+1}")

                fig.savefig(f'{figure_path}/{data_des}_predicting_fate_trajectory_separate_BW{map_backward}.{settings.file_format_figs}')  
            else:

                col=2; row=1
                fig = plt.figure(figsize=(fig_width * col, fig_height * row))
                ax0=plt.subplot(row,col,1)
                customized_embedding(x_emb,y_emb,prob_0r_0,ax=ax0,point_size=point_size,title=f"Initial: t={sort_time_info[0]}");

                ax1=plt.subplot(row,col,2)
                customized_embedding(x_emb,y_emb,cumu_prob,ax=ax1,point_size=point_size,title=f"All time")

                fig.savefig(f'{figure_path}/{data_des}_predicting_fate_trajectory_allTime_BW{map_backward}.{settings.file_format_figs}')

            if color_bar:
                fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Reds), ax=ax1,label='Fate Probability')

            combined_prob=cumu_prob+prob_0r
            adata.obs[f'traj_{fate_name}']=combined_prob



def gene_expression_dynamics(adata,selected_fate,gene_name_list,traj_threshold=0.1,
    invert_PseudoTime=False,mask=None,
    compute_new=True,gene_exp_percentile=99,n_neighbors=8,
    plot_raw_data=False,stat_smooth_method='loess'):
    """
    Plot gene trend along the inferred dynamic trajectory.

    We assume that the dynamic trajecotry at given specification is already
    available at adata.obs[f'traj_{fate_name}'], which can be created via
    :func:`.dynamic_trajectory_via_iterative_mapping` or
    :func:`.dynamic_trajectory_from_fate_bias`.

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
    invert_PseudoTime: `bool`, optional (default: False)
        If true, invert the pseudotime: 1-pseuotime. This is useful when the direction
        of pseudo time does not agree with intuition.
    mask: `np.array`, optional (default: None)
        A boolean array for further selecting cell states.  
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
    """
    
    if mask==None:
        final_mask=np.ones(adata.shape[0]).astype(bool)
    else:
        if (mask.shape[0]==adata.shape[0]):
            final_mask=mask
        else:
            logg.error("mask must be a boolean array with the same size as adata.shape[0].")
            return None

    hf.check_available_map(adata)
    fig_width=settings.fig_width; fig_height=settings.fig_height; point_size=settings.fig_point_size

    if len(adata.uns['available_map'])==0:
        logg.error(f"There is no transition map available yet")

    else:

        if type(selected_fate)==str:
            selected_fate=[selected_fate]

        mega_cluster_list,valid_fate_list,fate_array_flat,sel_index_list=hf.analyze_selected_fates(selected_fate,adata.obs['state_info'])
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
            file_name=f'{data_path}/{data_des}_fate_trajectory_pseudoTime_{fate_name}.npy'



            traj_name=f'traj_{fate_name}'
            if traj_name not in adata.obs.keys():
                logg.error(f"The target fate trajectory for {fate_name} have not been inferred yet.\n" 
                    "Please infer the trajectory with first with cs.pl.dynamic_trajectory_from_fate_bias, \n"
                    "or cs.pl.dynamic_trajectory_via_iterative_mapping.")
                
            else:
                prob_0=np.array(adata.obs[traj_name])
                
                sel_cell_idx=(prob_0>traj_threshold*np.max(prob_0)) & final_mask

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
                fig.savefig(f'{figure_path}/{data_des}_fate_trajectory_pseudoTime_{fate_name}.{settings.file_format_figs}')

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

                gplot.save(f'{figure_path}/{data_des}_fate_trajectory_pseutoTime_gene_expression_{fate_name}.{settings.file_format_figs}',width=fig_width, height=fig_height,verbose=False)
                gplot.draw()



##################

# Clone related #

##################


def clones_on_manifold(adata,selected_clone_list=[0],clone_point_size=12,
    color_list=['red','blue','purple','green','cyan','black'],selected_times=None,title=True):
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
    selected_times: `list`, optional (default: all)
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
    sp_idx=hf.selecting_cells_by_time_points(time_info,selected_times)
    selected_times=np.sort(list(set(time_info[sp_idx])))

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
            for j, xx in enumerate(selected_times):
                idx_t0=time_info==selected_times[j]
                idx_t=idx_t0 | idx_t
            
            customized_embedding(x_emb[idx_t],y_emb[idx_t],np.zeros(len(y_emb[idx_t])),ax=ax,point_size=point_size)
            for j, xx in enumerate(selected_times):
                idx_t=time_info==selected_times[j]
                idx_clone=clone_annot[:,my_id].A.flatten()>0
                idx=idx_t & idx_clone
                ax.plot(x_emb[idx],y_emb[idx],'.',color=color_list[j%len(color_list)],markersize=clone_point_size*point_size,markeredgecolor='white',markeredgewidth=point_size)

                if title:
                    ax.set_title(f'ID: {my_id}')

            fig.savefig(f'{figure_path}/{data_des}_different_clones_{my_id}.{settings.file_format_figs}')



# this is based on Fisher-Exact test, much faster. The Pvalue is corrected. No ranked profile for randomized clones.
def clonal_fate_bias(adata,selected_fate='',show_histogram=True,FDR=0.05,alternative='two-sided'):
    """
    Plot clonal fate bias towards a cluster.

    The clonal fate bias is -log(Q-value). We calculated a P-value that 
    that a clone is enriched (or depleted) in a fate, using Fisher-Exact 
    test (accounting for clone size). The P-value is then corrected to 
    give a Q-value by Benjamini-Hochberg procedure. The alternative 
    hypothesis options are: {'two-sided', 'greater', 'less'}. 
    The default is 'two-sided'.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    selected_fate: `str`
        The targeted fate cluster, from adata.obs['state_info'].
    show_histogram: `bool`, optional (default: True)
        If true, show the distribution of inferred fate probability.
    FDR: `float`, optional (default: 0.05)
        False-discovery rate after the Benjamini-Hochberg correction.
    alternative: `str`, optional (default: 'two-sided')
        Defines the alternative hypothesis. The following options are 
        available (default is ‘two-sided’): ‘two-sided’; 
        ‘less’: one-sided; ‘greater’: one-sided

    Returns
    -------
    result: `pd.DataFrame` 
    """

    if alternative not in ['two-sided','less','greater']:
        logg.warn("alternative not in ['two-sided','less','greater']. Use 'two-sided' instead.")
        alternative='two-sided'

    fig_width=settings.fig_width; fig_height=settings.fig_height; point_size=settings.fig_point_size
    state_info=adata.obs['state_info']
    data_des=adata.uns['data_des'][-1]
    X_clone=adata.obsm['X_clone']
    data_path=settings.data_path
    data_des=adata.uns['data_des'][-1]
    figure_path=settings.figure_path
    state_list=list(set(state_info))


    clone_N=X_clone.shape[1]
    cell_N=X_clone.shape[0]

    if type(selected_fate)==str:
        selected_fate=[selected_fate]

    mega_cluster_list,valid_fate_list,fate_array_flat,sel_index_list=hf.analyze_selected_fates(selected_fate,state_info)
    if len(mega_cluster_list)==0:
        logg.error("No cells selected. Computation aborted!")
        return None, None 
    else:
        fate_name=mega_cluster_list[0]
        target_idx=sel_index_list[0]


        ## target clone
        target_ratio_array=np.zeros(clone_N)
        P_value=np.zeros(clone_N)

        #null_ratio_array=np.zeros((clone_N,N_resampling))
        #P_value_up=np.zeros(clone_N)
        #P_value_down=np.zeros(clone_N)
        
        #P_value_rsp=np.zeros((clone_N,N_resampling))

        for m in range(clone_N):
            if m%50==0:
                logg.info(f"Current clone id: {m}")
            target_cell_idx=(X_clone[:,m].sum(1).A>0).flatten()
            target_clone_size=np.sum(target_cell_idx) 


            if target_clone_size>0:
                target_ratio=np.sum(target_idx[target_cell_idx])/target_clone_size
                target_ratio_array[m]=target_ratio
                cell_N_in_target=np.sum(target_idx[target_cell_idx])
                #N_resampling=int(np.floor(cell_N/target_clone_size))

                remain_cell_idx=~target_cell_idx
                remain_cell_N_in_target=np.sum(target_idx[remain_cell_idx])
                oddsratio, pvalue = stats.fisher_exact([[cell_N_in_target, target_clone_size-cell_N_in_target], [remain_cell_N_in_target, cell_N-remain_cell_N_in_target]],alternative=alternative)
                P_value[m]=pvalue

        P_value = statsmodels.sandbox.stats.multicomp.multipletests(P_value, alpha=0.05, method='fdr_bh')[1]

        ####### Plotting
        clone_size_array=X_clone.sum(0).A.flatten()

        resol=10**(-20)
        sort_idx=np.argsort(P_value)
        P_value=P_value[sort_idx]+resol
        fate_bias=-np.log10(P_value)
        #idx=clone_size_array[sort_idx]>=clone_size_thresh
        FDR_threshold=-np.log10(FDR)


        fig=plt.figure(figsize=(fig_width,fig_height));ax=plt.subplot(1,1,1)
        ax.plot(np.arange(len(fate_bias)),fate_bias,'.',color='blue',markersize=5)#,markeredgecolor='black',markeredgewidth=0.2)
        #ax.plot(np.arange(len(fate_bias))[~idx],fate_bias[~idx],'.',color='blue',markersize=5,label=f'Size $<$ {int(clone_size_thresh)}')#,markeredgecolor='black',markeredgewidth=0.2)
        #ax.plot(np.arange(len(fate_bias))[idx],fate_bias[idx],'.',color='red',markersize=5,label=f'Size $\ge$ {int(clone_size_thresh)}')#,markeredgecolor='black',markeredgewidth=0.2)
        ax.plot(np.arange(len(fate_bias)),np.zeros(len(fate_bias))+FDR_threshold,'-.',color='grey',markersize=5,label=f'FDR={FDR}')#,markeredgecolor='black',markeredgewidth=0.2)
                
        #ax.plot(np.arange(len(fate_bias_rsp)),fate_bias_rsp,'.',color='grey',markersize=5,label='Randomized')#,markeredgecolor='black',markeredgewidth=0.2)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        #ax.set_xlabel('Clone rank')
        #plt.rc('text', usetex=True)
        #ax.set_ylabel('Fate bias ($-\\log_{10}P_{value}$)')
        ax.set_ylabel('Clonal fate bias')
        ax.legend()
        #ax.set_xlim([0,0.8])
        fig.tight_layout()
        fig.savefig(f'{figure_path}/{data_des}_clonal_fate_bias.{settings.file_format_figs}')
        #plt.rc('text', usetex=False)
        #plt.show()

        result=pd.DataFrame({'Clone ID':sort_idx,'Clone size':clone_size_array[sort_idx],'Q_value':P_value,'Fate bias':fate_bias})

        if show_histogram:
            target_fraction_array=(X_clone.T*target_idx)/clone_size_array
            fig=plt.figure(figsize=(fig_width,fig_height));ax=plt.subplot(1,1,1)
            ax.hist(target_fraction_array,color='#2ca02c',density=True)
            ax.set_xlim([0,1])
            ax.set_xlabel('Clonal fraction in selected fates')
            ax.set_ylabel('Density')
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_title(f'Average: {int(np.mean(target_fraction_array)*100)/100};   Expect: {int(np.mean(target_idx)*100)/100}')
            fig.savefig(f'{figure_path}/{data_des}_observed_clonal_fraction.{settings.file_format_figs}')

        return result


# this is based on direct simulation, providing the ranked profile for randomized clones
def clonal_fate_bias_v0(adata,selected_fate='',clone_size_thresh=3,
    N_resampling=1000,compute_new=True,show_histogram=True):
    """
    Plot clonal fate bias towards a cluster.

    This is just -log(P-value), where P-value is for the observation 
    cell fraction of a clone in the targeted cluster as compared to 
    randomized clones, where the randomized sampling produces clones 
    of the same size as the targeted clone. The computed results will 
    be saved at the directory settings.data_path.

    This function is based on direct simulation, which is time-consuming. 
    The time cost scales linearly with the number of resampling 
    and the number of clones. It provides a ranked profile
    for randomized clones.

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

    if type(selected_fate)==str:
        selected_fate=[selected_fate]
    
    mega_cluster_list,valid_fate_list,fate_array_flat,sel_index_list=hf.analyze_selected_fates(selected_fate,state_annote_new)
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

        return fate_bias,sort_idx,clone_size_array[sort_idx]



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


# version v1
def ordered_heatmap(figure_path, data_matrix, variable_names,int_seed=10,col_range=[0,99],
    data_des='',log_transform=True,color_map=plt.cm.Reds,vmin=np.nan,vmax=np.nan,fig_width=4,fig_height=6,
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
    col_range: `tuple`, optional (default: None)
        The default setting is to plot the actual value of the vector. 
        If col_range is set within [0,100], it will plot the percentile of the values,
        and the color_bar will show range [0,1]. This re-scaling is useful for 
        visualizing gene expression. 
    """

    o = hf.get_hierch_order(data_matrix)
    if log_transform:
        new_data=np.log(data_matrix[o,:]+1)/np.log(10)
    else:
        new_data=data_matrix[o,:]
        
    #mask_0=np.ones(new_data.shape)
    #col_data=new_data[np.triu(mask_0,k=1)>0].flatten()
    col_data=new_data.flatten()
    if np.isnan(vmin):
        if col_range is None:
            vmin=np.min(col_data)
        else:
            vmin = np.percentile(col_data, col_range[0])
        
    if np.isnan(vmax):
        if col_range is None:
            vmax=np.max(col_data)
        else:
            vmax = np.percentile(col_data, col_range[1])
            
            
    plt.figure(int_seed)
    plt.imshow(new_data, aspect='auto',cmap=color_map, vmin=vmin,vmax=vmax)

    if type(variable_names)==str:
        if variable_names=='':
            plt.xticks([])
    else:
        plt.xticks(np.arange(data_matrix.shape[1])+.4, variable_names, rotation=70, ha='right')
    
    plt.yticks([])
    if color_bar:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cbar=plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=color_map))
        
        if log_transform:
            cbar.set_label('Number of barcodes (log10)', rotation=270, labelpad=20)
        else:
            cbar.set_label('Number of barcodes', rotation=270, labelpad=20)
    plt.gcf().set_size_inches((fig_width,fig_height))
    plt.tight_layout()
    plt.savefig(figure_path+f'/{data_des}_data_matrix.{settings.file_format_figs}')




def barcode_heatmap(adata,selected_times=None,selected_fates=None,color_bar=True,rename_fates=None,log_transform=False,fig_width=4,fig_height=6):
    """
    Plot barcode heatmap among different fate clusters.

    We clonal measurement at selected time points and show the 
    corresponding heatmap among selected fate clusters. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    selected_times: `list`, optional (default: None)
        Time points to select the cell states.
    selected_fates: `list`, optional (default: all)
        List of fate clusters to use. If set to be [], use all.
    color_bar: `bool`, optional (default: True)
        Plot color bar. 
    rename_fates: `list`, optional (default: None)
        Provide new names in substitution of names in selected_fates.
        For this to be effective, the new name list needs to have names 
        in exact correspondence to those in the old list. 
    log_transform: `bool`, optional (default: False)
        If true, perform a log transform. This is needed when the data 
        matrix has entries varying by several order of magnitude. 
    fig_width: `float`, optional (default: 4)
        Figure width.
    fig_height: `float`, optional (default: 6)
        Figure height.
    """

    time_info=np.array(adata.obs['time_info'])
    if selected_times is not None:
        if type(selected_times) is not list:
            selected_times=[selected_times] 
    sp_idx=hf.selecting_cells_by_time_points(time_info,selected_times)
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

            if rename_fates is None:
                rename_fates=mega_cluster_list

            if len(rename_fates)!=len(mega_cluster_list):
                logg.warn('rename_fates does not have the same length as selected_fates, thus not used.') 
                rename_fates=mega_cluster_list

            ordered_heatmap(figure_path, coarse_clone_annot.T, rename_fates,data_des=data_des,log_transform=log_transform,fig_width=fig_width,fig_height=fig_height)




def fate_coupling_from_clones(adata,selected_times=None,selected_fates=None,color_bar=True,rename_fates=None,plot_heatmap=True,method='Weinreb'):
    """
    Plot fate coupling based on clonal information.

    We select one time point with clonal measurement and show the normalized 
    clonal covariance among these fates. See :func:`~cospar.hf.get_normalized_covariance`.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    selected_times: `list`, optional (default: None)
        Time points to select the cell states.
    selected_fates: `list`, optional (default: all)
        List of fate clusters to use. If set to be None, use all.
    color_bar: `bool`, optional (default: True)
        Plot color bar. 
    rename_fates: `list`, optional (default: None)
        Provide new names in substitution of names in selected_fates.
        For this to be effective, the new name list needs to have names 
        in exact correspondence to those in the old list. 
    plot_heatmap: `bool`, optional (default: True)
        Plot the inferred fate coupling in heatmap.
    method: `str`, optional (default: 'SW')
        Method to normalize the coupling matrix: {'SW', 'Weinreb'}.

    Returns
    -------
    X_coupling: `np.array`
        A inferred coupling matrix between selected fate clusters.
    """

    time_info=np.array(adata.obs['time_info'])
    if selected_times is not None:
        if type(selected_times) is not list:
            selected_times=[selected_times]

    sp_idx=hf.selecting_cells_by_time_points(time_info,selected_times)

    clone_annot=adata[sp_idx].obsm['X_clone']
    state_annote=adata[sp_idx].obs['state_info']

    if (np.sum(sp_idx)==0):
        logg.error("No cells selected. Computation aborted!")
        return None
    else:
        mega_cluster_list,valid_fate_list,fate_array_flat,sel_index_list=hf.analyze_selected_fates(selected_fates,state_annote)
        if (len(mega_cluster_list)==0):
            logg.error("No cells selected. Computation aborted!")
            return None
        else:
            x_emb=adata.obsm['X_emb'][:,0]
            y_emb=adata.obsm['X_emb'][:,1]
            data_des=adata.uns['data_des'][-1]
            data_des=f'{data_des}_clonal_fate_coupling'
            figure_path=settings.figure_path

            coarse_clone_annot=np.zeros((len(mega_cluster_list),clone_annot.shape[1]))
            for j, idx in enumerate(sel_index_list):
                coarse_clone_annot[j,:]=clone_annot[idx].sum(0)

            if rename_fates is None:
                rename_fates=mega_cluster_list

            if len(rename_fates)!=len(mega_cluster_list):
                logg.warn('rename_fates does not have the same length as selected_fates, thus not used.') 
                rename_fates=mega_cluster_list

            X_coupling = hf.get_normalized_covariance(coarse_clone_annot.T,method=method)

            if plot_heatmap:
                heatmap(figure_path, X_coupling, rename_fates,color_bar_label='Coupling',color_bar=color_bar,data_des=data_des)

            return X_coupling


#################

## Fate hierarchy

#################


def fate_hierarchy_from_Tmap(adata,selected_fates=None,used_Tmap='transition_map',selected_times=None,
      method='SW',rename_fates=None,
             plot_history=True):
    """
    Construct the fate hierarchy from the transition map.

    Based on the fate coupling matrix from :func:`.fate_coupling_from_Tmap`,
    we use neighbor-joining to build the fate hierarchy iteratively. 
    This function is adapted from clinc package https://pypi.org/project/clinc/
    (Weinreb & Klein, PNAS, 2021).

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_fates: `list`
        List of cluster ids consistent with adata.obs['state_info']. 
        It allows a nested structure. If so, we merge clusters within 
        each sub-list into a mega-fate cluster.
    used_Tmap: `str`
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map',...}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    selected_times: `list`, optional (default: all)
        A list of time points to further restrict the cell states to plot. 
        The default choice is not to constrain the cell states to show. 
    method: `str`, optional (default: 'SW')
        Method to normalize the coupling matrix: {'SW', 'Weinreb'}.
    rename_fates: `list`, optional (default: None)
        Provide new names in substitution of names in selected_fates.
        For this to be effective, the new name list needs to have names 
        in exact correspondence to those in the old list. 
    plot_history: `bool`, optional (default: True)
        Plot the history of constructing the hierarchy.
    """
    
    hf.check_available_map(adata)
    if used_Tmap not in adata.uns['available_map']:
        logg.error(f"used_Tmap should be among {adata.uns['available_map']}")
        return None
    else:        
        state_annote=adata.obs['state_info']
        if selected_fates is None:
            selected_fates=list(set(state_annote))
        if (rename_fates is None): 
            rename_fates=selected_fates

        if (len(rename_fates)!=len(selected_fates)):
            logg.warn('rename_fates does not have the same length as selected_fates, thus not used.')
            rename_fates=selected_fates                   
                   
    parent_map, node_groups, history=build_hierarchy_from_Tmap(adata,selected_fates=selected_fates,used_Tmap=used_Tmap,
        selected_times=selected_times,method=method)
                   
    
    print_hierarchy(parent_map, rename_fates)
    if plot_history:
        plot_neighbor_joining(settings.figure_path, node_groups, rename_fates, 
                              history[0], history[1], history[2])


def fate_hierarchy_from_clones(adata,selected_times=None,selected_fates=None,rename_fates=None,
                               plot_history=True,method='SW'):
    """
    Construct the fate hierarchy from clonal data.

    Based on the fate coupling matrix from :func:`.fate_coupling_from_clones`,
    we use neighbor-joining to build the fate hierarchy iteratively. 
    This function is adapted from clinc package https://pypi.org/project/clinc/
    (Weinreb & Klein, PNAS, 2021).

    Parameters
    ----------
    selected_times: `str`, optional (default: all)
        Time point to select the cell states.
    selected_fates: `list`, optional (default: all)
        List of fate clusters to use. If set to be None, use all.
    color_bar: `bool`, optional (default: True)
        Plot color bar. 
    rename_fates: `list`, optional (default: None)
        Provide new names in substitution of names in selected_fates.
        For this to be effective, the new name list needs to have names 
        in exact correspondence to those in the old list. 
    plot_history: `bool`, optional (default: True)
        Plot the history of constructing the hierarchy.
    method: `str`, optional (default: 'SW')
        Method to normalize the coupling matrix: {'SW', 'Weinreb'}.
    """
    
    hf.check_available_map(adata)
  
    state_annote=adata.obs['state_info']
    if selected_fates is None:
        selected_fates=list(set(state_annote))
               
    if (rename_fates is None): 
        rename_fates=selected_fates

    if (len(rename_fates)!=len(selected_fates)):
       logg.warn('rename_fates does not have the same length as selected_fates, thus not used.') 

    parent_map, node_groups, history=build_hierarchy_from_clones(adata,selected_fates=selected_fates,
        selected_times=selected_times,method=method)
        
    print_hierarchy(parent_map, rename_fates)           
    if plot_history:
        
        plot_neighbor_joining(settings.figure_path, node_groups, rename_fates, 
                              history[0], history[1], history[2])


def build_hierarchy_from_clones(adata,selected_fates=None,selected_times=None,method='SW'):
#selected_fates=fate_array
#used_Tmap='transition_map'

    fate_N=len(selected_fates)
    X_history = []
    merged_pairs_history = []
    node_names_history = []
    node_groups = {i:[i] for i in range(fate_N)}

    parent_map = {}
    selected_fates_tmp=[]
    for xx in selected_fates:
        if type(xx) is not list:
            xx=[xx]
        selected_fates_tmp.append(xx)
    node_names = list(range(fate_N))
    next_node = fate_N

    while len(node_names) > 2: 
        fate_N_tmp=len(selected_fates_tmp)
        node_names_history.append(node_names)
        X=fate_coupling_from_clones(adata,selected_fates=selected_fates_tmp,
               plot_heatmap=False,selected_times=selected_times,method=method)

        X_history.append(np.array(X))
        floor = X.min() - 100
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if i >= j: X[i,j] = floor


        ii = np.argmax(X.max(1))
        jj = np.argmax(X.max(0))
        merged_pairs_history.append((ii,jj))
        node_groups[next_node] = node_groups[node_names[ii]]+node_groups[node_names[jj]]

        parent_map[node_names[ii]] = next_node
        parent_map[node_names[jj]] = next_node

        ix = np.min([ii,jj])
        node_names = [n for n in node_names if not n in np.array(node_names)[np.array([ii,jj])]]
        new_ix = np.array([i for i in range(fate_N_tmp) if not i in [ii,jj]])

        if len(new_ix)==0: break
        new_fate=selected_fates_tmp[ii]+selected_fates_tmp[jj]
        selected_fates_tmp_1=[selected_fates_tmp[new_ix[xx]] for xx in range(ix)]
        selected_fates_tmp_1.append(new_fate)
        for xx in range(ix,fate_N_tmp-2):
            selected_fates_tmp_1.append(selected_fates_tmp[new_ix[xx]])
        selected_fates_tmp=selected_fates_tmp_1
        node_names.insert(ix,next_node)
        next_node += 1


    for i in node_names:
        parent_map[i] = next_node

    return parent_map, node_groups, (X_history, merged_pairs_history, node_names_history)


def build_hierarchy_from_Tmap(adata,selected_fates=None,used_Tmap='transition_map',
    selected_times=None,method='SW'):


    fate_N=len(selected_fates)
    X_history = []
    merged_pairs_history = []
    node_names_history = []
    node_groups = {i:[i] for i in range(fate_N)}

    parent_map = {}
    selected_fates_tmp=[]
    for xx in selected_fates:
        if type(xx) is not list:
            xx=[xx]
        selected_fates_tmp.append(xx)
    node_names = list(range(fate_N))
    next_node = fate_N

    while len(node_names) > 2: 
        fate_N_tmp=len(selected_fates_tmp)
        node_names_history.append(node_names)
        X=fate_coupling_from_Tmap(adata,selected_fates=selected_fates_tmp,
               used_Tmap=used_Tmap,plot_heatmap=False,selected_times=selected_times,
                method=method)
        X_history.append(np.array(X))
        floor = 0
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if i >= j: X[i,j] = floor


        ii = np.argmax(X.max(1))
        jj = np.argmax(X.max(0))
        merged_pairs_history.append((ii,jj))
        node_groups[next_node] = node_groups[node_names[ii]]+node_groups[node_names[jj]]

        parent_map[node_names[ii]] = next_node
        parent_map[node_names[jj]] = next_node

        ix = np.min([ii,jj])
        node_names = [n for n in node_names if not n in np.array(node_names)[np.array([ii,jj])]]
        new_ix = np.array([i for i in range(fate_N_tmp) if not i in [ii,jj]])

        if len(new_ix)==0: break
        new_fate=selected_fates_tmp[ii]+selected_fates_tmp[jj]
        selected_fates_tmp_1=[selected_fates_tmp[new_ix[xx]] for xx in range(ix)]
        selected_fates_tmp_1.append(new_fate)
        for xx in range(ix,fate_N_tmp-2):
            selected_fates_tmp_1.append(selected_fates_tmp[new_ix[xx]])
        selected_fates_tmp=selected_fates_tmp_1
        node_names.insert(ix,next_node)
        next_node += 1


    for i in node_names:
        parent_map[i] = next_node

    return parent_map, node_groups, (X_history, merged_pairs_history, node_names_history)


def plot_neighbor_joining(output_directory, node_groups, celltype_names, X_history, merged_pairs_history, node_names_history):
    fig,axs = plt.subplots(1,len(X_history))
    for i,X in enumerate(X_history):
        vmax = 1.2*np.max(np.triu(X,k=1))
        axs[i].imshow(X,vmax=vmax)
        ii,jj = merged_pairs_history[i]
        axs[i].scatter([jj],[ii],s=100, marker='*', c='white')

        column_groups = [node_groups[n] for n in node_names_history[i]]
        column_labels = [' + '.join([celltype_names[n] for n in grp]) for grp in column_groups]
        axs[i].set_xticks(np.arange(X.shape[1])+.2)
        axs[i].set_xticklabels(column_labels, rotation=90, ha='right')
        axs[i].set_xlim([-.5,X.shape[1]-.5])
        axs[i].set_ylim([X.shape[1]-.5,-.5])
        axs[i].set_yticks(np.arange(X.shape[1])+.2)
        axs[i].set_yticklabels(['' for grp in column_groups], rotation=90, ha='right')
    fig.set_size_inches((16,4))
    plt.savefig(output_directory+'/neighbor_joint_heatmaps.pdf')

def print_hierarchy(parent_map, celltype_names):    
    child_map = {i:[] for i in set(list(parent_map.values())+list(parent_map.keys()))}
    for i,j in parent_map.items():
        child_map[j].append(i)

    leaf_names = {i:n for i,n in enumerate(celltype_names)}
    def get_newick(n):
        if n in leaf_names: return leaf_names[n]
        else: return '('+','.join([get_newick(nn) for nn in sorted(child_map[n])[::-1]])+')'
    tree_string = get_newick(np.max(list(child_map.keys())))+';'
    
    
    t = Tree(tree_string)
    print(t)


##############

## Gene expression heat map

#############


####### plot heat maps for genes
def heatmap_v1(figure_path, data_matrix, variable_names_x,variable_names_y,int_seed=10,
    data_des='',log_transform=False,color_map=plt.cm.Reds,vmin=None,vmax=None,fig_width=4,fig_height=6,
    color_bar=True,color_bar_label=''):
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
        cbar.set_label(color_bar_label, rotation=270, labelpad=20)
    plt.gcf().set_size_inches((fig_width,fig_height))
    plt.tight_layout()
    plt.savefig(figure_path+f'/{data_des}_data_matrix.{settings.file_format_figs}')



def gene_expression_heat_map(adata, selected_genes=None,selected_fates=None,rename_fates=None,color_bar=True,method='relative',fig_width=6,fig_height=3,horizontal='True',vmin=None,vmax=None):
    """
    Plot heatmap of gene expression within given clusters.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    selected_genes: `list`, optional (default: None)
        A list of selected genes.
    selected_fates: `list`, optional (default: all)
        List of cluster ids consistent with adata.obs['state_info']. 
        It allows a nested structure. If so, we merge clusters within 
        each sub-list into a mega-fate cluster.
    method: `str`, optional (default: 'relative')
        Method to normalize gene expression. Options: {'relative','zscore'}. 
        'relative': given coarse-grained gene expression 
        in given clusters, normalize the expression across clusters to be 1;
        'zscore': given coarse-grained gene expression in given clusters, compute its zscore.
    rename_fates: `list`, optional (default: None)
        Provide new names in substitution of names in selected_fates.
        For this to be effective, the new name list needs to have names 
        in exact correspondence to those in the old list. 
    color_bar: `bool`, optional (default: True)
        If true, show the color bar.
    fig_width: `int`, optional (default: 6)
        Figure width.
    fig_width: `int`, optional (default: 3)
        Figure height.
    horizontal: `bool`, optional (default: True)
        Figure orientation.
    vmin: `float`, optional (default: None)
        Minimum value to show.
    vmax: `float`, optional (default: None)
        Maximum value to show.

    Returns
    -------
    gene_expression_matrix: `np.array`
    """

    if method not in ['relative','zscore']:
        logg.warn("method not in ['relative','zscore']; set it to be 'relative'")
        method='relative'

    gene_list=selected_genes
    state_info=np.array(adata.obs['state_info'])
    mega_cluster_list,valid_fate_list,fate_array_flat,sel_index_list=hf.analyze_selected_fates(selected_fates,state_info)
    gene_full=np.array(adata.var_names)
    gene_list=np.array(gene_list)
    sel_idx=np.in1d(gene_full,gene_list)
    valid_sel_idx=np.in1d(gene_list,gene_full)
    
    if np.sum(valid_sel_idx)>0:
        cleaned_gene_list=gene_list[valid_sel_idx]
        if np.sum(valid_sel_idx)<len(gene_list):
            invalid_gene_list=gene_list[~valid_sel_idx]
            logg.info(f"These are invalid gene names: {invalid_gene_list}")
    else:
        logg.error("No valid genes selected.")
    gene_expression_matrix=np.zeros((len(mega_cluster_list),len(cleaned_gene_list)))
    
    X=adata.X
    resol=10**(-10)
    
    if method=='zscore':
        logg.hint("Using zscore (range: [-2,2], or [-1,1]")
        color_bar_label='Z-score'
    else:
        logg.hint("Using relative gene expression. Range [0,1]")
        color_bar_label='Relative expression'

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
        
        
    if (rename_fates is None) or (len(rename_fates) != len(mega_cluster_list)):
        rename_fates=mega_cluster_list
        
    if horizontal:
        heatmap_v1(settings.figure_path, gene_expression_matrix, cleaned_gene_list,rename_fates,int_seed=10,
        data_des='',log_transform=False,color_map=plt.cm.coolwarm,fig_width=fig_width,fig_height=fig_height,
        color_bar=color_bar,vmin=vmin,vmax=vmax,color_bar_label=color_bar_label)
    else:
        heatmap_v1(settings.figure_path, gene_expression_matrix.T,rename_fates, cleaned_gene_list,int_seed=10,
        data_des='',log_transform=False,color_map=plt.cm.coolwarm,fig_width=fig_height,fig_height=fig_width,
        color_bar=color_bar,vmin=vmin,vmax=vmax,color_bar_label=color_bar_label)
    
    return gene_expression_matrix