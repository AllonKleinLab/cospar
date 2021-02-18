Getting Started
---------------

Here, we explain the basics of using CoSpar. In the next tutorial, we will demonstrate its usage in a sub-sampled dataset of hematopoiesis. 

The input data for CoSpar are matrices for state and clonal information, and a vector for temporal annotation. We assume that the dataset has more than one time point. 



First, import CoSpar with::
    
    import cospar as cs

For better visualization you can change the matplotlib settings to our defaults with::
    
    cs.settings.set_figure_params()

If you want to adjust parameters for a particular plot, just pass the parameters into this function. 

Initialization
''''''''''''''
Given the gene expression matrix, clonal matrix, and other information, initialize the anndata object using::
    
    adata = cs.pp.initialize_adata_object(RNA_count_matrix,gene_names,time_info,
    X_clone=[],X_pca=[],X_emb=[],state_info=[],data_des='cospar')

The :class:`~anndata.AnnData` object adata stores the count matrix (``adata.X``), gene names (``adata.var_names``), and temporal annotation of cells (``adata.obs['time_info']``).  The clonal matrix ``X_clone`` is optional and will be stored at  ``adata.obsm['X_clone']``.  If not provided, you can still infer transition map based on state information alone, and proceed with the analysis. You can also provide the selected PCA matrix `X_pca`,  the embedding matrix ``X_emb``, and the state annotation ``state_info``, which will be stored at ``adata.obsm['X_pca']``, ``adata.obsm['X_emb']``, and ``adata.obs['state_info']``, respectively.  ``data_des`` is a string to label a dataset (``adata.uns['data_des']``), and should be unique for each dataset to avoid conflicts.  

Sometimes, you may have many useful annotations attached to your own preprocessed adata object, and you may not want to run cs.pp.initialize_adata_object, which erases these annotations. In this case, you can just attach the above information (``time_info`` etc.) into the corresponding field of your own adata object.

.. raw:: html

    <img src="http://falexwolf.de/img/scanpy/anndata.svg" style="width: 300px">

If you do not have a dataset yet, you can still play around using one of the built-in datasets, e.g.::
    
    adata = cs.datasets.hematopoiesis_subsampled()



Preprocessing & dimension reduction
'''''''''''''''''''''''''''''''''''
Assuming basic quality control (excluding cells with low read count etc.) have been done, we provide basic preprocessing (gene selection and normalization) and dimension reduction related analysis (PCA, UMAP embedding etc.)  at ``cs.pp.*``::
    
    cs.pp.get_highly_variable_genes(adata,**params)
    cs.pp.remove_cell_cycle_correlated_genes(adata,**params)
    cs.pp.get_X_pca(adata,**params)
    cs.pp.get_X_emb(adata,**params)
    cs.pp.get_state_info(adata,**params)

The first step ``get_highly_variable_genes`` also includes count matrix normalization. The second step, which is optional but recommended, removes cell cycle correlated genes among the selected highly variable genes. In ``get_X_pca``, we apply z-score transformation for each gene expression before computing the PCA. In ``get_X_emb``, we simply use the umap function from :mod:`~scanpy`. We also extract state information using leiden clustering implemented in :mod:`~scanpy`. These steps can also be performed by external packages like :mod:`~scanpy` directly, which is also built around the :class:`~anndata.AnnData` object.  


Simple clonal analysis
''''''''''''''''''''''
We provide a few plotting functions to help visually exploring the clonal data before any downstream analysis. You can visualize clones on state manifold directly:: 
    
    cs.pl.clones_on_manifold(adata,**params)

You can generate the barcode heatmap across given clusters to inspect clonal behavior::
    
    cs.pl.barcode_heatmap(adata,**params)

You can quantify the clonal coupling across different fate clusters::
    
    cs.pl.fate_coupling_from_clones(adata,**params)

Strong coupling implies the existence of bi-potent or multi-potent cell states at the time of barcoding. Finally, you can infer the fate bias of each clone towards a designated fate cluster::
    
    cs.pl.clonal_fate_bias(adata,**params)

A biased clone towards this cluster has a statistically significant cell fraction in this cluster.




Transition map inference
''''''''''''''''''''''''
The core of the software is the efficient and robust inference of a transition map by integrating state and clonal information. If the dataset has multiple clonal time points, you can run::
    
    adata=cs.tmap.infer_Tmap_from_multitime_clones(adata_orig,clonal_time_points,**params) 

It subsamples the input data according to selected time points (at least 2) with clonal information, computes the transition map (stored at ``adata.uns['transition_map']``), and returns the subsampled adata object. The inferred map allows transitions between neighboring time points. For example, if selected_clonal_time_points=['day1', 'day2', 'day3'], then it computes transitions for pairs ('day1', 'day2') and ('day2', 'day3'), but not ('day1', 'day3'). As a byproduct, it also returns a transition map that allows only intra-clone transitions (``adata.uns['intraclone_transition_map']``). The intra-clone transition map can also be computed from ``adata.uns['transition_map']``) at preferred parameters by running:: 
    
    cs.tmap.infer_intraclone_Tmap(adata,**params)

If the dataset has only one clonal time point, or you wish to infer the transition map just based on a single clonal time point, you can run::

    cs.tmap.infer_Tmap_from_one_time_clones(adata_orig,initial_time_points, clonal_time_point,initialize_method='OT',**params)

You need to define both ``initial_time_points`` and ``clonal_time_point``. We provide two methods for initializing the map using state information alone: 1) ``OT`` for using standard optimal transport approach; 2) ``HighVar`` for a customized approach that converts highly variable genes into pseudo multi-time clones and run ``cs.tmap.infer_Tmap_from_multitime_clones`` to construct the map. Depending on the choice,  the initialized map is stored at ``adata.uns['OT_transition_map']`` or  ``adata.uns['HighVar_transition_map']``. Afterwards CoSpar performs a joint optimization to infer both the initial clonal structure and also the transition map. The final product is stored at ``adata.uns['transition_map']``. This method returns a map for transitions from all given initial time points to the designated clonal time point.  For example, if initial_time_points=['day1', 'day2'], and clonal_time_point='day3', then the method computes transitions for pairs ('day1', 'day3') and ('day2', 'day3'). However, there are no transitions from 'day1' to 'day2'. 

If you do not have any clonal information, you can still run::
    
    cs.tmap.infer_Tmap_from_state_info_alone(adata_orig,initial_time_points,target_time_point,method='OT',**params)

It is the same as ``cs.tmap.infer_Tmap_from_one_time_clones`` except that we exclude the final joint optimization that requires clonal information. The resulting map is stored at ``adata.uns['OT_transition_map']`` or  ``adata.uns['HighVar_transition_map']``, depending on the method choice. 

We also provide simple methods that infer transition map from only the clonal information::

    cs.tmap.infer_Tmap_from_clonal_info_alone(adata,**params)

The result is stored at ``adata.uns['clonal_transition_map']``. 

Visualization
'''''''''''''

Finally, each of the computed transition maps can be explored on state embedding at the single-cell level using a variety of plotting functions. There are some common parameters: 

* ``used_map_name`` (``str``). It determines which transition map to use for analysis. Choices: {'transition_map', 'intraclone_transition_map', 'OT_transition_map', 'HighVar_transition_map','clonal_transition_map'}

* ``selected_fates`` (``list`` of ``str``). Selected clusters to aggregate differentiation dynamics and visualize fate bias etc.. It allows a nested structure, e.g., ``selected_fates``=['a', ['b', 'c']] selects two clusters:  cluster 'a' and the other that combines 'b' and 'c'. 

* ``map_backwards`` (``bool``, default ``True``).  We can analyze either the forward transitions, i.e., where the selected states or clusters are going (``map_backwards=False``), or the backward transitions, i.e., where these selected states or clusters came from (``map_backwards=False``). The latter is more useful and is the default. 

* ``normalize_by_fate_size`` (``bool``, default ``False``). Normalize the fate probability Prob(X) towards a cluster X by the expected probability, i.e., the fraction of cells within the targeted cluster X at the corresponding time point.

Below, we frame the task in the language of analyzing backward transitions (map_backwards=True) for convenience. To see where a cell came from, run:: 
    
    cs.pl.single_cell_transition(adata,**params)

To see the probability of initial cell states to give rise to given fate clusters, run::
    
    cs.pl.fate_map(adata,**params)

To infer the relative fate bias of initial cell states towards given fate clusters, run::
    
    cs.pl.binary_fate_bias(adata,**params)

The fate bias of initial states are defined by competition between two fate clusters A and B, i.e., how strongly A is favored than B. 

To infer the dynamic trajectory towards given fate clusters, run::

    cs.pl.dynamic_trajectory_from_binary_fate_bias(adata,**params)
    cs.pl.dynamic_trajectory_via_iterative_mapping(adata,**params)

The first method assumes two input fate clusters and infer each trajectory by thresholding the corresponding fate bias. It exports the selected ancestor states for the two fate clusters at ``adata.obs['cell_group_A']`` and ``adata.obs['cell_group_B']``, which can be used to infer the driver genes for fate bifurcation by running::
    
    cs.pl.differential_genes(adata,**params)

The second method (``dynamic_trajectory_via_iterative_mapping``) infers the trajectory by iteratively tracing a selected fate cluster all the way back to the initial time point. For both methods,  the inferred trajectory for each fate will be saved at ``adata.uns['dynamic_trajectory'][fate_name]``, and we can explore the gene expression dynamics along this trajectory using:: 

    cs.pl.gene_expression_dynamics(adata,selected_fate,gene_name_list,**params)

The ``selected_fate`` should be among those that have pre-computed dynamic trajectories. 


If there are multiple mature fate clusters, you can infer their differentiation coupling by::

    cs.pl.fate_coupling_from_Tmap(adata,**params)    



