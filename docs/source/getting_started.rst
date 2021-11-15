Getting Started
---------------

Here, we explain the basics of using CoSpar. In :doc:`CoSpar basics <20210121_cospar_tutorial_v2>`, we will demonstrate its usage in a sub-sampled dataset of hematopoiesis.

CoSpar requires the count matrix not log-transformed. This is specifically assumed in selecting highly variable genes, in computing PCA, and in the HighVar method for initializing the joint optimization using a single clonal time point. CoSpar also assumes that the dataset has more than one time point. However, if you have only a snapshot, you can still manually cluster the cells into more than one time point to use CoSpar.

First, import CoSpar with::

    import cospar as cs

For better visualization you can change the matplotlib settings to our defaults with::

    cs.settings.set_figure_params()

If you want to adjust parameters for a particular plot, just pass the parameters into this function.

Initialization
''''''''''''''
Given the gene expression matrix, clonal matrix, and other information, initialize the anndata object using::

    adata_orig = cs.pp.initialize_adata_object(adata=None,**params)

The :class:`~anndata.AnnData` object ``adata_orig`` stores the count matrix (``adata_orig.X``), gene names (``adata_orig.var_names``), and temporal annotation of cells (``adata_orig.obs['time_info']``).  Optionally, you can also provide the clonal matrix ``X_clone``, selected PCA matrix ``X_pca``,  the embedding matrix ``X_emb``, and the state annotation ``state_info``, which will be stored at ``adata_orig.obsm['X_clone']``,  ``adata_orig.obsm['X_pca']``, ``adata_orig.obsm['X_emb']``, and ``adata_orig.obs['state_info']``, respectively.

If an adata object is provided as an input, the initialization function will try to automatically generate the correct data structure, and all annotations associated with the provided adata will remain intact. You can add new annotations to supplement or override existing annotations in the adata object.


.. raw:: html

    <img src="http://falexwolf.de/img/scanpy/anndata.svg" style="width: 300px">

If you do not have a dataset yet, you can still play around using one of the built-in datasets, e.g.::

    adata_orig = cs.datasets.hematopoiesis_subsampled()



Preprocessing & dimension reduction
'''''''''''''''''''''''''''''''''''
Assuming basic quality control (excluding cells with low read count etc.) have been done, we provide basic preprocessing (gene selection and normalization) and dimension reduction related analysis (PCA, UMAP embedding etc.)  at ``cs.pp.*``::

    cs.pp.get_highly_variable_genes(adata_orig,**params)
    cs.pp.remove_cell_cycle_correlated_genes(adata_orig,**params)
    cs.pp.get_X_pca(adata_orig,**params)
    cs.pp.get_X_emb(adata_orig,**params)
    cs.pp.get_state_info(adata_orig,**params)
    cs.pp.get_X_clone(adata_orig,**params)

The first step ``get_highly_variable_genes`` also includes count matrix normalization. The second step, which is optional but recommended, removes cell cycle correlated genes among the selected highly variable genes. In ``get_X_pca``, we apply z-score transformation for each gene expression before computing the PCA. In ``get_X_emb``, we simply use the umap function from :mod:`~scanpy`. With ``get_state_info``, we extract state information using leiden clustering implemented in :mod:`~scanpy`.
In ``get_X_clone``, we faciliate the conversion of the raw clonal data into a cell-by-clone matrix. As mentioned before, this preprocessing assumes that the count matrix is not log-transformed.




Basic clonal analysis
''''''''''''''''''''''
We provide a few plotting functions to help visually exploring the clonal data before any downstream analysis. You can visualize clones on state manifold directly::

    cs.pl.clones_on_manifold(adata_orig,**params)

You can generate the barcode heatmap across given clusters to inspect clonal behavior::

    cs.pl.barcode_heatmap(adata_orig,**params)

You can quantify the clonal coupling across different fate clusters::

    cs.pl.fate_coupling_from_clones(adata_orig,**params)

Strong coupling implies the existence of bi-potent or multi-potent cell states at the time of barcoding. You can visualize the fate hierarchy by a simple neighbor-joining method::

    cs.pl.fate_hierarchy_from_clones(adata_orig,**params)

Finally, you can infer the fate bias :math:`-log_{10}(P_{value})` of each clone towards a designated fate cluster::

    cs.pl.clonal_fate_bias(adata_orig,**params)

A biased clone towards this cluster has a statistically significant cell fraction within or outside this cluster.




Transition map inference
''''''''''''''''''''''''
The core of the software is efficient and robust inference of a transition map by integrating state and clonal information. If the dataset has multiple clonal time points, you can run::

    adata=cs.tmap.infer_Tmap_from_multitime_clones(adata_orig,clonal_time_points=None,later_time_point=None,**params)

It subsamples the input data at selected time points and computes the transition map, stored at ``adata.uns['transition_map']`` and ``adata.uns['intraclone_transition_map']``, with the latter restricted to intra-clone transitions. Depending on ``later_time_point``, it has two modes of inference:

1) When ``later_time_point=None``, it infers a transition map between neighboring time points. For example, for clonal_time_points=['day1', 'day2', 'day3'], it computes transitions for pairs ('day1', 'day2') and ('day2', 'day3'), but not for ('day1', 'day3').

2) If ``later_time_point`` is specified, it generates a transition map between this time point and each of the earlier time points. In the previous example, if ``later_time_point=='day3'``, we infer transitions for pairs ('day1', 'day3') and ('day2', 'day3'). This applies to the following map inference functions.


-------------------------------------

If the dataset has only one clonal time point, you can run::

    adata=cs.tmap.infer_Tmap_from_one_time_clones(adata_orig,initial_time_points=None, later_time_point=None,initialize_method='OT',**params)

which jointly optimizes the transition map and the initial clonal structure. It requires initializing the transition map using state information alone. We provide two methods for such initialization: 1) ``OT`` for using the standard optimal transport approach; 2) ``HighVar`` for a customized approach, assuming that cells similar in gene expression across time points share clonal origin. For the ``OT`` method, if you wish to utilize the growth rate information as Waddington-OT, you can directly pass the growth rate estimate for each cell to the input AnnaData object at ``adata_orig.obs["cell_growth_rate"]``. Depending on the choice,  the initialized map is stored at ``adata.uns['OT_transition_map']`` or  ``adata.uns['HighVar_transition_map']``. The final product is stored at ``adata.uns['transition_map']``.

``HighVar`` converts highly variable genes into pseudo multi-time clones and infers a putative map with coherent sparse optimization. We find the ``HighVar`` method performs better than the `OT` method, especially when there are large differentiation effects over the observed time window, or batch effects.

If ``initial_time_points`` and ``later_time_point`` are not specified, a map with transitions from all time points to the last time point is generated.

-------------------------------------

If you do not have any clonal information, you can still run::

    adata=cs.tmap.infer_Tmap_from_state_info_alone(adata_orig,initial_time_points=None,later_time_point=None,initialize_method='OT',**params)

It is the same as ``cs.tmap.infer_Tmap_from_one_time_clones`` except that we assume a pseudo clonal data where each cell at the later time point occupies a unique clone.

-------------------------------------

We also provide simple methods that infer transition map from clonal information alone::

    adata=cs.tmap.infer_Tmap_from_clonal_info_alone(adata_orig,clonal_time_points=None,later_time_point=None,**params)

The result is stored at ``adata.uns['clonal_transition_map']``.

Visualization
'''''''''''''

Finally, each of the computed transition maps can be explored on state embedding at the single-cell level using a variety of plotting functions. There are some common parameters: 1) ``used_Tmap``, for choosing one of the pre-computed transition maps for analysis; 2) ``selected_fates``, for visualizing the fate bias towards/against given fate clusters; 3) ``map_backward``, for analyzing forward or backward transitions; 4) ``method``, for different methods in fate probability analysis. See :doc:`CoSpar basics <20210121_cospar_tutorial_v2>` for more details.


Below, we frame the task in the language of analyzing backward transitions for convenience. To see where a cell came from, run::

    cs.pl.single_cell_transition(adata,**params)

To visualize the fate probability of initial cell states, run::

    cs.pl.fate_map(adata,**params)

To infer the fate bias of initial cell states between two fate clusters, run::

    cs.pl.fate_bias(adata,**params)

To infer the dynamic trajectory towards given fate clusters, run::

    cs.pl.progenitor(adata,**params)
    cs.pl.iterative_differentiation(adata,**params)

The first method assumes two input fate clusters and infers each trajectory by thresholding the corresponding fate bias. The second method infers the trajectory by iteratively tracing a selected fate cluster all the way back to its putative origin at the initial time point. For both methods,  the inferred trajectory for each fate will be saved at ``adata.obs[f'traj_{fate_name}']``, and we can explore the gene expression dynamics along this trajectory using::

    cs.pl.gene_expression_dynamics(adata,**params)

Additionally, the first method (``cs.pl.progenitor``) exports the selected ancestor states for the two fate clusters at ``adata.obs['cell_group_A']`` and ``adata.obs['cell_group_B']``, which can be used to infer the driver genes for fate bifurcation by running::

    cs.pl.differential_genes(adata,**params)


If there are multiple mature fate clusters, you can infer their differentiation coupling from the fate probabilities of initial cells by::

    cs.pl.fate_coupling_from_Tmap(adata,**params)

You can also infer the fate hierarchy from::

    cs.pl.fate_hierarchy_from_Tmap(adata,**params)
