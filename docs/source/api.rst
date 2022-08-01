.. automodule:: cospar

API
===

..    include:: <isonum.txt>

Import CoSpar as::

   import cospar as cs


CoSpar is built around the :class:`~anndata.AnnData` object (usually called `adata`). For each cell, we store its RNA count matrix at ``adata.X``, the gene names at ``adata.var_names``,time information at ``adata.obs['time_info']``, state annotation at ``adata.obs['state_info']``,  clonal information at ``adata.obsm['X_clone']``, and 2-d embedding at ``adata.obsm['X_emb']``.


Once the :class:`~anndata.AnnData` object is initialized via :func:`cs.pp.initialize_adata_object`, the typical flow of analysis is to 1) perform preprocessing and dimension reduction (``cs.pp.*``); 2) visualize and analyze clonal data alone (``cs.pl.*``); 3) infer transition map (``cs.tmap.*``); and 4) analyze inferred map (``cs.tl.*``) and then visualize the results with the plotting functions (``cs.pl.*``). Typically, each ``cs.tl.*`` function has a corresponding ``cs.pl.*`` function. We also provide several built-in datasets (``cs.datasets.*``) and miscellaneous functions to assist with the analysis (``cs.hf.*``). See :doc:`tutorial <getting_started>` for details.



Preprocessing
-------------

.. autosummary::
   :toctree: .

   pp.initialize_adata_object
   pp.get_highly_variable_genes
   pp.remove_cell_cycle_correlated_genes
   pp.get_X_pca
   pp.get_X_emb
   pp.get_X_clone
   pp.get_state_info
   pp.refine_state_info_by_marker_genes
   pp.refine_state_info_by_leiden_clustering




Transition map inference
------------------------


.. autosummary::
   :toctree: .

   tmap.infer_Tmap_from_multitime_clones
   tmap.infer_Tmap_from_one_time_clones
   tmap.infer_Tmap_from_state_info_alone
   tmap.infer_Tmap_from_clonal_info_alone


Analysis
----------

.. autosummary::
   :toctree: .

   tl.clonal_fate_bias
   tl.fate_biased_clones
   tl.fate_coupling
   tl.fate_hierarchy
   tl.fate_map
   tl.fate_potency
   tl.fate_bias
   tl.progenitor
   tl.iterative_differentiation
   tl.differential_genes


Plotting
---------


**Clone analysis** (clone visualization, clustering etc.)

.. autosummary::
   :toctree: .

   pl.clones_on_manifold
   pl.barcode_heatmap
   pl.clonal_fate_bias
   pl.fate_coupling
   pl.fate_hierarchy
   pl.clonal_fates_across_time
   pl.clonal_reports
   pl.visualize_tree



**Transition map analysis** (fate bias etc.)

.. autosummary::
   :toctree: .

   pl.single_cell_transition
   pl.fate_map
   pl.fate_potency
   pl.fate_bias
   pl.progenitor
   pl.iterative_differentiation
   pl.gene_expression_dynamics
   pl.fate_coupling
   pl.fate_hierarchy

**General**

.. autosummary::
   :toctree: .

   pl.embedding
   pl.embedding_genes
   pl.gene_expression_on_manifold
   pl.gene_expression_heatmap
   settings.set_figure_params


Datasets
--------

.. autosummary::
   :toctree: .

   datasets.hematopoiesis
   datasets.hematopoiesis_130K
   datasets.hematopoiesis_subsampled
   datasets.hematopoiesis_Gata1_states
   datasets.lung
   datasets.reprogramming
   datasets.reprogramming_Day0_3_28
   datasets.synthetic_bifurcation

Help functions
--------------

.. autosummary::
   :toctree: .

   hf.read
   hf.save_map
   hf.save_preprocessed_adata
   hf.check_adata_structure
   hf.check_available_choices
   hf.update_time_ordering
   hf.update_data_description
   tl.get_normalized_covariance
   hf.get_X_clone_with_reference_ordering


Simulations
-----------

.. autosummary::
   :toctree: .

   simulate.linear_differentiation_model
   simulate.bifurcation_model
   simulate.quantify_correlation_with_ground_truth_fate_bias_BifurcationModel
   simulate.quantify_transition_peak_TPR_LinearDifferentiation
