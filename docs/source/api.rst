.. automodule:: cospar

API
===

Import CoSpar as::

   import cospar as cs


CoSpar is built around the :class:`~anndata.AnnData` object (usually called `adata`). For each cell, we store its RNA count matrix at ``adata.X``, the gene names at ``adata.var_names``,time information at ``adata.obs['time_info']``, state annotation at ``adata.obs['state_info']``,  clonal information at ``adata.obsm['X_clone']``, and 2-d embedding at ``adata.obsm['X_emb']``. 


Once the :class:`~anndata.AnnData` object is initialized via :func:`cs.pp.initialize_adata_object`, the typical flow of analysis is to 1) perform preprocessing and dimension reduction (``cs.pp.*``); 2) visualize and analyzing clonal data alone (``cs.pl.*``); 3) infer transition map (``cs.tmap.*``); and 4) analyze inferred map using the plotting functions (``cs.pl.*``). We also provide several built-in datasets (``cs.datasets.*``) and miscellaneous functions to assist with the analysis (``cs.hf.*``). See :doc:`tutorial <getting_started>` for details. 



Preprocessing
-------------

.. autosummary::
   :toctree: .

   pp.initialize_adata_object
   pp.get_highly_variable_genes
   pp.remove_cell_cycle_correlated_genes
   pp.get_X_pca
   pp.get_X_emb
   pp.get_state_info
   pp.refine_state_info_by_marker_genes
   pp.refine_state_info_by_leiden_clustering



Transition map inference
------------------------


.. autosummary::
   :toctree: .

   tmap.infer_Tmap_from_multitime_clones
   tmap.infer_intraclone_Tmap
   tmap.infer_Tmap_from_one_time_clones
   tmap.infer_Tmap_from_state_info_alone
   tmap.infer_Tmap_from_clonal_info_alone

..
   **Internal functions** 

   .. autosummary::
      :toctree: .

      tmap.generate_similarity_matrix
      tmap.generate_initial_similarity
      tmap.generate_final_similarity

      tmap.select_time_points

      tmap.infer_Tmap_from_multitime_clones_private
      tmap.refine_Tmap_through_cospar_noSmooth
      tmap.refine_Tmap_through_cospar

      tmap.compute_custom_OT_transition_map
      tmap.Tmap_from_highly_variable_genes
      tmap.infer_Tmap_from_one_time_clones_private
      tmap.infer_Tmap_from_one_time_clones_twoTime


Plotting
--------


**Clone analysis** (clone visualization, clustering etc.)

.. autosummary::
   :toctree: .

   pl.clones_on_manifold
   pl.barcode_heatmap
   pl.clonal_fate_bias
   pl.fate_coupling_from_clones
   pl.fate_hierarchy_from_clones



**Transition map analysis** (fate bias etc.)

.. autosummary::
   :toctree: .

   pl.single_cell_transition
   pl.fate_map
   pl.fate_bias
   pl.dynamic_trajectory_from_fate_bias
   pl.dynamic_trajectory_via_iterative_mapping
   pl.gene_expression_dynamics
   pl.fate_coupling_from_Tmap
   pl.fate_hierarchy_from_Tmap


**Differential gene expression analysis** 

.. autosummary::
   :toctree: .
   
   pl.differential_genes
   pl.differential_genes_for_given_fates
   
**General**

.. autosummary::
   :toctree: .

   pl.embedding
   pl.gene_expression_on_manifold
   settings.set_figure_params


Datasets
--------

.. autosummary::
   :toctree: .

   datasets.hematopoiesis_subsampled
   datasets.hematopoiesis_all
   datasets.lung
   datasets.reprogramming_static_BC
   datasets.reprogramming_dynamic_BC
   datasets.synthetic_bifurcation_static_BC
   datasets.synthetic_bifurcation_dynamic_BC



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



..
   hf.get_normalized_covariance
   hf.add_neighboring_cells_to_a_map
   hf.compute_shortest_path_distance
   hf.get_dge_SW
   hf.compute_fate_probability_map
   hf.compute_state_potential
   hf.filter_genes
   hf.compute_fate_map_and_intrinsic_bias
   hf.mapout_trajectories


