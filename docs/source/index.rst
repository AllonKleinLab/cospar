|PyPI| |Docs|

CoSpar - dynamic inference by integrating transcriptome and lineage information
===============================================================================

.. image:: https://user-images.githubusercontent.com/4595786/104988296-b987ce00-59e5-11eb-8dbe-a463b355a9fd.png
   :width: 300px
   :align: left

**CoSpar** is a toolkit for dynamic inference from lineage-traced single cells. |br|
The methods are based on
`S.-W. Wang & A.M. Klein (ToBeSubmitted, 2021) <https://doi.org/xxx>`_.

Dynamic inference based on single-cell state measurement alone requires serious simplifications. On the other hand, direct dynamic measurement via lineage tracing only captures partial information and is very noisy. CoSpar integrates both state and lineage information to infer the transition map of a development/differentiation system. It gains superior robustness and accuracy by exploiting both the local coherence and sparsity of differentiation transitions, i.e., neighboring initial states share similar yet sparse fate outcomes.  Building around the most popular :class:`~anndata.AnnData` object in the single-cell community, CoSpar is dedicated to building an integrated analysis framework for datasets with both state and lineage information. It offers essential toolkits for analyzing clonal information, state information, or their integration. 

CoSpar's key applications
^^^^^^^^^^^^^^^^^^^^^^^^^
- infer transition maps using only clonal information, state information, or their integration. 
- identify early fate bias/commitment of a cell 
- infer differentiation trajectories leading to a fate.
- infer gene expression dynamics along the trajectory. 
- infer putative driver genes.
- infer fate coupling.


.. Upcoming talks
.. ^^^^^^^^^^^^^^
.. - `Sep 15: Temporal Single-Cell Analysis (SCOG) <https://twitter.com/fabian_theis/status/1305621028056465412>`_
.. - `Nov 12: Single Cell Biology (SCB) <https://coursesandconferences.wellcomegenomecampus.org/our-events/single-cell-biology-2020/>`_



Reference
^^^^^^^^^
Shou-Wen Wang & Allon M. Klein (2021), Coherent sparsity optimization for dynamic inference by integrating state and lineage information,
`ToBeSubmitted <https://doi.org/xxx>`_.
|dim|


Support
^^^^^^^
Feel free to submit an `issue <https://github.com/AllonKleinLab/cospar/issues/new/choose>`_
or send us an `email <mailto:wangsw09@gmail.com>`_.
Your help to improve CoSpar is highly appreciated.



.. toctree::
   :caption: Main
   :maxdepth: 1
   :hidden:

   about
   api
   

.. toctree::
   :caption: Tutorial 
   :maxdepth: 1
   :hidden:

   installation
   getting_started
   20210121_cospar_tutorial

.. toctree::
   :caption: Examples
   :maxdepth: 1
   :hidden:

   20210121_all_hematopoietic_data
   20210121_reprogramming_data_merge_tags
   20210121_reprogramming_data_no_merge_tags
   20210121_lung_data
   20210120-Bifurcation_model_static_barcoding
   20210120-Bifurcation_model_dynamic_barcoding




.. |PyPI| image:: https://img.shields.io/pypi/v/cospar.svg
   :target: https://pypi.org/project/cospar


.. |Docs| image:: https://readthedocs.org/projects/cospar/badge/?version=latest
   :target: https://cospar.readthedocs.io


..
  .. |travis| image:: https://travis-ci.org/theislab/cospar.svg?branch=master
     :target: https://travis-ci.org/theislab/cospar


.. |br| raw:: html

  <br/>

..
 .. |meet| raw:: html



.. |dim| raw:: html

   <span class="__dimensions_badge_embed__" data-doi="xxx" data-style="small_rectangle"></span>
   <script async src="https://badge.dimensions.ai/badge.js" charset="utf-8"></script>