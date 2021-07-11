|PyPI| |PyPIDownloads| |Docs|

CoSpar - dynamic inference by integrating state and lineage information
=======================================================================

.. image:: https://user-images.githubusercontent.com/4595786/104988296-b987ce00-59e5-11eb-8dbe-a463b355a9fd.png
   :width: 300px
   :align: left

**CoSpar** is a toolkit for dynamic inference from lineage-traced single cells. |br|
The methods are based on
`Wang et al. (2021) <https://www.biorxiv.org/content/10.1101/2021.05.06.443026v1>`_.

Dynamic inference based on single-cell state measurement alone requires serious simplifications. On the other hand, direct dynamic measurement via lineage tracing only captures partial information and its interpretation is challenging. CoSpar integrates both state and lineage information to infer a finite-time transition map of a development/differentiation system. It gains superior robustness and accuracy by exploiting both the local coherence and sparsity of differentiation transitions, i.e., neighboring initial states share similar yet sparse fate outcomes.  Building around the :class:`~anndata.AnnData` object, CoSpar provides an integrated analysis framework for datasets with both state and lineage information. When only state information is available, CoSpar also improves upon existing dynamic inference methods by imposing sparsity and coherence. It offers essential toolkits for analyzing lineage data, state information, or their integration. 

CoSpar's key applications
^^^^^^^^^^^^^^^^^^^^^^^^^
- infer transition maps from lineage data, state measurements, or their integration. 
- predict the fate bias of progenitor cells. 
- order cells along a differentiation trajectory leading to a given cell fate.
- predict gene expression dynamics along a trajectory. 
- predict genes whose expression correlates with future fate outcome.
- generate a putative fate hierarchy, ordering fates by their lineage distances. 


.. Upcoming talks
.. ^^^^^^^^^^^^^^
.. - `Sep 15: Temporal Single-Cell Analysis (SCOG) <https://twitter.com/fabian_theis/status/1305621028056465412>`_
.. - `Nov 12: Single Cell Biology (SCB) <https://coursesandconferences.wellcomegenomecampus.org/our-events/single-cell-biology-2020/>`_



Reference
^^^^^^^^^
Shou-Wen Wang and Allon M. Klein (2021), Learning dynamics by computational integration of single cell genomic and lineage information,
`bioRxiv <https://www.biorxiv.org/content/10.1101/2021.05.06.443026v1>`_.



Support
^^^^^^^
Feel free to submit an `issue <https://github.com/AllonKleinLab/cospar/issues/new/choose>`_
or send us an `email <mailto:wangsw09@gmail.com>`_.
Your help to improve CoSpar is highly appreciated.

Acknowledgment
^^^^^^^^^^^^^^
Shou-Wen Wang wants to acknowledge `Xiaojie Qiu <https://dynamo-release.readthedocs.io/>`_ for inspiring him to make this website. He also wants to acknowledge the community that maintains `scanpy <https://scanpy.readthedocs.io/>`_ and `scvelo <https://scvelo.readthedocs.io/>`_, where he learned about proper code documentation. He thanks Tal Debrobrah Scully, Qiu Wu  and other lab members for testing the package. Shou-Wen  wants to thank especially Allon Klein for his mentorship. Finally, he wants to acknowledge the generous support of the Damon Runyon Foundation through the Quantitative Biology Fellowship.     


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
   20210121_cospar_tutorial_v2
   20210601_loading_data

.. toctree::
   :caption: Examples
   :maxdepth: 1
   :hidden:

   20210121_all_hematopoietic_data_v3
   20210121_reprogramming_static_barcoding_v2
   20210121_lung_data_v2
   20210120_bifurcation_model_static_barcoding





.. |PyPI| image:: https://img.shields.io/pypi/v/cospar.svg
   :target: https://pypi.org/project/cospar

.. |PyPIDownloads| image:: https://pepy.tech/badge/cospar
   :target: https://pepy.tech/project/cospar

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