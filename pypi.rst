|PyPI| |Docs|

CoSpar - dynamic inference by integrating state and lineage information
=======================================================================

.. image:: https://user-images.githubusercontent.com/4595786/104988296-b987ce00-59e5-11eb-8dbe-a463b355a9fd.png
   :width: 300px
   :align: left

**CoSpar** is a toolkit for dynamic inference from lineage-traced single cells.
The methods are based on
`Wang et al. (2021) <https://www.biorxiv.org/content/10.1101/2021.05.06.443026v1>`_.

Dynamic inference based on single-cell state measurement alone requires serious simplifications. On the other hand, direct dynamic measurement via lineage tracing only captures partial information and its interpretation is challenging. CoSpar integrates both state and lineage information to infer a finite-time transition map of a development/differentiation system. It gains superior robustness and accuracy by exploiting both the local coherence and sparsity of differentiation transitions, i.e., neighboring initial states share similar yet sparse fate outcomes.  Building around the anndata_ object, CoSpar provides an integrated analysis framework for datasets with both state and lineage information. When only state information is available, CoSpar also improves upon existing dynamic inference methods by imposing sparsity and coherence. It offers essential toolkits for analyzing lineage data, state information, or their integration. 

See `<https://cospar.readthedocs.io>`_ for documentation and tutorials.


Reference
---------
Shou-Wen Wang and Allon M. Klein (2021), Learning dynamics by computational integration of single cell genomic and lineage information,
`bioRxiv <https://www.biorxiv.org/content/10.1101/2021.05.06.443026v1>`_.

Support
-------
Feel free to submit an `issue <https://github.com/AllonKleinLab/cospar/issues/new/choose>`_
or send us an `email <mailto:wangsw09@gmail.com>`_.
Your help to improve CoSpar is highly appreciated.

.. |PyPI| image:: https://img.shields.io/pypi/v/cospar.svg
   :target: https://pypi.org/project/cospar

.. |Docs| image:: https://readthedocs.org/projects/cospar/badge/?version=latest
   :target: https://cospar.readthedocs.io


.. _anndata: https://anndata.readthedocs.io