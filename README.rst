|PyPI| |PyPIDownloads| |Docs|

CoSpar - dynamic inference by integrating state and lineage information
=======================================================================

.. image:: https://user-images.githubusercontent.com/4595786/104988296-b987ce00-59e5-11eb-8dbe-a463b355a9fd.png
   :width: 300px
   :align: left

**CoSpar** is a toolkit for dynamic inference from lineage-traced single cells.
The methods are based on
`Wang et al. Nat. Biotech. (2022) <https://www.nature.com/articles/s41587-022-01209-1>`_.

Dynamic inference based on single-cell state measurement alone requires serious simplifications. On the other hand, direct dynamic measurement via lineage tracing only captures partial information and its interpretation is challenging. CoSpar integrates both state and lineage information to infer a finite-time transition map of a development/differentiation system. It gains superior robustness and accuracy by exploiting both the local coherence and sparsity of differentiation transitions, i.e., neighboring initial states share similar yet sparse fate outcomes.  Building around the anndata_ object, CoSpar provides an integrated analysis framework for datasets with both state and lineage information. When only state information is available, CoSpar also improves upon existing dynamic inference methods by imposing sparsity and coherence. It offers essential toolkits for analyzing lineage data, state information, or their integration.

See `<https://cospar.readthedocs.io>`_ for documentation and tutorials.

Package development relocation
------------------------------
Effective on April 1st 2023, Shou-Wen Wang is leaving the Klein lab to start `his own group at Westlake University <https://www.shouwenwang-lab.com/>`_, and he will no longer maintain this repository. Further development of CoSpar will continue in his own lab under this respository `https://github.com/ShouWenWang-Lab/cospar <https://github.com/ShouWenWang-Lab/cospar>`_. Please reach out there for any issues related to CoSpar.

Recorded talks
--------
`Jun 1: Single-Cell Data Science 2022 <https://singlecell2022.hku.hk/>`_. This is a 20-min short talk focusing more on the utility of CoSpar: `talk video <https://www.youtube.com/watch?v=HrDQpW3kJFo>`_

`Oct 19, 2022: Invited MIA talk at Broad Institute <https://www.broadinstitute.org/talks/learning-cell-differentiation-dynamics-lineage-tracing-datasets>`_. This is an one-hour talk focusing on the Machine Learning part of CoSpar: `talk video <https://www.youtube.com/watch?v=rYzQUYPPNlU>`_.  The talk slides can be found `here <https://github.com/ShouWenWang-Lab/cospar/files/10042131/20221118_public_posted_slides.pdf>`_.

Reference
---------
`S.-W. Wang*, M. Herriges, K. Hurley, D. Kotton, A. M. Klein*, CoSpar identifies early cell fate biases from single cell transcriptomic and lineage information, Nat. Biotech. (2022) <https://www.nature.com/articles/s41587-022-01209-1>`_. [* corresponding authors]

Support
-------
Feel free to submit an `issue <https://github.com/ShouWenWang-Lab/cospar/issues/new/choose>`_
or send us an `email <mailto:wangshouwen@westlake.edu.cn>`_.
Your help to improve CoSpar is highly appreciated.



.. _anndata: https://anndata.readthedocs.io

.. |PyPI| image:: https://img.shields.io/pypi/v/cospar.svg
   :target: https://pypi.org/project/cospar

.. |PyPIDownloads| image:: https://pepy.tech/badge/cospar
   :target: https://pepy.tech/project/cospar

.. |Docs| image:: https://readthedocs.org/projects/cospar/badge/?version=latest
   :target: https://cospar.readthedocs.io
