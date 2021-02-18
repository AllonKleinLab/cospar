|PyPI| |Docs|

CoSpar - dynamic inference by integrating state and lineage information
=======================================================================

.. image:: https://user-images.githubusercontent.com/4595786/104988296-b987ce00-59e5-11eb-8dbe-a463b355a9fd.png
   :width: 300px
   :align: left

**CoSpar** is a toolkit for dynamic inference from lineage-traced single cells.
The methods are based on
`S.-W. Wang & A.M. Klein (ToBeSubmitted, 2021) <https://doi.org/xxx>`_.

Dynamic inference based on single-cell state measurement alone requires serious simplifications. On the other hand, direct dynamic measurement via lineage tracing only captures partial information and is very noisy. CoSpar integrates both state and lineage information to infer the transition map of a development/differentiation system. It gains superior robustness and accuracy by exploiting both the local coherence and sparsity of differentiation transitions, i.e., neighboring initial states share similar yet sparse fate outcomes.  Building around the most popular anndata_ object in the single-cell community, CoSpar provides an integrated analysis framework for datasets with both state and lineage information. It offers essential toolkits for analyzing lineage data, state information, or their integration. 

See `<https://cospar.readthedocs.io>`_ for documentation and tutorials.

CoSpar's key applications
-------------------------
- infer transition maps from lineage data, state measurements, or their integration. 
- predict early fate bias or commitment. 
- learn differentiation trajectories leading to a fate.
- predict gene expression dynamics along the trajectory. 
- predict putative driver genes.
- estimate fate coupling.


Reference
---------
Shou-Wen Wang & Allon M. Klein (2021), Learning clonal dynamics by coherent sparse optimization,
`ToBeSubmitted <https://doi.org/xxx>`_.

Support
-------
Feel free to submit an `issue <https://github.com/AllonKleinLab/cospar/issues/new/choose>`_
or send us an `email <mailto:wangsw09@gmail.com>`_.
Your help to improve CoSpar is highly appreciated.


Acknowledgment
--------------
Shou-Wen Wang wants to acknowledge `Xiaojie Qiu <https://dynamo-release.readthedocs.io/>`_ for inspiring him to make this website during a simulated discussion. He also wants to acknowledge the community that maintains `scanpy <https://scanpy.readthedocs.io/>`_ and `scvelo <https://scvelo.readthedocs.io/>`_. He learned a lot about code documentation and how to make a technical website from these two packages. Finally, he wants to thank Allon Klein for his mentorship and thoughtful suggestions, and Tal Debrobrah Scully and Qiu Wu for testing the package and making suggestions.   

.. _anndata: https://anndata.readthedocs.io

.. |PyPI| image:: https://img.shields.io/pypi/v/cospar.svg
   :target: https://pypi.org/project/cospar

.. |Docs| image:: https://readthedocs.org/projects/cospar/badge/?version=latest
   :target: https://cospar.readthedocs.io