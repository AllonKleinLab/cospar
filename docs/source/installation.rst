Installation
------------

CoSpar requires Python 3.6 or later. We recommend using Miniconda_.

PyPI
^^^^

Install CoSpar from PyPI_ using::

    pip install --upgrade cospar

If you get a ``Permission denied`` error, use ``pip install --upgrade cospar --user`` instead.


Development Version
^^^^^^^^^^^^^^^^^^^

To work with the latest development version, install from GitHub_ using::

    pip install git+https://github.com/AllonKleinLab/cospar

or::

    git clone https://github.com/AllonKleinLab/cospar
    pip install -e cospar

``-e`` is short for ``--editable`` and links the package to the original cloned location such that pulled changes are also reflected in the environment.


Dependencies
^^^^^^^^^^^^

- `anndata <https://anndata.readthedocs.io/>`_ - annotated data object.
- `scanpy <https://scanpy.readthedocs.io/>`_ - toolkit for single-cell analysis.
- `numpy <https://docs.scipy.org/>`_, `scipy <https://docs.scipy.org/>`_, `pandas <https://pandas.pydata.org/>`_, `scikit-learn <https://scikit-learn.org/>`_, `matplotlib <https://matplotlib.org/>`_, `plotnine <https://plotnine.readthedocs.io/>`_, 



Jupyter Notebook
^^^^^^^^^^^^^^^^

To run the tutorials in a notebook locally, please install::

   conda install notebook

and run ``jupyter notebook`` in the terminal. If you get the error ``Not a directory: 'xdg-settings'``,
use ``jupyter notebook --no-browser`` instead and open the url manually (or use this
`bugfix <https://github.com/jupyter/notebook/issues/3746#issuecomment-444957821>`_).


If you run into issues, do not hesitate to approach us or raise a `GitHub issue`_.

.. _Miniconda: http://conda.pydata.org/miniconda.html
.. _PyPI: https://pypi.org/project/cospar
.. _Github: https://github.com/AllonKleinLab/cospar/
.. _`Github issue`: https://github.com/AllonKleinLab/cospar/issues/new/choose