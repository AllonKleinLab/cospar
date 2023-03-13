Installation
------------

CoSpar requires Python 3.6 or later. We recommend using Miniconda_ for package management. For a computer that does not have a package management tool yet, please install Miniconda_ first, and activate it by running the following command in the terminal::

	source ~/.bash_profile

PyPI
^^^^

Install CoSpar from PyPI_ using::

    pip install --upgrade cospar

If you get a ``Permission denied`` error, use ``pip install --upgrade cospar --user`` instead.

If you get errors related to 'gcc', try to specify the following gcc path for installation::

	env CXX=/usr/local/Cellar/gcc/8.2.0/bin/g++-8 CC=/usr/local/Cellar/gcc/8.2.0/bin/gcc-8 pip install  cospar

If you get errors for version conflicts with existing packages, try::

	    pip install --ignore-installed --upgrade cospar

Development Version
^^^^^^^^^^^^^^^^^^^

To work with the latest development version, install from GitHub_ using::

    pip install git+https://github.com/ShouWenWang-Lab/cospar

or::

    git clone https://github.com/ShouWenWang-Lab/cospar
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
.. _Github: https://github.com/ShouWenWang-Lab/cospar/
.. _`Github issue`: https://github.com/ShouWenWang-Lab/cospar/issues/new/choose


Testing CoSpar in a new environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In case you want to test cospar without affecting existing python packages, you can create a new conda environment and install CoSpar there::

	conda create -n test_cospar python=3.6
	conda activate test_cospar
	pip install cospar

Now, install jupyter notebook in this environment::

	pip install --user ipykernel

If you encounter an error related to ``nbconvert``, run (this is optional)::

	pip3 install --upgrade --user nbconvert

Finally, install the jupyter notebook kernel related to this environment::

	python -m ipykernel install --user --name=test_cospar

Now, you can open jupyter notebook by running ``jupyter notebook`` in the terminal, and select the kernel ``test_cospar`` to run CoSpar.
