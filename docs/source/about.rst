About CoSpar
------------

The following information is adapted from `Wang et al. (2021) <https://www.biorxiv.org/content/10.1101/2021.05.06.443026v1>`_.
High-throughput single-cell measurements have enabled unbiased studies of development and differentiation, leading to numerous methods for dynamic inference. However, single-cell RNA sequencing (scRNA-seq) data alone does not fully constrain the differentiation dynamics, and existing methods inevitably operate under simplified assumptions. In parallel, the lineage information of individual cells can be profiled simultaneously along with their transcriptome by using a heritable and expressible DNA barcode as a lineage tracer. The barcode may remain static or evolve over time.


However, the lineage data could be challenging to analyze.  These challenges include stochastic differentiation and variable expansion of clones; cells loss during analysis; barcode homoplasy wherein cells acquire the same barcode despite not having a lineage relationship; access to clones only at a single time point; and clonal dispersion due to a lag time between labeling cells and the first sampling (the lag time is necessary to allow the clone to grow large for resampling).


CoSpar, developed by `Wang et al. (2021) <https://www.biorxiv.org/content/10.1101/2021.05.06.443026v1>`_, is the first tool ever to perform dynamic inference by integrating state and lineage information. It solves for the transition probability map from cell states at an earlier time point to states at a later time point. It achieves accuracy and robustness by learning a sparse and coherent transition map, where neighboring initial states share similar yet sparse fate outcomes. Built upon the finite-time transition map, CoSpar can 1) infer fate potential of early states; 2) detect early fate bias (thus, fate boundary) among a heterogeneous progenitor population; 3) identify putative driver genes for fate bifurcation; 4) infer fate coupling or hierarchy; 5) visualize gene expression dynamics along an inferred differential trajectory. CoSpar also provides several methods to analyze clonal data by itself, including the clonal coupling between fate clusters and the bias of a clone towards a given fate, etc.  We envision CoSpar to be a platform to integrate key methods needed to analyze data with both state and lineage information.

.. image:: https://user-images.githubusercontent.com/4595786/113746452-56e4cb00-96d4-11eb-8278-0aac0469ba9d.png
   :width: 1000px
   :align: center


Coherent sparse optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One formalization of dynamic inference is to identify a transition map, a matrix :math:`T_{ij} (t_1,t_2)`, which describes the probability of a cell, initially in some state :math:`i` at time :math:`t_1`, giving rise to progeny in a state :math:`j` at time :math:`t_2`.  We define :math:`T_{ij} (t_1,t_2)` specifically as the fraction of progeny from state :math:`i` that occupy state :math:`j`. This transition matrix does not capture all we can learn about cell dynamics: it already combines the effects of cell division, loss, and differentiation. As will be seen, even learning :math:`T_{ij} (t_1,t_2)` will prove useful for several applications.

The transition map :math:`T` has properties that make it accessible to inference. We expect it to be a sparse matrix, since most cells can access just a few states during the experiment. And we expect it to be locally coherent, meaning that neighboring cell states share similar fate outcomes.  As inputs, CoSpar requires a clone-by-cell matrix :math:`I(t)` that encodes the clonal information at time :math:`t`, and a data matrix for observed cell states (e.g. from scRNA-Seq data).

CoSpar is formulated assuming that we have initial clonal information by re-sampling clones. When we only have the clonal information at one time point :math:`t_2`, we infer a putative transition map by jointly optimizing the matrix :math:`T` and the initial clonal data :math:`I(t_1)`. In this joint optimization, one must initialize the transition map, and we found that the final result is robust to initialization. This approach can be used for clones with nested structure. Finally, coherence and sparsity provide reasonable constraints when no clonal information is available, offering an approach to infer transition maps from state heterogeneity alone. We have extended CoSpar to this case.

.. image:: https://user-images.githubusercontent.com/4595786/113746670-93b0c200-96d4-11eb-89c0-d1e7d72383e7.png
   :width: 1000px
   :align: center

Below, we formalize the coherent sparse optimization by which CoSpar infers the transition map.

In a model of stochastic differentiation, cells in a clone are distributed across states with a time-dependent  density profile :math:`P(t)`. A transition map :math:`T` directly links clonal density profiles :math:`P(t_{1,2})`  between time points:

.. math::
	\begin{equation}
	P_i(t_2 )= \sum_j P_j(t_1 )T_{ji}(t_1,t_2),
	\end{equation}

From multiple clonal observations, our goal is to learn :math:`T`. To do so, we denote :math:`I(t)` as a clone-by-cell matrix and introduce :math:`S` as a matrix of cell-cell similarity over all observed cell states, including those lacking clonal information. The density profiles of all observed clones are estimated as :math:`P(t)\approx I(t)S(t)`.


Since the matrices :math:`P(t_{1,2})` are determined directly from data, with enough information :math:`T(t_1,t_2)` could be learnt by matrix inversion. However, in most cases, the number of clones is far less than the number of states. To constrain the map, we require that: 1)  :math:`T` is a sparse matrix; 2)  :math:`T` is locally coherent; and 3) :math:`T` is a non-negative matrix. With these requirements, the inference becomes an optimization problem:

.. math::
	\begin{equation}
	 \min_{T} ||T||_1+\alpha ||LT||_2,  \; \text{s.t.} \; ||P(t_2)- P(t_1) T(t_1,t_2)||_{2}\le\epsilon;\; T\ge 0; \text{Normalization}.
	 \end{equation}

Here, :math:`‖T‖_1` quantifies the sparsity of the matrix T through its l-1 norm, while  :math:`‖LT‖_2` quantifies the local coherence of :math:`T` (:math:`L` is the Laplacian of the cell state similarity graph, and :math:`LT` is the local divergence). The remaining constraints enforce the observed clonal dynamics, non-negativity of :math:`T`, and map normalization, respectively. At :math:`\alpha=0`, the minimization takes the form of Lasso, an algorithm for compressed sensing. Our formulation extends compressed sensing from vectors to matrices, and to enforce local coherence. The local coherence extension is reminiscent of the fused Lasso problem. An iterative, heuristic approach solves the CoSpar optimization efficiently. See `Wang et al. (2021) <https://www.biorxiv.org/content/10.1101/2021.05.06.443026v1>`_ for a detailed exposition of the method and its implementation.
