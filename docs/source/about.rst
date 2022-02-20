About CoSpar
------------

The following information is adapted from `Wang et al. Nat. Biotech. (2022) <https://www.nature.com/articles/s41587-022-01209-1>`_.
High-throughput single-cell measurements have enabled unbiased studies of development and differentiation, leading to numerous methods for dynamic inference. However, single-cell RNA sequencing (scRNA-seq) data alone does not fully constrain the differentiation dynamics, and existing methods inevitably operate under simplified assumptions. In parallel, the lineage information of individual cells can be profiled simultaneously along with their transcriptome by using a heritable and expressible DNA barcode as a lineage tracer (we call lineage-tracing scRNAseq, or LT-scSeq). The barcode may remain static or evolve over time.


However, the lineage data could be challenging to analyze.  These challenges include stochastic differentiation and variable expansion of clones; cells loss during analysis; barcode homoplasy wherein cells acquire the same barcode despite not having a lineage relationship; access to clones only at a single time point; and clonal dispersion due to a lag time between labeling cells and the first sampling (the lag time is necessary to allow the clone to grow large for resampling).


CoSpar, developed by `Wang et al. Nat. Biotech. (2022) <https://www.nature.com/articles/s41587-022-01209-1>`_, is among the first tools to perform dynamic inference by integrating state and lineage information. It solves for the transition probability map from cell states at an earlier time point to states at a later time point. It achieves accuracy and robustness by learning a sparse and coherent transition map, where neighboring initial states share similar yet sparse fate outcomes. Built upon the finite-time transition map, CoSpar can 1) infer fate potential of early states; 2) detect early fate bias (thus, fate boundary) among a heterogeneous progenitor population; 3) identify putative driver genes for fate bifurcation; 4) infer fate coupling or hierarchy; 5) visualize gene expression dynamics along an inferred differential trajectory. CoSpar also provides several methods to analyze clonal data by itself, including the clonal coupling between fate clusters and the bias of a clone towards a given fate, etc.  We envision CoSpar to be a platform to integrate key methods needed to analyze data with both state and lineage information.

.. image:: https://user-images.githubusercontent.com/4595786/113746452-56e4cb00-96d4-11eb-8278-0aac0469ba9d.png
   :width: 1000px
   :align: center
(copy right: Nature Biotechnology)

Coherent sparse optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One formalization of dynamic inference is to identify a transition map, a matrix :math:`T_{ij} (t_1,t_2)`, which describes the probability of a cell, initially in some state :math:`i` at time :math:`t_1`, giving rise to progeny in a state :math:`j` at time :math:`t_2`.  We define :math:`T_{ij} (t_1,t_2)` specifically as the fraction of progeny from state :math:`i` that occupy state :math:`j`. This transition matrix averages the effects of cell division, loss, and differentiation, but it nonetheless proves useful for several applications.


We make two reasonable assumptions about the nature of biological dynamics to constrain inference of the transition map. We assume the map to be a sparse matrix, since most cells can access just a few states during an experiment. And we assume the map to be locally coherent, meaning that cells in similar states should share similar fate outcomes. These constraints together force transition maps to be parsimonious and smooth, which we reasoned would make them robust to practical sources of noise in LT-scSeq experiments. As inputs, CoSpar requires a barcode-by-cell matrix :math:`I(t)`` that encodes the clonal information at time :math:`t`, and a data matrix for observed cell states (e.g. from scRNA-seq). Clonal data may have nested structure reflecting subclonal labeling.

CoSpar is formulated assuming that we have information on the same clones at more than one time point. More often, one might observe clones at only a later time point :math:`t_2`. For these cases inference is not fully constrained, one must learn both the transition map T and the initial clonal data :math:`I(t_1)`. We approximate a solution additionally constrained by a minimum global transport cost. We show that this approach is robust to initialization in tested datasets. Finally, coherence and sparsity provide reasonable constraints to the simpler problem of predicting dynamics from state heterogeneity alone without lineage data. We extended CoSpar to this case. Thus, CoSpar is flexible to different experimental designs, as summarized by the above figure.  Our core algorithms are illustrated below.


.. image:: https://user-images.githubusercontent.com/4595786/113746670-93b0c200-96d4-11eb-89c0-d1e7d72383e7.png
   :width: 1000px
   :align: center
(copy right: Nature Biotechnology)

Below, we formalize the coherent sparse optimization by which CoSpar infers the transition map.

In a model of stochastic differentiation, cells in a clone are distributed across states with a time-dependent  density vector :math:`\vec{P}(t)`. A transition map :math:`T` directly links clonal density profiles :math:`\vec{P}(t_{1,2})`  between time points:

.. math::
	\begin{equation}
	P_i(t_2 )= \sum_j P_j(t_1 )T_{ji}(t_1,t_2),   \quad \quad \quad \text{Eq. (1)}
	\end{equation}

From multiple clonal observations, our goal is to learn :math:`T`. To do so, we consider each observed cell transcriptome as a distinct state (:math:`\vec{P}(t)\in R^{N_t}`) for :math:`N_t`` cells profiled at time :math:`t``), and introduce :math:`S(t)\in R^{N_t\times N_t}` as a matrix of cell-cell similarity over all observed cell states, including those lacking clonal information. Denoting :math:`I(t)\in \{0,1\}^{M\times N_t}` as a clone-by-cell matrix of :math:`M` clonal barcodes, the density profiles of observed clones :math:`P(t)\in R^{M\times N_t}` are estimated as :math:`P(t)\approx I(t)S(t)`. In matrix form, the constraint in Eq. (1) from all observed clones then becomes :math:`P(t_2)\approx P(t_1)T(t_1,t_2)`.


Since the matrices :math:`P(t_{1,2})` are determined directly from data, with enough information :math:`T(t_1,t_2)` could be learnt by matrix inversion. However, in most cases, the number of clones is far less than the number of states. To constrain the map, we require that: 1)  :math:`T` is a sparse matrix; 2)  :math:`T` is locally coherent; and 3) :math:`T` is a non-negative matrix. With these requirements, the inference becomes an optimization problem:

.. math::
	\begin{equation}
	 \min_{T} ||T||_1+\alpha ||LT||_2,  \; \text{s.t.} \; ||P(t_2)- P(t_1) T(t_1,t_2)||_{2}\le\epsilon;\; T\ge 0; \text{Normalization}.
	 \end{equation}

Here, :math:`‖T‖_1` quantifies the sparsity of the matrix T through its l-1 norm, while  :math:`‖LT‖_2` quantifies the local coherence of :math:`T` (:math:`L` is the Laplacian of the cell state similarity graph, and :math:`LT` is the local divergence). The remaining constraints enforce the observed clonal dynamics, non-negativity of :math:`T`, and map normalization, respectively. At :math:`\alpha=0`, the minimization takes the form of Lasso, an algorithm for compressed sensing. Our formulation extends compressed sensing from vectors to matrices, and to enforce local coherence. The local coherence extension is reminiscent of the fused Lasso problem.
An iterative, heuristic approach solves the CoSpar optimization efficiently, replacing :math:`(\alpha,\epsilon)` with parameters that explicitly control coherence and sparsity. See `Wang et al. Nat. Biotech. (2022) <https://www.nature.com/articles/s41587-022-01209-1>`_ for a detailed exposition of the method and its implementation.
