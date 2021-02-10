About CoSpar
------------

High-throughput single-cell measurements have enabled unbiased study of development and differentiation, leading to numerous methods for dynamic inference. However, single-cell RNA sequencing (scRNA-seq) data alone does not fully constrain the differentiation dynamics, and existing methods inevitably operate under simplified assumptions. In parallel, the lineage information of individual cells can be profiled simultaneously along with their transcriptome by using a heritable and expressible DNA barcode as a lineage tracer. The barcode may remain static or evolve. However, sequencing is killing; we cannot get the actual single-cell dynamics in the transcriptomic space. In most cases, we can only get a snapshot of cells' lineage information at a single time point. In some *in vitro* systems like hematopoiesis or iPS, we can approximate the dynamics by re-sampling the same clone, i.e., cells from the same ancestor, over time. The approximation is inevitably crude due to the tradeoff between the wish to observe a clone earlier and the need to allow it to expand before sampling. Therefore, clonal data also provide only partial information of the dynamics. This opens up the opportunity to integrate state and lineage (clonal) information for dynamic inference. 

CoSpar, developed by `Wang & Klein (2021) <https://doi.org/>`_, is the first tool ever to perform dynamic inference by integrating state and lineage information. It solves for the transition probability map from cell states at an earlier time point to states at a later time point. It achieves accuracy and robustness by making use of intrinsic sparsity and coherence of the transition dynamics: neighboring initial states share similar yet sparse fate outcomes. Built upon the finite-time transition map, CoSpar can 1) infer fate potential of early states; 2) detect early fate bias among a heterogeneous progenitor population, and identify the early boundary of fate bifurcation; 3) identify putative driver genes for fate bifurcation; 4) by selecting progenitor states more accurately, we also provide a better pseudo time analysis, enabling a more accurate picture of gene expression dynamics during differentiation. CoSpar also provides several methods to analyze clonal data by itself, including the clonal coupling between fate clusters and clonal fate bias.  We envision CoSpar to be a platform to integrate key methods needed to analyze data with both state and lineage information. 



Coherent sparsity optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
One formulation of dynamic inference is to identify a finite-time transition matrix :math:`T_{ij} (t_1,t_2)`, which describes the probability of a cell, initially in some state :math:`i` at time :math:`t_1`, giving rise to progeny in a state :math:`j` at time :math:`t_2`. However, scRNA-seq data alone do not constrain these maps fully, and strong assumptions have to be made.  We now extend these ideas to incorporate information from lineage tracing.

The clonal data directly constrains the transition map. We denote *I(t)* as a clone-by-cell matrix that encodes the clonal information at time :math:`t`, and introduce  :math:`S`  as a cell-state similarity matrix that encodes the state information. The observed clonal data is sampled from a particular realization of the stochastic differentiation dynamics. To account for this bias and technical noises, we locally smooth the raw data to obtain the “average” cell density profile per clone:  :math:`P(t)=I(t)S`.  The transition map :math:`T` directly links density profiles at two time points: 

.. math::
	\begin{equation}
	P(t_2 )\approx P(t_1 )T(t_1,t_2)
	\end{equation}

In most cases, the number of clones (i.e., constraints) is less than that of initial states (i.e., variables), and  :math:`T` is not sufficiently constrained.To further constrain the map, we observe that: 1)  :math:`T` is a sparse matrix, since most cell states have sparse differentiation outcomes; 2)  :math:`T` is locally coherent as neighboring cell states share similar fate outcomes; 3) :math:`T` is a non-negative matrix. With these, the inference becomes an optimization problem:

.. math::
	\begin{equation}
	 \min_{T} ||T||_1+\alpha ||LT||_2,  \; \text{s.t.} \; ||P(t_2)- P(t_1) T(t_1,t_2)||_{2}\le\epsilon;\; T\ge 0; \sum_j T_{ij}=1.
	 \end{equation}

Here, :math:`‖T‖_1` quantifies the sparsity of the matrix T through its l-1 norm, while  :math:`‖LT‖_2` quantifies the local coherence of :math:`T` (:math:`L` is the Laplacian of the cell state similarity graph, and :math:`LT` is the local divergence). The remaining constraints are from clonal observation, non-negativity of :math:`T`, and map normalization, respectively. Both :math:`\alpha` and :math:`\epsilon` are tunable parameters.  At :math:`\alpha=0`, the minimization takes the form of Lasso, a traditional algorithm for compressed sensing. Our formulation extends compressed sensing from vector-oriented to matrix-oriented, and improves its robustness by incorporating the local coherence constraint. The local coherence extension is reminiscent of the fused Lasso problem. We have developed CoSpar to solve the optimization. 
	
The above optimization is formulated as if we have initial clonal information by re-sampling clones. When we only have the clonal information at :math:`t_2`, we can still infer the transition map by jointly optimizing the map :math:`T` and the initial clonal data :math:`I(t_1)` such that the cost function is minimized. In this joint optimization, :math:`I(t_1 )` is further constrained such that all initial cell states are labeled by clones, and non-overlapping clones at :math:`t_2` label different cells at :math:`t_1`. 


See `Wang & Klein (2021) <https://doi.org/>`_ for a detailed exposition of the methods.
