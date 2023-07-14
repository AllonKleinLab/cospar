Release notes
-------------

v0.3.1
''''''
- Update cospar.tl.fate_coupling. The coupling score now account for sampling heterogeneity between cell types (so that each row adds to 1). Besides, each clone represents an independent measurement and therefore should contribute equally to the coupling calculation, we then normalize the matrix per column to calculate within each clone the fraction of cells in each cell type (clone normalization).

- Add cospar.tl.pvalue_for_fate_coupling


v0.2.1
''''''

Major changes from v0.1.8 to v0.2.1:
    - Split each plotting function into two parts: computing the results (stored at cospar.tl.**) and actually plotting the result (stored at cospar.pl.**).
    - Update the notebooks to accomodate these changes.
    - Update the datasets in the cloud to add more annotations.
    - Re-organize the content of the plot, tool, and tmap modules.
    - Fix stochasticity when running HighVar method to generate the initialized map.
    - Fix generating X_clone from the cell_id-by-barcode_id list.
    - Add a few more functions: :func:`cospar.pl.clonal_fates_across_time`, :func:`cospar.pl.clonal_reports`,  :func:`cospar.pl.embedding_genes`, :func:`cospar.tl.fate_biased_clones`
    - Update :func:`cospar.pl.barcode_heatmap` to order clones in a better way
    - Fix the docs.
    - Adopt "Raise ValueError" method for error handling.
    - Unify error checking at the beginning of several functions.

v0.1.8
''''''

This is used in running the notebooks that generate figures for the published paper. To run the original notebooks, you should switch to this version.
