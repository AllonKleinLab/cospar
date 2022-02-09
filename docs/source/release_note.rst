Release notes
-------------

v0.2.1
''''''

Major changes from v0.1.8 to v0.2.1:
    - Split each plotting function into two parts: computing the results (stored at cospar.tl.**) and actually plotting the result (stored at cospar.pl.**).
    - Update the notebooks to accomodate these changes.
    - Update the datasets in the cloud to add more annotations.
    - Re-organize the content of the plot, tool, and tmap modules.
    - Fix stochasticity when running HighVar method to generate the initalized map.
    - Fix generating X_clone from the cell_id-by-barcode_id list.
    - Add a few more functions: :func:`cospar.pl.clonal_fates_across_time`, :func:`cospar.pl.clonal_reports`,  :func:`cospar.pl.embedding_genes`, :func:`cospar.tl.fate_biased_clones`
    - Update :func:`cospar.pl.barcode_heatmap` for better clonal ordering
    - Fix the docs.
    - Adopt "Raise ValueError" method for error handling.
    - Unify error checking at the beginning of several functions.

v0.1.8
''''''

The version of code used in running the notebooks that generate figures for the published paper. To run the original notebooks, you should switch to this version.
