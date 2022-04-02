import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

from matplotlib import pyplot as plt

from tests.context import cospar as cs

# be careful not to change this global parameter
selected_fates = [
    "Ccr7_DC",
    "Mast",
    "Meg",
    "pDC",
    "Eos",
    "Lymphoid",
    "Erythroid",
    "Baso",
    "Neutrophil",
    "Monocyte",
]


def config(shared_datadir):
    cs.settings.data_path = os.path.join(shared_datadir, "..", "output")
    cs.settings.figure_path = os.path.join(shared_datadir, "..", "output")
    cs.settings.verbosity = 0  # range: 0 (error),1 (warning),2 (info),3 (hint).
    cs.settings.set_figure_params(
        format="png", figsize=[4, 3.5], dpi=25, fontsize=14, pointsize=3, dpi_save=25
    )
    cs.hf.set_up_folders()  # setup the data_path and figure_path


def test_load_dataset(shared_datadir):
    config(shared_datadir)
    print("-------------------------load dataset")
    # cs.datasets.hematopoiesis_subsampled()
    # cs.datasets.hematopoiesis()
    # cs.datasets.hematopoiesis_130K()
    # cs.datasets.hematopoiesis_Gata1_states()
    # cs.datasets.reprogramming()
    # cs.datasets.lung()
    cs.datasets.synthetic_bifurcation()
    # cs.datasets.reprogramming_Day0_3_28()


def test_load_data_from_scratch(shared_datadir):
    import numpy as np
    import pandas as pd
    import scipy.io as sio

    config(shared_datadir)
    df_cell_id = pd.read_csv(os.path.join(shared_datadir, "cell_id.txt"))
    file_name = os.path.join(shared_datadir, "test_adata_preprocessed.h5ad")
    adata_orig = cs.hf.read(file_name)
    adata_orig = cs.pp.initialize_adata_object(
        adata_orig,
        cell_names=df_cell_id["Cell_ID"],
    )
    df_X_clone = pd.read_csv(
        os.path.join(shared_datadir, "clonal_data_in_table_format.txt")
    )
    cs.pp.get_X_clone(adata_orig, df_X_clone["Cell_ID"], df_X_clone["Clone_ID"])
    print(adata_orig.obsm["X_clone"].shape)
    # cs.pl.embedding(adata_orig, color="state_info")


def test_preprocessing(shared_datadir):
    config(shared_datadir)
    file_name = os.path.join(shared_datadir, "test_adata_preprocessed.h5ad")
    adata_orig_0 = cs.hf.read(file_name)
    print("------------------------Test preprocessing")
    data_des = "test"
    # This is just a name to indicate this data for saving results. Can be arbitrary but should be unique to this data.
    X_state = adata_orig_0.X  # np.array or sparse matrix, shape (n_cell, n_gene)
    gene_names = adata_orig_0.var_names  # List of gene names, shape (n_genes,)
    # Clonal data matrix, np.array or sparse matrix, shape: (n_cell, n_clone)
    X_clone = adata_orig_0.obsm["X_clone"]
    # 2-d embedding, np.array, shape: (n_cell, 2)
    X_emb = adata_orig_0.obsm["X_emb"]
    # A vector of cluster id for each cell, np.array, shape: (n_cell,),
    state_info = adata_orig_0.obs["state_info"]
    # principle component matrix, np.array, shape: (n_cell, n_pcs)
    X_pca = adata_orig_0.obsm["X_pca"]
    # A vector of time info, np.array of string, shape: (n_cell,)
    time_info = adata_orig_0.obs["time_info"]

    print("------------initialize_adata_object")
    adata_orig = cs.pp.initialize_adata_object(
        X_state=X_state,
        gene_names=gene_names,
        time_info=time_info,
        X_clone=X_clone,
        data_des=data_des,
    )

    adata_orig = cs.pp.initialize_adata_object(adata=adata_orig_0, X_clone=X_clone)

    print("------------get_highly_variable_genes")
    cs.pp.get_highly_variable_genes(
        adata_orig,
        normalized_counts_per_cell=10000,
        min_counts=3,
        min_cells=3,
        min_gene_vscore_pctl=90,
    )

    print("------------remove_cell_cycle_correlated_genes")
    cs.pp.remove_cell_cycle_correlated_genes(
        adata_orig,
        cycling_gene_list=["Ube2c"],
    )

    print("------------get_X_pca")
    cs.pp.get_X_pca(adata_orig, n_pca_comp=40)

    print("------------get_X_emb")
    cs.pp.get_X_emb(adata_orig, n_neighbors=20, umap_min_dist=0.3)

    print("------------get_state_info (this modifies the state info. Need to reload")
    cs.pp.get_state_info(adata_orig, n_neighbors=20, resolution=0.5)

    plt.close("all")


def test_clonal_analysis(shared_datadir):
    config(shared_datadir)

    file_name = os.path.join(shared_datadir, "test_adata_preprocessed.h5ad")
    adata = cs.hf.read(file_name)
    print("------------------------------Basic clonal analysis")
    print("----------barcode_heatmap")
    selected_times = None

    cs.pl.barcode_heatmap(adata, log_transform=True, selected_fates=selected_fates)
    plt.close("all")

    print("----------fate_coupling_from_clones")

    cs.tl.fate_coupling(adata, source="X_clone")
    cs.pl.fate_coupling(adata, source="X_clone")

    print("----------fate_hierarchy_from_clones")
    cs.tl.fate_hierarchy(adata, source="X_clone")
    cs.pl.fate_hierarchy(adata, source="X_clone")
    plt.close("all")

    print("----------clonal_fate_bias")
    cs.tl.clonal_fate_bias(adata, selected_fate="Neutrophil")
    cs.pl.clonal_fate_bias(adata)
    plt.close("all")

    print("----------clones_on_manifold")
    cs.pl.clones_on_manifold(adata, selected_clone_list=[1, 2, 3])
    plt.close("all")


def test_Tmap_inference(shared_datadir):
    config(shared_datadir)
    file_name = os.path.join(shared_datadir, "test_adata_preprocessed.h5ad")
    adata_orig = cs.hf.read(file_name)
    print("------------------------------T map inference")

    print("---------infer_Tmap_from_one_time_clones")
    adata_1 = cs.tmap.infer_Tmap_from_one_time_clones(
        adata_orig,
        initial_time_points=["2"],
        later_time_point="4",
        initialize_method="OT",
        OT_cost="GED",
        smooth_array=[5, 5, 5],
        sparsity_threshold=0.1,
    )

    print("---------infer_Tmap_from_state_info_alone")
    adata_2 = cs.tmap.infer_Tmap_from_state_info_alone(
        adata_orig,
        initial_time_points=["4"],
        later_time_point="6",
        initialize_method="HighVar",
        HighVar_gene_pctl=85,
        max_iter_N=[10, 10],
        epsilon_converge=[0.01, 0.01],
        smooth_array=[5, 5, 5],
        sparsity_threshold=0.1,
    )

    print("---------infer_Tmap_from_clonal_info_alone")
    adata_3 = cs.tmap.infer_Tmap_from_clonal_info_alone(
        adata_orig,
        method="weinreb",
        later_time_point="6",
        selected_fates=selected_fates,
    )

    print("-------------------------save maps")
    # cs.hf.save_map(adata_3)


def test_Tmap_analysis(shared_datadir):
    config(shared_datadir)

    load_pre_compute_map = False
    if load_pre_compute_map:
        # this is for fast local testing
        file_name = os.path.join(
            cs.settings.data_path,
            "test_MultiTimeClone_Later_FullSpace0_t*2*4*6_adata_with_transition_map.h5ad",
        )
        adata = cs.hf.read(file_name)

        # adata = cs.hf.read(
        #     "/Users/shouwenwang/Dropbox (HMS)/Python/CoSpar/docs/source/data_cospar/LARRY_sp500_ranking1_MultiTimeClone_Later_FullSpace0_t*2*4*6_adata_with_transition_map.h5ad"
        # )
    else:
        file_name = os.path.join(shared_datadir, "test_adata_preprocessed.h5ad")
        adata_orig = cs.hf.read(file_name)
        print("---------infer_Tmap_from_multitime_clones")
        adata = cs.tmap.infer_Tmap_from_multitime_clones(
            adata_orig,
            clonal_time_points=["2", "4"],
            later_time_point="6",
            smooth_array=[5, 5, 5],
            sparsity_threshold=0.1,
            intraclone_threshold=0.2,
            max_iter_N=5,
            epsilon_converge=0.01,
        )

    X_clone = adata.obsm["X_clone"]
    print(type(X_clone))

    selected_fates = [
        "Ccr7_DC",
        "Mast",
        "Meg",
        "pDC",
        "Eos",
        "Baso",
        "Lymphoid",
        "Erythroid",
        "Neutrophil",
        "Monocyte",
    ]

    cs.tl.fate_coupling(adata, source="transition_map")
    cs.pl.fate_coupling(adata, source="transition_map")

    cs.tl.fate_hierarchy(adata, source="transition_map")
    cs.pl.fate_hierarchy(adata, source="transition_map")

    selected_fates = [
        "Neutrophil",
        "Monocyte",
    ]
    cs.tl.fate_map(adata, source="transition_map", selected_fates=selected_fates)
    cs.pl.fate_map(
        adata,
        source="transition_map",
        selected_fates=selected_fates,
        show_histogram=True,
        selected_times="4",
    )

    cs.tl.fate_potency(
        adata, source="transition_map", selected_fates=selected_fates, fate_count=True
    )
    cs.pl.fate_potency(
        adata,
        source="transition_map",
        show_histogram=True,
        selected_times="4",
    )

    selected_fates = [
        "Neutrophil",
        "Monocyte",
    ]
    cs.tl.fate_bias(
        adata,
        source="transition_map",
        selected_fates=selected_fates,
        sum_fate_prob_thresh=0.01,
    )
    cs.pl.fate_bias(
        adata,
        source="transition_map",
        show_histogram=True,
        selected_times="4",
    )
    cs.pl.fate_bias(
        adata,
        source="transition_map",
        show_histogram=True,
        selected_fates=selected_fates,
        selected_times="4",
    )

    selected_fates = [
        "Neutrophil",
        "Monocyte",
    ]
    cs.tl.progenitor(
        adata,
        source="transition_map",
        selected_fates=selected_fates,
        sum_fate_prob_thresh=0.01,
        avoid_target_states=True,
    )
    cs.pl.progenitor(adata, source="transition_map", selected_times="4")

    cs.tl.iterative_differentiation(
        adata,
        source="transition_map",
        selected_fates="Neutrophil",
        apply_time_constaint=False,
    )
    cs.pl.iterative_differentiation(
        adata,
        source="transition_map",
    )

    cs.pl.gene_expression_dynamics(
        adata, selected_fate="Neutrophil", gene_name_list=["Gata1"]
    )

    gene_list = [
        "Mpo",
        "Elane",
        "Gstm1",
        "Mt1",
        "S100a8",
        "Prtn3",
        "Gfi1",
        "Dstn",
        "Cd63",
        "Ap3s1",
        "H2-Aa",
        "H2-Eb1",
        "Ighm",
    ]

    selected_fates = [
        "Neutrophil",
        "Monocyte",
        ["Baso", "Eos", "Erythroid", "Mast", "Meg"],
        ["pDC", "Ccr7_DC", "Lymphoid"],
    ]
    renames = ["Neu", "Mon", "Meg-Ery-MBaE", "Lym-Dc"]

    cs.pl.gene_expression_heatmap(
        adata,
        selected_genes=gene_list,
        selected_fates=selected_fates,
        rename_fates=renames,
        fig_width=12,
    )

    cs.pl.gene_expression_on_manifold(
        adata, selected_genes=["Gata1", "Elane"], savefig=True
    )

    df1, df2 = cs.tl.differential_genes(
        adata, cell_group_A="Neutrophil", cell_group_B="Monocyte"
    )
    import numpy as np

    state_info = np.array(adata.obs["state_info"])
    df1, df2 = cs.tl.differential_genes(
        adata,
        cell_group_A=(state_info == "Neutrophil"),
        cell_group_B=(state_info == "Monocyte"),
    )
    print(df1)

    cs.pl.single_cell_transition(
        adata, selected_state_id_list=[1, 2], savefig=True, map_backward=False
    )


def test_simulated_data():
    print("---------- bifurcation model ------------")
    L = 10
    adata = cs.simulate.bifurcation_model(t1=2, M=20, L=L)
    adata = cs.tmap.infer_Tmap_from_multitime_clones(
        adata, smooth_array=[10, 10, 10], compute_new=True
    )
    Tmap = adata.uns["transition_map"]
    state_info = adata.obs["state_info"]
    cell_id_t1 = adata.uns["Tmap_cell_id_t1"]
    cell_id_t2 = adata.uns["Tmap_cell_id_t2"]
    correlation_cospar = (
        cs.simulate.quantify_correlation_with_ground_truth_fate_bias_BifurcationModel(
            Tmap, state_info, cell_id_t1, cell_id_t2
        )
    )
    print(
        f"Fate bias correlation from the predicted transition map: {correlation_cospar:.3f}"
    )

    print("---------------Linear differentiation---------------")
    adata = cs.simulate.linear_differentiation_model(
        Nt1=50, progeny_N=1, used_clone_N=10, always_simulate_data=True
    )
    adata = cs.tmap.infer_Tmap_from_multitime_clones(
        adata, smooth_array=[10, 10, 10], compute_new=True
    )
    Tmap = adata.uns["transition_map"]
    state_info = adata.obs["state_info"]
    cell_id_t1 = adata.uns["Tmap_cell_id_t1"]
    cell_id_t2 = adata.uns["Tmap_cell_id_t2"]

    X_t1 = adata.obsm["X_orig"][cell_id_t1]
    X_t2 = adata.obsm["X_orig"][cell_id_t2]
    TPR_cospar = cs.simulate.quantify_transition_peak_TPR_LinearDifferentiation(
        Tmap, X_t1, X_t2
    )
    print(f"True positive rate for the predicted transition map: {TPR_cospar:.3f}")


def test_clean_up():
    print("---------Clean up")
    if Path(cs.settings.data_path).is_dir():
        os.system("rm -r output")


# os.chdir(os.path.dirname(__file__))
# cs.settings.verbosity = 3  # range: 0 (error),1 (warning),2 (info),3 (hint).
# # test_load_dataset("data")
# # test_preprocessing("data")
# # test_load_data_from_scratch("data")
# # test_clonal_analysis("data")
# # test_Tmap_inference("data")
# test_Tmap_analysis("data")
