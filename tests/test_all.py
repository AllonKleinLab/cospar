import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from matplotlib import pyplot as plt
from pathlib import Path
import cospar as cs

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


def config():
    print("data_path:", cs.settings.data_path)
    test_path = os.path.dirname(__file__)
    os.chdir(test_path)
    cs.settings.data_path = os.path.join(test_path, "output")
    cs.settings.figure_path = os.path.join(test_path, "output")
    cs.settings.verbosity = 0  # range: 0 (error),1 (warning),2 (info),3 (hint).
    cs.settings.set_figure_params(
        format="png", figsize=[4, 3.5], dpi=25, fontsize=14, pointsize=3, dpi_save=25
    )
    cs.hf.set_up_folders()  # setup the data_path and figure_path


def test_load_dataset():
    config()
    print("-------------------------load dataset")
    # cs.datasets.hematopoiesis_subsampled()
    # cs.datasets.hematopoiesis()
    # cs.datasets.hematopoiesis_130K()
    # cs.datasets.hematopoiesis_Gata1_states()
    # cs.datasets.reprogramming()
    # cs.datasets.lung()
    cs.datasets.synthetic_bifurcation()
    # cs.datasets.reprogramming_Day0_3_28()


def test_preprocessing():
    config()
    file_name = os.path.join(
        os.path.dirname(__file__), "data", "test_adata_preprocessed.h5ad"
    )
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


def test_clonal_analysis():
    config()

    file_name = os.path.join(
        os.path.dirname(__file__), "data", "test_adata_preprocessed.h5ad"
    )
    adata_orig = cs.hf.read(file_name)
    print("------------------------------Basic clonal analysis")
    print("----------barcode_heatmap")
    selected_times = None

    cs.pl.barcode_heatmap(
        adata_orig,
        selected_times=selected_times,
        selected_fates=selected_fates,
        color_bar=True,
        log_transform=False,
    )
    plt.close("all")

    print("----------fate_coupling_from_clones")
    selected_times = None
    cs.pl.fate_coupling_from_clones(
        adata_orig,
        selected_times=selected_times,
        selected_fates=selected_fates,
        color_bar=True,
        method="Weinreb",
    )
    plt.close("all")

    print("----------fate_hierarchy_from_clones")
    cs.pl.fate_hierarchy_from_clones(
        adata_orig,
        selected_times=selected_times,
        selected_fates=selected_fates,
        plot_history=True,
    )
    plt.close("all")

    print("----------clonal_fate_bias")
    result = cs.pl.clonal_fate_bias(
        adata_orig, selected_fate="Monocyte", alternative="two-sided"
    )
    plt.close("all")

    print("----------clones_on_manifold")
    ids = result["Clone ID"][:2]
    cs.pl.clones_on_manifold(
        adata_orig,
        selected_clone_list=ids,
        color_list=["black", "red", "blue"],
        clone_point_size=10,
    )
    plt.close("all")


def test_Tmap_inference():
    config()
    file_name = os.path.join(
        os.path.dirname(__file__), "data", "test_adata_preprocessed.h5ad"
    )
    adata_orig = cs.hf.read(file_name)
    print("------------------------------T map inference")

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

    print("---------infer_Tmap_from_one_time_clones")
    adata_1 = cs.tmap.infer_Tmap_from_one_time_clones(
        adata_orig,
        initial_time_points=["4"],
        later_time_point="6",
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
    cs.hf.save_map(adata)


def test_Tmap_plotting():
    config()
    file_name = os.path.join(
        os.path.dirname(__file__), "data", "test_adata_preprocessed.h5ad"
    )
    adata_orig = cs.hf.read(file_name)
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

    print("-------------------------plotting")

    selected_state_id_list = [1, 10]
    map_backward = False

    print("---------single_cell_transition")
    cs.pl.single_cell_transition(
        adata,
        selected_state_id_list=selected_state_id_list,
        used_Tmap="transition_map",
        map_backward=map_backward,
    )
    plt.close("all")

    print("---------fate map")
    cs.pl.fate_map(
        adata,
        selected_fates=["Neutrophil", "Monocyte"],
        used_Tmap="transition_map",
        map_backward=True,
        plot_target_state=True,
        horizontal=True,
    )
    plt.close("all")

    print("---------fate entropy")
    cs.pl.fate_potency(
        adata,
        used_Tmap="transition_map",
        map_backward=True,
        method="norm-sum",
        color_bar=True,
        fate_count=True,
    )
    plt.close("all")

    print("---------fate bias")
    cs.pl.fate_bias(
        adata,
        selected_fates=["Neutrophil", "Monocyte"],
        used_Tmap="transition_map",
        pseudo_count=0,
        plot_target_state=False,
        map_backward=True,
        sum_fate_prob_thresh=0.002,
        method="norm-sum",
    )
    plt.close("all")

    print("---------dynamic_trajectory_from_fate_bias")
    cs.pl.dynamic_trajectory_from_fate_bias(
        adata,
        selected_fates=["Neutrophil", "Monocyte"],
        used_Tmap="transition_map",
        map_backward=True,
        bias_threshold_A=0.5,
        bias_threshold_B=0.5,
        sum_fate_prob_thresh=0.2,
        avoid_target_states=True,
    )
    plt.close("all")

    print("---------DGE analysis")
    dge_gene_A, dge_gene_B = cs.pl.differential_genes(adata, plot_gene_N=0)
    plt.close("all")

    print("---------gene expression on manifold")
    selected_genes = dge_gene_A["gene"][:2]
    cs.pl.gene_expression_on_manifold(
        adata, selected_genes=selected_genes, color_bar=True, savefig=False
    )
    plt.close("all")

    print("---------gene expression heatmap")
    gene_list = list(dge_gene_A["gene"][:20]) + list(
        dge_gene_B["gene"][:20]
    )  # select the top 20 genes from both populations

    cs.pl.gene_expression_heat_map(
        adata,
        selected_genes=gene_list,
        selected_fates=[
            "Neutrophil",
            "Monocyte",
            ["Baso", "Eos", "Erythroid", "Mast", "Meg"],
            ["pDC", "Ccr7_DC", "Lymphoid"],
        ],
        rename_fates=["Neu", "Mon", "Meg-Ery-MBaE", "Lym-Dc"],
        fig_width=12,
    )
    plt.close("all")

    print("---------gene expression dynamics")
    cs.pl.gene_expression_dynamics(
        adata,
        selected_fate="Neutrophil",
        gene_name_list=["Gata1", "Mpo", "Elane", "S100a8"],
        traj_threshold=0.2,
        invert_PseudoTime=False,
        compute_new=True,
        gene_exp_percentile=99,
        n_neighbors=8,
        plot_raw_data=False,
    )
    plt.close("all")

    print("---------Fate coupling from Tmap")
    cs.pl.fate_coupling_from_Tmap(
        adata,
        selected_fates=selected_fates,
        used_Tmap="transition_map",
    )
    plt.close("all")

    print("---------Fate hierachy from Tmap")
    cs.pl.fate_hierarchy_from_Tmap(
        adata,
        selected_fates=selected_fates,
        used_Tmap="transition_map",
        rename_fates=selected_fates,
    )
    plt.close("all")

    # # For some reason, these two functions are extremely slow to test (they generate figures and stop the process)
    # print("---------Refine state info from marker genes")
    # confirm_change = False
    # marker_genes = ["Mpo", "Elane", "S100a8"]
    # cs.pp.refine_state_info_by_marker_genes(
    #     adata,
    #     marker_genes,
    #     express_threshold=0.1,
    #     selected_times=["4"],
    #     new_cluster_name="new",
    #     add_neighbor_N=10,
    #     confirm_change=confirm_change,
    # )
    # plt.close("all")

    # print("---------Refine state info by leiden clustering")
    # confirm_change = False
    # cs.pp.refine_state_info_by_leiden_clustering(
    #     adata,
    #     selected_times=["2"],
    #     n_neighbors=20,
    #     resolution=0.5,
    #     confirm_change=confirm_change,
    # )
    # plt.close("all")

    print("---------Differential genes for given fates")
    cs.pl.differential_genes_for_given_fates(
        adata, selected_fates=["Neutrophil", "Monocyte"], plot_gene_N=2
    )
    plt.close("all")

    print("---------Dynamic trajectory via iterative mapping")
    cs.pl.dynamic_trajectory_via_iterative_mapping(
        adata,
        selected_fate="Neutrophil",
        plot_separately=True,
        used_Tmap="intraclone_transition_map",
    )
    plt.close("all")


def test_clean_up():
    print("---------Clean up")
    if Path(cs.settings.data_path).is_dir():
        # os.system(
        #     "mkdir -p temp; mv data/test_adata_preprocessed.h5ad temp/test_adata_preprocessed.h5ad; rm -r data; rm -r figure; mv temp data"
        # )
        os.system("rm -r output")


# test_load_dataset()
# test_preprocessing()
# test_clonal_analysis()
# test_Tmap_inference()
# test_Tmap_plotting()
