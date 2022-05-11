import os
import time

import numpy as np
import scanpy as sc
import scipy.sparse as ssp
from matplotlib import pyplot as plt
from tqdm import tqdm

from . import hf, settings, tl


def sigma(x, diff_sigma):
    #    return 0.1*np.sin(0.1*x)+0.1
    return diff_sigma


# growth is not used below
def growth(x):
    #    return 0.1*np.cos(0.2*x)+1
    return 1


def progression(x, dL):
    #    return 0.1*np.sin(0.1*x)+1
    return dL


def kernel(x, y_vec, smooth_sigma):
    temp = []
    for y in y_vec:
        d = abs(x - y)

        temp.append(np.exp(-((d) ** 2) / (2 * (smooth_sigma**2))))

    norm_Prob = np.array(temp) / np.sum(temp)

    return norm_Prob


def kernel_matrix(x_vec, smooth_sigma):
    matrix = np.zeros((len(x_vec), len(x_vec)))
    for j, x in enumerate(x_vec):
        matrix[j] = kernel(x, x_vec, smooth_sigma)

    return matrix


def kernel_matrix_v1(smooth_sigma, shortest_distance):
    shortest_distance = np.array(shortest_distance)
    matrix_temp = np.exp(-((shortest_distance) ** 2) / (2 * (smooth_sigma**2)))
    norm_kernel = np.zeros(matrix_temp.shape)
    for j in range(matrix_temp.shape[0]):
        norm_kernel[j] = matrix_temp[j] / np.sum(matrix_temp[j])

    return norm_kernel


def shortest_path_distance(x_vec, bif_x=0, mode="1d"):
    N = len(x_vec)
    D = np.zeros((N, N))  # distance matrix

    for i, x in enumerate(x_vec):
        for j, y in enumerate(x_vec):
            if mode == "1d":
                d = abs(x - y)
            else:
                # this is the bifurcation model
                if x[1] == y[1]:  # belonging to the same branch
                    d = abs(x[0] - y[0])
                else:
                    d = abs(x[0] - bif_x) + abs(y[0] - bif_x)

            D[i, j] = d

    return D


def transition_prob(x, y_vec, diff_sigma, dL):
    temp = []
    for y in y_vec:
        temp.append(
            np.exp(
                -((x - y + progression(x, dL)) ** 2) / (2 * (sigma(x, diff_sigma) ** 2))
            )
        )

    norm_Prob = np.array(temp) / np.sum(temp)

    return norm_Prob


def simulate_cell_position_next_time_point(x_t0, parameter, mode="1d"):
    L = parameter["L"]
    dL = parameter["dL"]
    diff_sigma = parameter["diff_sigma"]
    progeny_N = parameter["progeny_N"]
    dx = parameter["dx"]
    lattice = parameter["lattice"]
    bifurcation = parameter["bifurcation"]

    x_next = []
    id_next = []

    if mode == "1d":
        # 1 d problem
        for j, x in enumerate(x_t0):
            ## this method is much faster
            if x < L - (1.5 * progression(x, dL) + 5 * sigma(x, diff_sigma)):
                bb = np.random.normal(
                    x + progression(x, dL), sigma(x, diff_sigma), progeny_N
                )
                x_next = x_next + list(bb)
                id_next = id_next + list(np.floor(bb / dx).astype(int))
            else:
                ## this method is much slower
                for k in range(progeny_N):
                    cum_Prob = np.cumsum(transition_prob(x, lattice, diff_sigma, dL))
                    # pick the first event that is larger than the random variable
                    new_id = np.nonzero(cum_Prob > np.random.rand())[0][0].astype(int)
                    id_next.append(new_id)
                    x_next.append(lattice[new_id])
    else:

        for j, X in enumerate(x_t0):
            x = X[0]
            ## this method is much faster
            if x < L - (1.5 * progression(x, dL) + 5 * sigma(x, diff_sigma)):
                bb = np.random.normal(
                    x + progression(x, dL), sigma(x, diff_sigma), progeny_N
                )
                id_next = id_next + list(np.floor(bb / dx).astype(int))

            else:
                ## this method is much slower
                bb = []
                for k in range(progeny_N):
                    cum_Prob = np.cumsum(transition_prob(x, lattice, diff_sigma, dL))
                    # pick the first event that is larger than the random variable
                    new_id = np.nonzero(cum_Prob > np.random.rand())[0][0].astype(int)
                    id_next.append(new_id)
                    bb.append(lattice[new_id])

            ## update cell state
            if x > bifurcation:
                x_next = x_next + [[y, X[1]] for y in bb]
            else:
                temp_next = []
                for y in bb:
                    if y < bifurcation:
                        temp_next.append([y, -1])
                    else:
                        fate = np.random.randint(2)
                        temp_next.append([y, fate])
                x_next = x_next + temp_next

    return x_next, id_next


def simulate_cell_position_next_time_point_continuous_barcoding(
    x_t0, parameter, mode="1d", barcoding_rate=0.1
):
    L = parameter["L"]
    dL = parameter["dL"]
    diff_sigma = parameter["diff_sigma"]
    progeny_N = parameter["progeny_N"]
    dx = parameter["dx"]
    lattice = parameter["lattice"]
    bifurcation = parameter["bifurcation"]

    x_next = []
    id_next = []

    if mode == "1d":
        # 1 d problem
        for j, X in enumerate(x_t0):  #
            ## this method is much faster
            x = X[0]  # X[0], location; X[1], barcode information
            if x < L - (1.5 * progression(x, dL) + 5 * sigma(x, diff_sigma)):
                bb = np.random.normal(
                    x + progression(x, dL), sigma(x, diff_sigma), progeny_N
                )
                x_next_temp = []
                for y in bb:
                    barcode_list = X[1]
                    if np.random.rand() < barcoding_rate:
                        barcode_list.append(np.random.randint(1000000))

                    x_next_temp.append([y, barcode_list])
                x_next = x_next + x_next_temp
                id_next = id_next + list(np.floor(bb / dx).astype(int))
            else:
                ## this method is much slower
                for k in range(progeny_N):
                    cum_Prob = np.cumsum(transition_prob(x, lattice, diff_sigma, dL))
                    # pick the first event that is larger than the random variable
                    new_id = np.nonzero(cum_Prob > np.random.rand())[0][0].astype(int)
                    id_next.append(new_id)
                    bb = lattice[new_id]
                    x_next_temp = []
                    for y in bb:
                        barcode_list = X[1]
                        if np.random.rand() < barcoding_rate:
                            barcode_list.append(np.random.randint(1000000))

                        x_next_temp.append([y, barcode_list])
                    x_next = x_next + x_next_temp

    else:

        for j, X in enumerate(x_t0):
            x = X[0]  # X[0], location; X[1], fate choice; X[2], barcode information
            ## this method is much faster
            if x < L - (1.5 * progression(x, dL) + 5 * sigma(x, diff_sigma)):
                bb = np.random.normal(
                    x + progression(x, dL), sigma(x, diff_sigma), progeny_N
                )
                id_next = id_next + list(np.floor(bb / dx).astype(int))

            else:
                ## this method is much slower
                bb = []
                for k in range(progeny_N):
                    cum_Prob = np.cumsum(transition_prob(x, lattice, diff_sigma, dL))
                    # pick the first event that is larger than the random variable
                    new_id = np.nonzero(cum_Prob > np.random.rand())[0][0].astype(int)
                    id_next.append(new_id)
                    bb.append(lattice[new_id])

            ##  fate choice
            fate_choice_list = []
            if x > bifurcation:
                fate_choice_list = [X[1] for y in bb]
            else:
                for y in bb:
                    if y < bifurcation:
                        fate_choice_list.append(-1)
                    else:
                        fate = np.random.randint(2)
                        fate_choice_list.append(fate)

            ## update barcode
            barcode_list_array = []
            for y in bb:
                barcode_list = X[2]
                if np.random.rand() < barcoding_rate:
                    barcode_list.append(np.random.randint(1000000))

                barcode_list_array.append(barcode_list.copy())

            for j in range(len(bb)):
                # pdb.set_trace()
                x_next.append([bb[j], fate_choice_list[j], barcode_list_array[j]])

    return x_next, id_next


def bifurcation_model(
    progeny_N=2,
    t1=5,
    p1=0.5,
    p2=1,
    M=50,
    L=10,
    diff_sigma=0.5,
    repeat_N=1,
    dL=1,
    no_loss=False,
    always_simulate_data=0,
):
    """
    Simulate bifurcation corrupted with clonal dispersion (See Fig. 3e)

    Parameters
    ----------
    progeny_N:
        Fold change of clone size after each generation. 2 means that it will double its clone size after one generation.
    t1:
        Initital sampling time point. Unit: cell cycle. By default
        t2=t1+1.
    p1:
        Probability to sample cells at t1
    p2:
        Probability to sample cells at t2
    M:
        Total number of clones to simulate
    L:
        Total length of the 1-d differentiation manifold
    diff_sigma:
        Differentiation noise
    dL:
        Step size of differentiation for one generation
    no_loss:
        whether the sampling kills the cell or not. Default: sampling kills cells.
        x_t0:
        List of inititial cell locations at t0 along a 1-d line with length L
    always_simulate_data:
        Simulate new data (do not load pre-saved datasets)

    Returns
    -------
    adata:
        An adata object with clonal matrix, time info etc. Ready to be plugged into CoSpar.
    """

    if always_simulate_data:
        file_name = "simulate_data"
    else:
        file_name = f"{settings.data_path}/simulated_clonal_data_bifurcation_M{M}_progeny{progeny_N}_L{L}_dL{dL}_diffSigma{diff_sigma}_t1{t1}_p1{p1}_p2{p2}_simuV{repeat_N}_noloss{no_loss}"
    if os.path.exists(file_name + "_clonal_annot.npz"):
        print("Load existing data")
        clone_annot = ssp.load_npz(file_name + "_clonal_annot.npz")
        simu_data = np.load(file_name + "_others.npz", allow_pickle=True)
        final_coordinates = simu_data["final_coordinates"]
        time_info = simu_data["time_info"]
        bifurcation = 0.5 * L
    #             n=M*(progeny_N**(t1+1))*20 # resolution of grid for sampling
    #             dx=L/n
    else:
        print("Generate new data")
        t = time.time()
        ############################################ simulate the clonal data for each clone
        n = M * (progeny_N ** (t1 + 1)) * 20  # resolution of grid for sampling
        dx = L / n
        lattice = np.linspace(0, L, n)
        t0_id = np.sort(
            np.random.choice(n, M, replace=False)
        )  # random sample of indices
        x_t0_array = np.array(lattice[t0_id])  # position of initial barcoding
        # x_t0_array=np.array(range(20))
        x_t0_array = x_t0_array[x_t0_array < (L - t1 * dL)]
        # print(x_t0_array)

        bifurcation = 0.5 * L

        parameter = {}
        parameter["L"] = L
        parameter["dL"] = dL
        parameter["diff_sigma"] = diff_sigma
        parameter["progeny_N"] = progeny_N
        parameter["dx"] = dx
        parameter["lattice"] = lattice
        parameter["bifurcation"] = bifurcation

        final_coordinates = []
        time_info = []
        clone_id = []
        temp_clone_id = 0
        previouse_cell_N = 0
        # x_t0_array=[0.5,1,3,24,40] # initially barcoded cell positions

        ## simulate the multi-generational drift
        for m31 in tqdm(range(len(x_t0_array))):
            x_t0 = x_t0_array[m31]
            # print("Current clone number:", m31)
            x_next = []
            if x_t0 < bifurcation:
                x_next.append([x_t0, -1])
            else:
                fate = np.random.randint(2)
                x_next.append([x_t0, fate])

            ## 1d simulation
            for j in range(t1):
                x_next, id_next = simulate_cell_position_next_time_point(
                    x_next, parameter, mode="2d"
                )

            ### sampling: t1
            sel_idx = np.random.rand(len(x_next)) < p1
            sel_id = list(np.nonzero(sel_idx)[0].astype(int))
            unsel_id = list(np.nonzero(~sel_idx)[0].astype(int))
            final_coordinates = final_coordinates + list(np.array(x_next)[sel_id])
            time_info = time_info + [1 for j in range(len(sel_id))]

            if no_loss:  # whether the sampling kills the cell or not
                new_x = np.array(x_next)[sel_id]  # remaining cells
            else:
                new_x = np.array(x_next)[unsel_id]  # remaining cells

            # new_x=np.array(x_next)[unsel_id]# remaining cells
            x_next, id_next = simulate_cell_position_next_time_point(
                new_x, parameter, mode="2d"
            )
            ### sampling: t2
            sel_idx = np.random.rand(len(x_next)) < p2
            sel_id = list(np.nonzero(sel_idx)[0].astype(int))
            final_coordinates = final_coordinates + list(np.array(x_next)[sel_id])
            time_info = time_info + [2 for j in range(len(sel_id))]

            clone_id = clone_id + [
                temp_clone_id for j in range(len(time_info) - previouse_cell_N)
            ]
            previouse_cell_N = len(time_info)
            temp_clone_id = temp_clone_id + 1

        ### Generate clonal data matrix
        clone_annot = np.zeros((len(time_info), len(x_t0_array)))
        # ini_clone_id=0
        # jump=np.diff(time_info)
        # clone_id=np.zeros(len(time_info))
        for j in range(len(time_info)):
            clone_annot[j, clone_id[j]] = 1
            # clone_id[j]=ini_clone_id

        clone_annot = ssp.csr_matrix(clone_annot)
        if not always_simulate_data:
            ssp.save_npz(file_name + "_clonal_annot.npz", clone_annot)
            np.savez(
                file_name + "_others.npz",
                final_coordinates=final_coordinates,
                time_info=time_info,
            )
        print("Time elapsed for generating clonal data: ", time.time() - t)

    ## transformation to 50-d
    UMAP_noise = 0.2  #
    final_coordinates = np.array(final_coordinates)
    state_annote_0 = final_coordinates[:, 1]
    x_caleb = np.zeros(len(final_coordinates))
    y_caleb = np.zeros(len(final_coordinates))
    idx = state_annote_0 == -1
    x_caleb[idx] = final_coordinates[idx, 0] - bifurcation
    y_caleb[idx] = 0 * final_coordinates[idx, 0]
    idx = state_annote_0 == 0
    x_caleb[idx] = 0.5 * (final_coordinates[idx, 0] - bifurcation)
    y_caleb[idx] = 0.5 * (final_coordinates[idx, 0] - bifurcation)
    idx = state_annote_0 == 1
    x_caleb[idx] = 0.5 * (final_coordinates[idx, 0] - bifurcation)
    y_caleb[idx] = -0.5 * (final_coordinates[idx, 0] - bifurcation)

    all_coord = np.zeros((len(x_caleb), 50))
    all_coord[:, 0] = x_caleb
    all_coord[:, 1] = y_caleb

    for j in range(48):
        all_coord[:, j + 2] = UMAP_noise * np.random.randn(len(x_caleb))
    adata = sc.AnnData(ssp.csr_matrix(all_coord))
    adata.obsm["X_clone"] = clone_annot
    adata.obs["time_info"] = time_info
    adata.obs["state_info"] = state_annote_0.astype(int).astype(str)
    adata.obsm["X_emb"] = all_coord[:, :2]
    adata.uns["data_des"] = ["bifurcation"]
    adata.obsm["X_orig"] = final_coordinates
    return adata


def linear_differentiation_model(
    Nt1=400,
    progeny_N=1,
    L=100,
    diff_sigma=0.5,
    dL=1,
    display=1,
    always_simulate_data=0,
    used_clone_N=100,
):
    """
    Simulate linear differentiation corrupted with barcode collision (See Fig. 3a)

    Parameters
    ----------
    Nt1:
        Number of initial cell states. They are randomly sampled along the differentiation manifold. Default 400.
    progeny_N:
        Number of progeny for each initial cell state
    L:
        The total progression length.
    diff_sigma:
        Differentiation noise.
    dL:
        Unit progression in a single step
    display:
        Plot figures or not
    always_simulate_data:
        Simulate new data (do not load pre-saved datasets)

    Returns
    -------
    adata:
        An adata object with clonal matrix, time info etc. Ready to be plugged into CoSpar.
    """

    if always_simulate_data:
        file_name = "simulate_data"
    else:
        file_name = f"{settings.data_path}/simulated_clonal_data_Nt1{Nt1}_progeny{progeny_N}_L{L}_dL{dL}_diffSigma{diff_sigma}"
    if os.path.exists(file_name + "_clonal_annot.npz"):
        print("Load existing data")
        clone_annot = ssp.load_npz(file_name + "_clonal_annot.npz")
        simu_data = np.load(file_name + "_others.npz", allow_pickle=True)
        x_t1 = simu_data["x_t1"]
        x_t2 = simu_data["x_t2"]
        Nt2 = simu_data["Nt2"]
        new_id_matrix = simu_data["new_id_matrix"]
        n = Nt1 * 20  # resolution of grid for sampling
        dx = L / n
    else:
        print("Generate new data")
        t = time.time()
        ## simulate the clonal data for each initial cell
        n = Nt1 * 20  # resolution of grid for sampling
        dx = L / n
        lattice = np.linspace(0, L, n)
        t1_id = np.sort(
            np.random.choice(n, Nt1, replace=False)
        )  # random sample of indices
        x_t1 = lattice[t1_id]
        id_matrix = np.zeros(
            (Nt1, progeny_N), dtype=int
        )  # information for single cell clones
        for j, x in enumerate(x_t1):
            ## this method is much faster
            if x < L - (1.5 * progression(x, dL) + 5 * sigma(x, diff_sigma)):
                bb = np.random.normal(
                    x + progression(x, dL), sigma(x, diff_sigma), progeny_N
                )
                id_matrix[j] = np.floor(bb / dx).astype(int)
            else:
                ## this method is much slower
                for k in range(progeny_N):
                    cum_Prob = np.cumsum(transition_prob(x, lattice, diff_sigma, dL))
                    # pick the first event that is larger than the random variable
                    id_matrix[j, k] = np.nonzero(cum_Prob > np.random.rand())[0][
                        0
                    ].astype(int)

        ## prepare clonal data
        t2_id = np.sort(id_matrix.flatten())
        x_t2 = np.sort(lattice[t2_id])
        Nt2 = len(t2_id)

        ## generate clonal matrix
        new_id_matrix = np.zeros(id_matrix.shape, dtype=int)
        # for j in range(id_matrix.shape[0]):
        for j in range(id_matrix.shape[0]):
            for k in range(id_matrix.shape[1]):
                # print(id_matrix[j,k])
                new_id_matrix[j, k] = Nt1 + np.nonzero(t2_id == id_matrix[j, k])[0][0]

        clone_annot = np.zeros((Nt1 + Nt2, Nt1), dtype=int)
        for j in range(Nt1):
            clone_annot[j, j] = 1
            clone_annot[new_id_matrix[j, :], j] = 1

        clone_annot = ssp.csr_matrix(clone_annot)

        # if display:
        #     idx = 3
        #     # cum_Prob=np.cumsum(transition_prob(x_t1[idx],lattice),dL)
        #     norm_prob = transition_prob(x_t1[idx], lattice, diff_sigma, dL)
        #     norm_prob = norm_prob / np.max(norm_prob)

        #     fig = plt.figure(figsize=(4, 3.5))
        #     ax = fig.add_subplot(1, 1, 1)
        #     ax.plot(lattice, norm_prob, label="Relative Prob.")
        #     x0 = lattice[id_matrix[idx]]
        #     y0 = np.array([1, 1, 1])
        #     ax.plot(x0, y0, ".r", label="Sampled point")
        #     leg = ax.legend()
        #     ax.set_xlabel("Differentiation progression")
        #     plt.tight_layout()
        #     fig.savefig(f"{settings.figure_path}/sample_data.eps")

        if not always_simulate_data:
            ssp.save_npz(file_name + "_clonal_annot.npz", clone_annot)
            np.savez(
                file_name + "_others.npz",
                x_t1=x_t1,
                x_t2=x_t2,
                Nt1=Nt1,
                Nt2=Nt2,
                new_id_matrix=new_id_matrix,
            )
        print("Time elapsed for generating clonal data: ", time.time() - t)

    ## cell annotation
    x_tot = np.array(list(x_t1) + list(x_t2))
    time_index_1 = np.array(range(Nt1))
    time_index_2 = np.array(range(Nt1, Nt1 + Nt2))
    time_info = np.zeros(len(x_tot))
    time_info[time_index_1] = 1
    time_info[time_index_2] = 2

    ## Compress the clones via random mixing
    compressed_N = np.round(Nt1 / used_clone_N).astype(int)
    CS_file_name = "generate new ones"
    if os.path.exists(CS_file_name):
        print("Load existing mixing matrix")
        mixing_matrix = np.load(CS_file_name)
    else:
        if display:
            print("Generate mixing matrix")
        N_cell, N_clone = clone_annot.shape

        post_PC_clone_N = np.floor(N_clone / compressed_N).astype(int)
        mixing_matrix = np.zeros((N_clone, post_PC_clone_N))
        current_clone_list = list(range(N_clone))
        for j in range(post_PC_clone_N):
            ri = np.random.choice(
                len(current_clone_list), compressed_N, replace=False
            )  # random sample of indices
            clone_ids = np.array(current_clone_list)[ri]
            mixing_matrix[clone_ids, j] = 1
            # remove these selected elements in the list
            for k in range(compressed_N):
                current_clone_list.remove(clone_ids[k])

    clone_annot_CS = ssp.csr_matrix(clone_annot * mixing_matrix)

    ## transformation to 50-d
    UMAP_noise = 0.2  #
    all_coord = np.zeros((len(x_tot), 50))
    all_coord[:, 0] = x_tot
    for j in range(49):
        all_coord[:, j + 1] = UMAP_noise * np.random.randn(len(x_tot))

    ## generate the adata
    adata = sc.AnnData(ssp.csr_matrix(all_coord))
    adata.obsm["X_clone"] = clone_annot_CS
    adata.obs["time_info"] = time_info
    adata.obs["state_info"] = np.zeros(len(x_tot)).astype(str)
    X_emb = np.zeros((len(x_tot), 2))
    X_emb[:, 0] = x_tot
    X_emb[:, 1] = x_tot
    adata.obsm["X_emb"] = X_emb
    adata.uns["data_des"] = ["linear_bifurcation"]
    adata.obsm["X_orig"] = x_tot
    return adata


def quantify_correlation_with_ground_truth_fate_bias_BifurcationModel(
    used_map, state_annote_0, cell_id_t1, cell_id_t2
):
    """
    Quantify the correlation of an inferred fate bias with the actual one, using a map from the Bifurcation Model.

    Parameters
    ----------
    used_map:
        Used tmap
    state_annote_0:
        The actual state annotation outputed from the Bifurcation Model.
    cell_id_t1:
        List of initial cell IDs
    cell_id_t2:
        List of later cell IDs

    Returns
    -------
    fate_bias_corr:
        Correlation between the predicted and actual fate bias
    """

    state_annote = state_annote_0[cell_id_t2]
    fate_array = ["0", "1", "-1"]

    potential_vector, fate_entropy = tl.compute_state_potential(
        used_map, state_annote, fate_array, fate_count=False
    )

    potential_vector = potential_vector + 0.01
    fate_bias = potential_vector[:, 0] / (
        potential_vector[:, 0] + potential_vector[:, 1]
    )

    state_annote_1 = state_annote_0[cell_id_t1]
    expected_fate_prob = np.zeros(len(state_annote_1))
    actual_fate_prob = np.zeros(len(state_annote_1))

    idx = state_annote_1 == "-1"
    expected_fate_prob[idx] = 0.5  # neutral
    actual_fate_prob[idx] = fate_bias[idx]

    idx = state_annote_1 == "0"
    expected_fate_prob[idx] = 1  # one branch
    actual_fate_prob[idx] = fate_bias[idx]

    idx = state_annote_1 == "1"
    expected_fate_prob[idx] = 0  # the other branch
    actual_fate_prob[idx] = fate_bias[idx]

    fate_bias_corr = np.corrcoef(expected_fate_prob, actual_fate_prob)[0, 1]

    return fate_bias_corr


def quantify_transition_peak_TPR_LinearDifferentiation(
    used_map, x_t1, x_t2, dL=1, relative_tolerance=3, diff_sigma=0.5, display=True
):
    """
    Quantify the True positive rate of a transition map for linear differentiation model.

    Parameters
    ----------
    used_map:
        Used tmap, in the form of numpy array, or sparse matrix
    x_t1:
        List of initial cell positions
    x_t2:
        List of later cell positions
    dL:
        Unit progression in a single time step
    relative_tolerance:
        Accept a prediction to be true if its most likely predicted target state is within
        relative_tolerance*diff_sigma of the expected target state.
    diff_sigma:
        Strength of differentiation noise.
    display:
        Show analysis figures if true.

    Returns
    --------
    Tmap_right_ratio:
        True positive rate of a transition map
    """

    if ssp.issparse(used_map):
        used_map = used_map.A

    Nt1 = len(x_t1)
    true_x = np.zeros(Nt1)
    predicted_x = np.zeros(Nt1)
    Tmap_right_ratio = []

    for ini_id in range(Nt1):
        true_x[ini_id] = x_t1[ini_id] + progression(x_t1[ini_id], dL)

        temp_map = used_map[ini_id]
        idx = temp_map > 0.999 * np.max(temp_map)
        if np.sum(idx) > 1:
            temp_sel_idx = np.random.choice(np.nonzero(idx)[0], 1)[0]
            predicted_temp_x = x_t2[temp_sel_idx]
        else:
            predicted_temp_x = x_t2[np.argsort(temp_map)[::-1][0]]

        predicted_x[ini_id] = predicted_temp_x

    relative_x = predicted_x - true_x
    Tmap_right_ratio = (
        1 - np.sum(abs(relative_x) > relative_tolerance * diff_sigma) / Nt1
    )

    if display:
        fig = plt.figure(figsize=(4, 3.5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(predicted_x, true_x, ".r", label="ICSLAM")
        ax.set_xlabel("Predicted progression: $x_2$")
        # ax.legend(loc='best')
        ax.set_ylabel("Actual progression: $x_2$")
        # ax.set_title(f'Ini. cell #={Nt1}, Clone #={clone_annot_CS.shape[1]}')
        plt.tight_layout()
        fig.savefig(f"{settings.figure_path}/progression_comparison_1.eps")

    return Tmap_right_ratio
