import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import os



def plot_heatmaps_new(pickle_path):
    with open(pickle_path, "rb") as f:
        data , kappa_values = pickle.load(f)   

    kappa_keys = list(data.keys())
    planning_keys = list(data[kappa_keys[0]].keys())
    algorithm_names = list(data[kappa_keys[0]][planning_keys[0]].keys())

    n_kappa = len(kappa_keys)
    n_planning = len(planning_keys)
    n_algs = len(algorithm_names)

    left_arr = np.zeros((n_algs, n_kappa, n_planning))
    right_arr = np.zeros((n_algs, n_kappa, n_planning))
    cumulative_arr = np.zeros((n_algs, n_kappa, n_planning))

    has_right = False

    for ki, kappa_key in enumerate(kappa_keys):
        for pi, planning_key in enumerate(planning_keys):
            for ai, alg_name in enumerate(algorithm_names):
                left_val, right_val = data[kappa_key][planning_key][alg_name]
                left_arr[ai, ki, pi] = left_val
                right_arr[ai, ki, pi] = right_val
                cumulative_arr[ai, ki, pi] = left_val + right_val
                if right_val != 0.0:
                    has_right = True

    kappa_floats = np.array(kappa_values.tolist()) if torch.is_tensor(kappa_values) else np.array(kappa_values)
    planning_floats = np.array(planning_keys)

    def make_figure(arr, title):
        fig, axes = plt.subplots(1, n_algs, figsize=(5 * n_algs, 5), constrained_layout=True)
        if n_algs == 1:
            axes = [axes]
        fig.suptitle(title, fontsize=16)

        vmin, vmax = arr.min(), arr.max()

        for ai, alg_name in enumerate(algorithm_names):
            ax = axes[ai]
            im = ax.imshow(
                arr[ai],
                aspect="auto",
                origin="lower",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )

            n_yticks = 6
            ytick_indices = np.linspace(0, n_kappa - 1, n_yticks, dtype=int)
            ax.set_yticks(ytick_indices)
            ax.set_yticklabels([f"{kappa_floats[i]:.1e}" for i in ytick_indices])

            n_xticks = min(n_planning, 6)
            xtick_indices = np.linspace(0, n_planning - 1, n_xticks, dtype=int)
            ax.set_xticks(xtick_indices)
            ax.set_xticklabels([str(planning_floats[i]) for i in xtick_indices])

            ax.set_title(alg_name, fontsize=12)
            ax.set_xlabel("Planning steps")
            if ai == 0:
                ax.set_ylabel("Kappa")

            fig.colorbar(im, ax=ax, shrink=0.8)

        return fig

    fig_left = make_figure(left_arr, "Cumulative reward — env A (stationary)")
    fig_right = make_figure(right_arr, "Cumulative reward — env B (after wall change)") if has_right else None
    fig_cum = make_figure(cumulative_arr, "Cumulative reward") if has_right else None

    return fig_left, fig_right, fig_cum


def plot_heatmaps_new_2x2(pickle_path):
    with open(pickle_path, "rb") as f:
        data, kappa_values = pickle.load(f)

    kappa_keys = list(data.keys())
    planning_keys = list(data[kappa_keys[0]].keys())
    algorithm_names = list(data[kappa_keys[0]][planning_keys[0]].keys())

    n_kappa = len(kappa_keys)
    n_planning = len(planning_keys)
    n_algs = len(algorithm_names)

    left_arr = np.zeros((n_algs, n_kappa, n_planning))
    right_arr = np.zeros((n_algs, n_kappa, n_planning))
    cumulative_arr = np.zeros((n_algs, n_kappa, n_planning))

    has_right = False

    for ki, kappa_key in enumerate(kappa_keys):
        for pi, planning_key in enumerate(planning_keys):
            for ai, alg_name in enumerate(algorithm_names):
                left_val, right_val = data[kappa_key][planning_key][alg_name]
                left_arr[ai, ki, pi] = left_val
                right_arr[ai, ki, pi] = right_val
                cumulative_arr[ai, ki, pi] = left_val + right_val
                if right_val != 0.0:
                    has_right = True

    kappa_floats = np.array(kappa_values.tolist()) if torch.is_tensor(kappa_values) else np.array(kappa_values)
    planning_floats = np.array(planning_keys)

    def make_figure(arr, title):
        ncols = 2
        nrows = int(np.ceil(n_algs / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(10, 5 * nrows),
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes).ravel()
        fig.suptitle(title, fontsize=16)

        vmin, vmax = arr.min(), arr.max()

        for ai, alg_name in enumerate(algorithm_names):
            ax = axes[ai]
            im = ax.imshow(
                arr[ai],
                aspect="auto",
                origin="lower",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )

            n_yticks = min(n_kappa, 6)
            ytick_indices = np.linspace(0, n_kappa - 1, n_yticks, dtype=int)
            ax.set_yticks(ytick_indices)
            ax.set_yticklabels([f"{kappa_floats[i]:.1e}" for i in ytick_indices])

            n_xticks = min(n_planning, 6)
            xtick_indices = np.linspace(0, n_planning - 1, n_xticks, dtype=int)
            ax.set_xticks(xtick_indices)
            ax.set_xticklabels([str(planning_floats[i]) for i in xtick_indices])

            ax.set_title(alg_name, fontsize=12)
            ax.set_xlabel("Planning steps")
            ax.set_ylabel("Kappa")

            fig.colorbar(im, ax=ax, shrink=0.8)

        for ax in axes[n_algs:]:
            ax.remove()

        return fig

    fig_left = make_figure(left_arr, "Cumulative reward — env A (stationary)")
    fig_right = make_figure(right_arr, "Cumulative reward — env B (after wall change)") if has_right else None
    fig_cum = make_figure(cumulative_arr, "Cumulative reward") if has_right else None

    return fig_left, fig_right, fig_cum

def plot_heatmaps(pickle_path, kappa_values):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    kappa_keys = list(data.keys())
    planning_keys = list(data[kappa_keys[0]].keys())
    algorithm_names = list(data[kappa_keys[0]][planning_keys[0]].keys())

    n_kappa = len(kappa_keys)
    n_planning = len(planning_keys)
    n_algs = len(algorithm_names)

    left_arr = np.zeros((n_algs, n_kappa, n_planning))
    right_arr = np.zeros((n_algs, n_kappa, n_planning))
    cumulative_arr = np.zeros((n_algs, n_kappa, n_planning))

    has_right = False

    for ki, kappa_key in enumerate(kappa_keys):
        for pi, planning_key in enumerate(planning_keys):
            for ai, alg_name in enumerate(algorithm_names):
                left_val, right_val = data[kappa_key][planning_key][alg_name]
                left_arr[ai, ki, pi] = left_val
                right_arr[ai, ki, pi] = right_val
                cumulative_arr[ai, ki, pi] = left_val + right_val
                if right_val != 0.0:
                    has_right = True

    kappa_floats = np.array(kappa_values.tolist()) if torch.is_tensor(kappa_values) else np.array(kappa_values)
    planning_floats = np.array(planning_keys)

    def make_figure(arr, title):
        fig, axes = plt.subplots(1, n_algs, figsize=(5 * n_algs, 5), constrained_layout=True)
        if n_algs == 1:
            axes = [axes]
        fig.suptitle(title, fontsize=16)

        vmin, vmax = arr.min(), arr.max()

        for ai, alg_name in enumerate(algorithm_names):
            ax = axes[ai]
            im = ax.imshow(
                arr[ai],
                aspect="auto",
                origin="lower",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )

            n_yticks = 6
            ytick_indices = np.linspace(0, n_kappa - 1, n_yticks, dtype=int)
            ax.set_yticks(ytick_indices)
            ax.set_yticklabels([f"{kappa_floats[i]:.1e}" for i in ytick_indices])

            n_xticks = min(n_planning, 6)
            xtick_indices = np.linspace(0, n_planning - 1, n_xticks, dtype=int)
            ax.set_xticks(xtick_indices)
            ax.set_xticklabels([str(planning_floats[i]) for i in xtick_indices])

            ax.set_title(alg_name, fontsize=12)
            ax.set_xlabel("Planning steps")
            if ai == 0:
                ax.set_ylabel("Kappa")

            fig.colorbar(im, ax=ax, shrink=0.8)

        return fig

    fig_left = make_figure(left_arr, "Cumulative reward — env A (stationary)")
    fig_right = make_figure(right_arr, "Cumulative reward — env B (after wall change)") if has_right else None
    fig_cum = make_figure(cumulative_arr, "Cumulative reward") if has_right else None

    return fig_left, fig_right, fig_cum

if __name__ == "__main__":
    os.chdir("content/posts/Reinforcement Learning/Planning and learning/programming_task")



    plt.style.use('dark_background')
    
    # filename = "kappa_planning_study_logspace_q+_vs_action_bonus"
    # kappa_values = torch.logspace(-20,-7,  1000)
    filename = "data/kappa_planning_study_logspace_planning_1_to_25_6000.pickle"
    kappa_values = torch.logspace(-4, 0, 1000)

    # plot_heatmaps(filename, kappa_values)

    # kappa_values = torch.logspace(-40, -20, 2000)
    # filename = "ultra_low_kappa"
    # fig_left, fig_right , fig_cum = plot_heatmaps(
    #     f"{filename}.pickle",
    #     kappa_values,
    # )


    fig_left, fig_right , fig_cum = plot_heatmaps_new_2x2("data/arbitrary_tie_breaking.pickle")

    # fig_left, fig_right , fig_cum = plot_heatmaps_new("data/low_kappa_arbitrary_tie_breaking.pickle")

    fig_left.savefig("dark_imgs/heatmap_envA.svg", bbox_inches="tight", transparent=True)
    if fig_right is not None:
        fig_right.savefig("dark_imgs/heatmap_envB.svg", bbox_inches="tight", transparent=True) 

    # if fig_cum is not None:
    #     fig_cum.savefig("dark_imgs/heatmap_cum.svg", bbox_inches="tight", transparent=True)

    plt.show()
    
