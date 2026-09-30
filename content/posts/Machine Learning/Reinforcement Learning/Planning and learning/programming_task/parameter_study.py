import os
import pickle

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import pickle
from heatmap import plot_heatmaps_new
from algorithms_param_study import (
    DynaQ,
    DynaQ_plus,
    DynaQ_plus_action_bonus,
    DynaQ_plus_selective_sample,
    Gridworld,
    set_wall_a,
    set_wall_b,
)

os.chdir("content/posts/Reinforcement Learning/Planning and learning/programming_task/data")


gridsize = (6, 9)
no_states = gridsize[0] * gridsize[1]
no_actions = 4
no_runs = 5
dynaQ_no_runs = 50
max_steps_envA = 3000
max_steps_envB = 3000
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
BASE_SEED = 1234


algorithms = {
    "DynaQ+ Selective": DynaQ_plus_selective_sample,
    "DynaQ+ Action Bonus": DynaQ_plus_action_bonus,
    "DynaQ+": DynaQ_plus,
    "DynaQ": DynaQ,
}

def _set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _run_dynaQ(kappa_keys,planning_values,  max_steps_envA, max_steps_envB):
    env = Gridworld(
        device,
        dynaQ_no_runs,
        len(planning_values),
        gridsize,
        goal_state=(0, 8),
    )
    planning_tensor = torch.tensor(
        planning_values, device=device, dtype=torch.long
    )
    agent = DynaQ(
        device,
        dynaQ_no_runs,
        len(planning_values),
        no_states,
        no_actions,
        env,
        gamma=0.8,
    )


    set_wall_a(agent.env)
    left = agent.train(max_steps_envA, planning_tensor).cpu()

    if max_steps_envB:
        set_wall_b(agent.env)
        right = agent.train(max_steps_envB, planning_tensor).cpu()
    else:
        right = torch.zeros((len(planning_values), 1))

    averaged_results = {
        planning_steps: (
            float(left[param_idx, -1].item()),
            float(right[param_idx, -1].item()),
        )
        for param_idx, planning_steps in enumerate(planning_values)
    }
    return {
        (kappa_key, planning_steps): averaged_results[planning_steps]
        for kappa_key in kappa_keys
        for planning_steps in planning_values
    }
        


def run_algorithms_grid(algorithms,
                        kappa_keys,
                        kappa_values,
                        planning_values,
                        max_steps_envA=5000,
                        max_steps_envB=5000,
                        seed=None,):
    results = {}
    planning_values = list(planning_values)
    kappa_values = list(kappa_values)

    if not planning_values:
        raise ValueError("planning_values must contain at least one value")
    if not kappa_values:
        raise ValueError("kappa_values must contain at least one value")

    combo_keys = []
    combo_kappas = []
    combo_planning = []
    for kappa_key, kappa_value in tqdm(
        list(zip(kappa_keys, kappa_values)),
        total=len(kappa_keys),
        leave=False,
        desc="Building parameter grid",
    ):
        for planning_steps in planning_values:
            combo_keys.append((kappa_key, planning_steps))
            combo_kappas.append(kappa_value)
            combo_planning.append(planning_steps)

    no_params = len(combo_keys)
    env = Gridworld(device, no_runs, no_params, gridsize, goal_state=(0, 8))
    batched_kappa = torch.tensor(combo_kappas, device=device, dtype=torch.float32)
    batched_planning_steps = torch.tensor(combo_planning, device=device, dtype=torch.long)

    for algorithm_idx, (name, algorithm) in enumerate(tqdm(algorithms.items(), leave=False, desc="Running Algorithms")):
        if seed is not None:
            _set_seed(seed + algorithm_idx)

        env.reset()

        if algorithm == DynaQ:
            results[name] = _run_dynaQ(kappa_keys=kappa_keys, 
                                       planning_values=planning_values,
                                       max_steps_envA=max_steps_envA,
                                       max_steps_envB=max_steps_envB)
            continue

        agent = algorithm(
            device,
            no_runs,
            no_params,
            no_states,
            no_actions,
            env,
            kappa=batched_kappa,
            gamma=0.8,
        )

        set_wall_a(agent.env)
        left = agent.train(max_steps_envA, batched_planning_steps).cpu()

        if max_steps_envB:
            set_wall_b(agent.env)
            right = agent.train(max_steps_envB, batched_planning_steps).cpu()
        else:
            right = torch.zeros((len(combo_keys), 1))

        results[name] = {
            combo_key: (
                float(left[param_idx, -1].item()),
                float(right[param_idx, -1].item()),
            )
            for param_idx, combo_key in enumerate(combo_keys)
        }

        # full_curve = torch.cat((left, right + left[:, -1:].clone()), dim=1)
        # print(f"Memory: {full_curve.numel() * full_curve.element_size() / (1024**2):.2f} MB")


        # results[name] = {
        #     combo_key: full_curve[param_idx]
        #     for param_idx, combo_key in enumerate(combo_keys)
        # }

    return results



def run_kappa_planning_study(
    kappa_start=1,
    kappa_stop=101,
    kappa_step=1,
    scale=1e-5,
    kappa_values=None,
    planning_start=1,
    planning_stop=26,
    planning_step=1,
    algorithms=algorithms,
    max_steps_envA=max_steps_envA,
    max_steps_envB=max_steps_envB,
):
    if kappa_values is None:
        kappa_keys = list(range(kappa_start, kappa_stop, kappa_step))
        kappa_values = [kappa_key * scale for kappa_key in kappa_keys]
    else:
        if torch.is_tensor(kappa_values):
            kappa_values = kappa_values.tolist()
        else:
            kappa_values = list(kappa_values)
        kappa_keys = list(range(len(kappa_values)))
    planning_values = list(range(planning_start, planning_stop, planning_step))

    batched_results = run_algorithms_grid(
        algorithms=algorithms,
        kappa_keys=kappa_keys,
        kappa_values=kappa_values,
        planning_values=planning_values,
        max_steps_envA=max_steps_envA,
        max_steps_envB=max_steps_envB,
        seed=BASE_SEED,
    )

    data = {}
    for kappa_key in tqdm(kappa_keys, desc="Collecting kappa slices", leave=False):
        data[kappa_key] = {}
        for planning_steps in planning_values:
            data[kappa_key][planning_steps] = {
                algorithm_name: per_param[(kappa_key, planning_steps)]
                for algorithm_name, per_param in batched_results.items()
            }

    return data



def plot(results, filename=None, show=True):
    styles = [("default", "light_imgs"), ("dark_background", "dark_imgs")]

    for i, (style, folder) in enumerate(styles):
        plt.style.use(style)
        plt.figure(figsize=(12, 4.5))

        for label, data in results.items():
            plt.plot(data, label=label)

        plt.xlabel("Steps", fontsize=18)
        plt.ylabel("Cumulative Reward", fontsize=18)
        plt.legend(fontsize=16)

        if filename is not None:
            save_path = f"{folder}/{filename}.svg"
            plt.savefig(save_path, bbox_inches="tight", transparent=True)

        if i == len(styles) - 1 and show:
            plt.show()

        plt.close()


if __name__ == "__main__":
    kappa_values = torch.logspace(-20, -7, 3500)

    data = run_kappa_planning_study(
        kappa_values=kappa_values,
        planning_start=1,
        planning_stop=26,
        planning_step=1,
        max_steps_envA=6000 ,
        max_steps_envB=6000,
    )

    data = (data, kappa_values)

    pickled_data = pickle.dumps(data)
    size_mb = len(pickled_data) / (1024 * 1024)
    print(f"Size of data: {size_mb:.2f} MB")


    filename = "low_kappa_arbitrary_tie_breaking.pickle"
    
    with open(filename, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    plot_heatmaps_new(filename)

