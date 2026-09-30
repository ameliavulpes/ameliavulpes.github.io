import torch
from tqdm import tqdm
from algorithms_param_study import (
    DynaQ,
    DynaQ_plus,
    DynaQ_plus_action_bonus,
    DynaQ_plus_selective_sample,
    Gridworld,
    set_wall_a,
    set_wall_b,
)


gridsize = (6, 9)
no_states = gridsize[0] * gridsize[1]
no_actions = 4
no_runs = 32
device = "cuda" if torch.cuda.is_available() else "cpu"
BASE_SEED = 1234


algorithms = {
    "DynaQ": DynaQ,
    "DynaQ+ Selective": DynaQ_plus_selective_sample,
    "DynaQ+": DynaQ_plus,
    "DynaQ+ Action Bonus": DynaQ_plus_action_bonus,
}


def _set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_algorithms_batched(
    algorithms,
    param_name,
    param_values,
    planning_steps,
    kappa,
    steps_envA=3000,
    steps_envB=3000,
    seed=BASE_SEED,
):
    results = {}
    param_values = list(param_values)
    no_params = len(param_values)

    if no_params == 0:
        raise ValueError("param_values must contain at least one value")

    env = Gridworld(device, no_runs, no_params, gridsize, goal_state=(0, 8))

    if param_name == "planning_steps":
        batched_planning_steps = torch.tensor(param_values, device=device, dtype=torch.long)
        batched_kappa = kappa
        result_keys = [int(v) for v in param_values]
    elif param_name == "kappa":
        batched_planning_steps = planning_steps
        batched_kappa = torch.tensor(param_values, device=device, dtype=torch.float32)
        result_keys = [float(v) for v in param_values]
    else:
        raise ValueError("param_name must be either 'planning_steps' or 'kappa'")

    for algorithm_idx, (name, algorithm) in tqdm(enumerate(algorithms.items())):
        _set_seed(seed + algorithm_idx)
        env.reset()

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
        left = agent.train(steps_envA, batched_planning_steps)
        set_wall_b(agent.env)
        right = agent.train(steps_envB, batched_planning_steps)

        full_curve = torch.cat((left, right + left[:, -1:].clone()), dim=1).cpu()
        results[name] = {
            key: full_curve[param_idx]
            for param_idx, key in enumerate(result_keys)
        }

    return results


def demo_planning_step_sweep():
    planning_values = [1, 5, 25]
    results = run_algorithms_batched(
        algorithms=algorithms,
        param_name="planning_steps",
        param_values=planning_values,
        planning_steps=None,
        kappa=0.001,
        steps_envA=500,
        steps_envB=500,
        seed=BASE_SEED,
    )

    print("Planning-step sweep")
    for algorithm_name, per_param in results.items():
        print(f"\n{algorithm_name}")
        for planning_steps, curve in per_param.items():
            print(
                f"  planning_steps={planning_steps:<3} "
                f"final cumulative reward={curve[-1].item():.3f}"
            )


def demo_kappa_sweep():
    # kappa_values = [1e-5, 1e-4, 1e-3]
    scale = 1e-5
    kappa_values = [unscaled_kappa * scale for unscaled_kappa in range(101)]
    results = run_algorithms_batched(
        algorithms=algorithms,
        param_name="kappa",
        param_values=kappa_values,
        planning_steps=10,
        kappa=None,
        steps_envA=500,
        steps_envB=500,
        seed=BASE_SEED,
    )

    print("\nKappa sweep")
    for algorithm_name, per_param in results.items():
        print(f"\n{algorithm_name}")
        for kappa, curve in per_param.items():
            print(
                f"  kappa={kappa:<8g} "
                f"final cumulative reward={curve[-1].item():.3f}"
            )


if __name__ == "__main__":
    # demo_planning_step_sweep()
    demo_kappa_sweep()
