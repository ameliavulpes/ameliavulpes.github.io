import torch
from algorithms import Gridworld, DynaQ, DynaQ_plus, DynaQ_plus_action_bonus, DynaQ_plus_selective_sample
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import os
os.chdir("content/posts/Reinforcement Learning/Planning and learning/programming_task")


gridsize = 6, 9
no_states = gridsize[0] * gridsize[1]
no_actions = 4
no_runs = 50
steps_envA = 3000
steps_envB = 3000
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(device)
BASE_SEED = 1234

env = Gridworld(device, no_runs, gridsize, goal_state=(0, 8))

def set_wall_a(env):
    env.reset_wall()
    env.wall[3, 1:9] = True

def set_wall_b(env):
    # enable reset if walls are added
    # env.reset() 
    env.reset_wall()
    env.wall[3, 1:8] = True

algorithms = {"DynaQ" : DynaQ, 
              "DynaQ+ Selective" : DynaQ_plus_selective_sample, 
              "DynaQ+ Action Bonus" : DynaQ_plus_action_bonus,
              "DynaQ+" : DynaQ_plus, 
              }

def _set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_algorithms(algorithms, planning_steps,kappa=0.001, steps_envA= 5000,steps_envB= 5000, seed=None):
    results = {}
    for algorithm_idx, (name, algorithm) in enumerate(tqdm(algorithms.items(), leave=False, desc="Running Algorithms")):
        if seed is not None:
            _set_seed(seed + algorithm_idx)
        env.reset()
        agent = algorithm(device,no_runs, no_states, no_actions, env,kappa=kappa, gamma=0.8)
        set_wall_a(agent.env)
        left = agent.train(steps_envA, planning_steps)
        set_wall_b(agent.env)
        right = agent.train(steps_envB, planning_steps)

        results[name] = torch.concat((left,right + left[-1])).cpu()

    return results

def run_planning_study(start=1, stop=50, step= 1,kappa=0.001, algorithms= algorithms, steps_envA = steps_envA, steps_envB=steps_envB):
    data = {}
    for planning_steps in tqdm(range(start, stop, step), desc="Iter over planning steps"):
        results = run_algorithms(algorithms, planning_steps=planning_steps, kappa=kappa,
                                steps_envA=steps_envA, steps_envB=steps_envB,
                                seed=BASE_SEED + planning_steps * 1000)        
        data[planning_steps] = results
    return data

def run_kappa_study(start=1, stop=10, step=1,scale=1e-4,
                        planning_steps=50, steps_envA=steps_envA, steps_envB=steps_envB):
    data = {}
    print(start * scale)
    print(stop * scale)
    for kappa_key in tqdm(range(start, stop, step), desc=f"Iter over kappa {planning_steps} planning steps"):
        kappa = kappa_key *  scale
        results = run_algorithms(algorithms, planning_steps=planning_steps, kappa=kappa,
                                steps_envA=steps_envA, steps_envB=steps_envB,
                                seed=BASE_SEED + planning_steps * 1000)        
        data[kappa_key] = results
    return data

def plot(results, filename=None, show= True):
    styles = [('default', 'light_imgs'), ('dark_background', 'dark_imgs')]
        
    for i, (style, folder) in enumerate(styles):
        plt.style.use(style)
        plt.figure(figsize=(12, 4.5))
        
        # Plotting logic
        for label, data in results.items():
            plt.plot(data, label=label)
        
        plt.xlabel("Steps", fontsize=18)
        plt.ylabel("Cumulative Reward", fontsize=18)
        plt.legend(fontsize=16)
        
        if filename is not None:
            save_path = f"{folder}/{filename}.svg"
            plt.savefig(save_path, bbox_inches="tight", transparent=True)

        if i == len(styles) -1 and show:
            plt.show()

        plt.close() 



# data = run_kappa_study(start=1, stop=101, step=1, planning_steps=1,
#                         steps_envA=6000, steps_envB=6000,
#                         scale=1e-5)
# with open('kappa_study_1e-5_to_1e-3_planning_1_6000.pickle', 'wb') as f:
#     pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

data = run_kappa_study(start=1, stop=101, step=1, planning_steps=25,
                        steps_envA=6000, steps_envB=6000,
                        scale=1e-5)
with open('kappa_study_1e-5_to_1e-3_planning_25_6000.pickle', 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)