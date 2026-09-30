import torch
from tqdm import tqdm


class Gridworld:
    def __init__(self, device, no_runs, gridsize, start_state=(5, 3), goal_state=(0, 8)) -> None:
        self.device = device
        self.no_runs = no_runs
        self.runs =  torch.arange(self.no_runs)

        # --- Static ENV for simplicity ---
        self.gridsize = gridsize
        self.max_row_idx = self.gridsize[0] - 1
        self.max_col_idx = self.gridsize[1] - 1


        self.start_state = torch.tensor(start_state, device=self.device)
        self.goal_state =  torch.tensor(goal_state, device=self.device)
        self.actions = torch.tensor((0,1,2,3), device=self.device)

        self.wall = torch.zeros(self.gridsize,device=device, dtype=torch.bool)
        self.state = torch.tile(self.start_state, (self.no_runs,1))

    def step(self, action):
        prev_state = self.state.clone()

        # update state for up, down, left, right
        self.state[self.runs,0] -= (action[self.runs] == 0).int()
        self.state[self.runs,0] += (action[self.runs] == 1  ).int()
        self.state[self.runs,1] -= (action[self.runs] == 2 ).int()
        self.state[self.runs,1] += (action[self.runs] == 3).int()

        # do not let agent exit grid
        self.state[self.state[self.runs, 0] > self.max_row_idx, 0] = self.max_row_idx
        self.state[self.state[self.runs, 1] > self.max_col_idx, 1] = self.max_col_idx

        self.state[self.state[self.runs, 0] < 0, 0] = 0
        self.state[self.state[self.runs, 1] < 0, 1] = 0

        # wall
        mask = self.wall[self.state[self.runs, 0], self.state[self.runs, 1]].bool()
        self.state[mask] = prev_state[mask]

        # Do not let agent leave terminal state
        terminal_mask = (prev_state[self.runs] == self.goal_state).all(dim=-1) 
        self.state[terminal_mask] = self.goal_state

        # Compute rewards
        done = (self.state[self.runs] == self.goal_state).all(dim=-1)
        mask = done &  ~terminal_mask
        reward = mask.long()


        processed_state = self.get_processed_state()
        return processed_state, reward, done
    
    def get_processed_state(self):
        # State is a number 0 - (6 * 9 - 1)
        return self.state[:, 0] * 9 + self.state[:, 1]


    def set_wall(self, index):
        self.wall[index] = True

    def reset_wall(self):
        self.wall = torch.zeros(self.gridsize, device=self.device)


    def reset(self):
        self.reset_wall()
        self.state = torch.tile(self.start_state, (self.no_runs, 1))  
        return self.start_state 
    
class DynaQ:
    def __init__(self,device, no_runs, no_states, no_actions=4 , kappa= 0.1, alpha= 0.1, epsilon= 0.1,gamma = 0.95) -> None:
        self.no_runs = no_runs
        self.runs = torch.arange(no_runs)

        self.Q = torch.zeros((no_runs,no_states,no_actions), device=device)
        
        self.kappa = kappa # unused for now
        self.alpha = alpha 
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.visited = torch.empty((no_runs, 0,2), dtype=torch.long, device=device) # state, action
        self.model = torch.full((no_runs,no_states,no_actions,2), -1, device=device) # (next state, reward)

        self.device = device

    def _select_action(self, state):
        random_actions = torch.randint(0, 4, (self.no_runs,), device=self.device)
        greedy_actions = self.Q[self.runs, state].argmax(dim=-1)
        mask = torch.rand(self.no_runs, device=self.device) < self.epsilon
        return torch.where(mask, random_actions, greedy_actions)
    
    def train(self, episodes,planning_steps,env: Gridworld):
        cumulative_reward = torch.zeros(episodes, device=self.device)
        for i in tqdm(range(episodes)):
            done = torch.zeros(self.no_runs, device=self.device)
            env.reset()
            cumulative_reward[i] += cumulative_reward[i-1]
            while not done.all():
                state = env.get_processed_state()

                action = self._select_action(state,)
                
                next_state, r, done = env.step(action)

                self.Q[self.runs,state, action] += self.alpha  * (r + self.gamma * self.Q[self.runs, next_state].max(dim=-1).values -  self.Q[self.runs, state, action])

                cumulative_reward[i] += r.float().mean()

                # Planning
                self.model[self.runs, state, action] = torch.stack([next_state, r], dim=1)
                
                new_entry = torch.stack([state, action], dim=1).unsqueeze(1)  

                self.visited = torch.cat([self.visited, new_entry], dim=1).unique(dim=1, sorted=False)

                # Sample unique states from the model
                sample_indicies = torch.randint(0, self.visited.size(1),(self.no_runs, planning_steps), device=self.device)
                entry: torch.Tensor = self.visited[self.runs.unsqueeze(1), sample_indicies] # we use self.runs.unsqueeze(1) to avoid mixing samples across runs
                state, action = entry.unbind(dim=-1)
                transition_entry = self.model[self.runs[:,None], state, action]
                next_state, r = transition_entry.unbind(-1)

                self.Q[self.runs[:,None],state, action] += self.alpha  *  \
                 (r + self.gamma * self.Q[self.runs[:,None], next_state].max(dim=-1).values - self.Q[self.runs[:,None],state, action])

        return cumulative_reward


gridsize = 6,9
no_states = gridsize[0] * gridsize[1]
no_actions = 4

no_runs = 1

no_episodes = 100
planning_steps = 5

device = "cuda" if torch.cuda.is_available() else "cpu"

env = Gridworld(device, no_runs, gridsize)

agent = DynaQ(device, no_runs,no_states,no_actions)


average_reward = agent.train(no_episodes , planning_steps, env)

import matplotlib.pyplot as plt

plt.plot(average_reward.cpu())
plt.xlabel('Episodes')
plt.ylabel('Cummulative Reward')
plt.show()

