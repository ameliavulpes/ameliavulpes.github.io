import torch
from tqdm import tqdm
import matplotlib.pyplot as plt



class Gridworld:
    def __init__(self, device, no_runs, gridsize, start_state=(5, 3), goal_state=(0, 8)) -> None:
        self.device = device
        self.no_runs = no_runs
        self.runs = torch.arange(self.no_runs, device=self.device)

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
        return self.state[:, 0] * self.gridsize[1] + self.state[:, 1]


    def set_wall(self, index):
        self.wall[index] = True

    def reset_wall(self):
        self.wall = torch.zeros(self.gridsize, device=self.device)

    def reset(self, index):
        self.state[index] = torch.tile(self.start_state, (self.no_runs, 1))  
        return self.start_state 
    
class DynaQ:
    def __init__(self,device, no_runs, no_states, no_actions=4 , kappa= 0.1, alpha= 0.1, epsilon= 0.1,gamma = 0.95) -> None:
        self.device = device
        self.no_runs = no_runs
        self.runs = torch.arange(no_runs, device=self.device)

        self.Q = torch.zeros((no_runs,no_states,no_actions), device=device)
        
        self.kappa = kappa # unused for now
        self.alpha = alpha 
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.visited = torch.empty((no_runs, 0,2), dtype=torch.long, device=device) # state, action
        self.model = torch.full((no_runs,no_states,no_actions,2), -1, device=device) # (next state, reward)


    def _select_action(self, state):
        random_actions = torch.randint(0, 4, (self.no_runs,), device=self.device)
        greedy_actions = self.Q[self.runs, state].argmax(dim=-1)
        mask = torch.rand(self.no_runs, device=self.device) < self.epsilon
        return torch.where(mask, random_actions, greedy_actions)
    
    def train(self, max_steps,planning_steps,env: Gridworld):
        cumulative_reward = torch.zeros(max_steps, device=self.device)

        done = torch.zeros(self.no_runs, device=self.device, dtype=torch.bool)
        for i in tqdm(range(max_steps)):

       
            # Real experience
            state = env.get_processed_state()
            action = self._select_action(state)
            next_state, r, done = env.step(action)

            td_target = r + (self.gamma * self.Q[self.runs, next_state].max(dim=-1).values) * ~done
            td_error = td_target - self.Q[self.runs, state, action]
            self.Q[self.runs,state, action] += self.alpha  * td_error 

            cumulative_reward[i] = cumulative_reward[i-1] + r.float().mean()


            # Model learning
            self.model[self.runs, state, action] = torch.stack([next_state, r], dim=1)
            
            new_entry = torch.stack([state, action], dim=1).unsqueeze(1)  

            self.visited = torch.cat([self.visited, new_entry], dim=1).unique(dim=1, sorted=False)

            # Sample unique states from the model
            sample_indicies = torch.randint(0, self.visited.size(1),(self.no_runs, planning_steps), device=self.device)
            entry: torch.Tensor = self.visited[self.runs.unsqueeze(1), sample_indicies] # we use self.runs.unsqueeze(1) to avoid mixing samples across runs
            planned_state, planned_action = entry.unbind(dim=-1)
            transition_entry = self.model[self.runs[:,None], planned_state, planned_action]
            planned_next_state, r = transition_entry.unbind(-1)

            # Planning    
            td_target = r + self.gamma * self.Q[self.runs[:,None], planned_next_state].max(dim=-1).values
            td_error = td_target - self.Q[self.runs[:,None], planned_state, planned_action]
            self.Q[self.runs[:,None],planned_state, planned_action] += self.alpha  * td_error


            # Reset the env for the guys that are done
            env.reset(done)
                
        return cumulative_reward




gridsize = 6, 9
no_states = gridsize[0] * gridsize[1]
no_actions = 4
max_steps = 3000
no_runs = 10
planning_steps = 5
device = "cuda" if torch.cuda.is_available() else "cpu"


env = Gridworld(device, no_runs, gridsize)
env.wall[3, 1:9] = True

agent = DynaQ(device,no_runs, no_states, no_actions, gamma=0.8)
left_env_reward = agent.train(max_steps, planning_steps, env)


env.reset_wall()
env.wall[3, 1:8] = True
agent = DynaQ(device,no_runs, no_states, no_actions, gamma=0.8)
right_env_reward = agent.train(max_steps, planning_steps, env)




plt.plot(torch.concat((left_env_reward,right_env_reward + left_env_reward[-1])))
plt.xlabel("Steps")
plt.ylabel("Cummulative Reward")
plt.show()
