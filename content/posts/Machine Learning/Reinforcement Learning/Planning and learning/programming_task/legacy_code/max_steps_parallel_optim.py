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

    def get_processed_goal_state(self):
        return self.goal_state[0] * self.gridsize[1] + self.goal_state[1]


    def set_wall(self, index):
        self.wall[index] = True

    def reset_wall(self):
        self.wall = torch.zeros(self.gridsize, device=self.device)

    def reset(self, index):
        self.state[index] = self.start_state

    def reset_all(self):
        self.state[:] = self.start_state

class DynaQ:
    def __init__(self,device, no_runs, no_states, no_actions,env: Gridworld , alpha= 0.1, epsilon= 0.1,gamma = 0.95) -> None:
        self.device = device
        self.no_runs = no_runs
        self.runs = torch.arange(no_runs, device=self.device)
        self.no_actions = no_actions
        self.env = env

        self.Q = torch.zeros((no_runs,no_states,no_actions), device=device)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.model = torch.full((no_runs,no_states,no_actions,2), -1, device=device) # (next state, reward)

        self.visited_mask = torch.zeros((no_runs, no_states, no_actions), dtype=torch.bool, device=device)


    def _select_action(self, state):
        random_actions = torch.randint(0, self.no_actions, (self.no_runs,), device=self.device)
        greedy_actions = self.Q[self.runs, state].argmax(dim=-1)
        mask = torch.rand(self.no_runs, device=self.device) < self.epsilon
        return torch.where(mask, random_actions, greedy_actions)

    def train(self, max_steps,planning_steps):
        avg_reward = torch.zeros(max_steps, device=self.device)
        goal_state = self.env.get_processed_goal_state()


        done = torch.zeros(self.no_runs, device=self.device, dtype=torch.bool)

        for i in range(max_steps):


            # Real experience
            state = self.env.get_processed_state()
            action = self._select_action(state)
            next_state, r, done = self.env.step(action)

            td_target = r + (self.gamma * self.Q[self.runs, next_state].max(dim=-1).values) * ~done
            td_error = td_target - self.Q[self.runs, state, action]
            self.Q[self.runs,state, action] += self.alpha  * td_error

            prev = avg_reward[i-1] if i > 0 else 0.0
            avg_reward[i] = prev + r.float().mean()


            # Model learning
            self.model[self.runs, state, action] = torch.stack([next_state, r], dim=1)

            self.visited_mask[self.runs, state, action] = True
            self.visited_mask[self.runs, goal_state] = False # disallow sampling from terminal state

            if self.visited_mask.any():
                # Sample unique states from the model
                flat_visited = self.visited_mask.view(self.no_runs, -1).float() # use an encoding for (s,a)
                planned_idx = torch.multinomial(flat_visited, planning_steps, replacement=True)
                planned_state = planned_idx // self.Q.size(-1)
                planned_action = planned_idx % self.Q.size(-1)
                planned_transition = self.model[self.runs[:, None], planned_state, planned_action]
                planned_next_state = planned_transition[..., 0]
                r = planned_transition[..., 1]

                # Planning
                planned_done = planned_next_state == goal_state
                td_target = r + (self.gamma * self.Q[self.runs[:,None], planned_next_state].max(dim=-1).values) * ~planned_done
                td_error = td_target - self.Q[self.runs[:,None], planned_state, planned_action]
                self.Q[self.runs[:,None],planned_state, planned_action] += self.alpha  * td_error


            # Reset the env for those that are done
            self.env.reset(done)

        return avg_reward

class DynaQ_plus_selective_sample:
    def __init__(self,device, no_runs, no_states, no_actions ,env: Gridworld, kappa= 0.001, alpha= 0.1, epsilon= 0.1,gamma = 0.95) -> None:
        self.device = device
        self.no_runs = no_runs
        self.runs = torch.arange(no_runs, device=self.device)
        self.no_states = no_states
        self.no_actions = no_actions
        self.env = env

        self.Q = torch.zeros((no_runs,no_states,no_actions), device=device)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.kappa = kappa

        self.model = torch.full((no_runs,no_states,no_actions,2), -1, device=device) # (next state, reward)

        self.visited_mask = torch.zeros((no_runs, no_states, no_actions), dtype=torch.bool, device=device)

    def _select_action(self, state):
        random_actions = torch.randint(0, self.no_actions, (self.no_runs,), device=self.device)
        greedy_actions = self.Q[self.runs, state].argmax(dim=-1)
        mask = torch.rand(self.no_runs, device=self.device) < self.epsilon
        return torch.where(mask, random_actions, greedy_actions)

    def train(self, max_steps,planning_steps):
        cumulative_reward = torch.zeros(max_steps, device=self.device)
        tau = torch.zeros(self.no_runs, self.no_states,self.no_actions,device=self.device)
        goal_state = self.env.get_processed_goal_state()


        done = torch.zeros(self.no_runs, device=self.device, dtype=torch.bool)

        for i in range(max_steps):

            # Real experience
            state = self.env.get_processed_state()
            action = self._select_action(state)
            next_state, r, done = self.env.step(action)

            td_target = r + (self.gamma * self.Q[self.runs, next_state].max(dim=-1).values) * ~done
            td_error = td_target - self.Q[self.runs, state, action]
            self.Q[self.runs,state, action] += self.alpha  * td_error

            # Reset tau for performed transitions
            tau[self.runs, state, action] = -1
            tau += 1

            prev = cumulative_reward[i-1] if i > 0 else 0.0
            cumulative_reward[i] = prev + r.float().mean()

            # Model learning
            self.model[self.runs, state, action] = torch.stack([next_state, r], dim=1)

            self.visited_mask[self.runs, state, action] = True
            self.visited_mask[self.runs, goal_state] = False # disallow sampling from terminal state

            if self.visited_mask.any():
                # Sample unique states from the model
                flat_visited = self.visited_mask.view(self.no_runs, -1).float() # use an encoding for (s,a)
                planned_idx = torch.multinomial(flat_visited, planning_steps, replacement=True)
                planned_state = planned_idx // self.Q.size(-1)
                planned_action = planned_idx % self.Q.size(-1)
                planned_transition = self.model[self.runs[:, None], planned_state, planned_action]
                planned_next_state = planned_transition[..., 0]
                planned_r = planned_transition[..., 1]

                # Planning
                planned_done = planned_next_state == goal_state

                bonus =  self.kappa *  tau[self.runs[:,None], planned_state,planned_action].sqrt()
                td_target = planned_r + bonus + (self.gamma * self.Q[self.runs[:,None], planned_next_state].max(dim=-1).values) * ~planned_done
                td_error = td_target - self.Q[self.runs[:,None], planned_state, planned_action]
                self.Q[self.runs[:,None],planned_state, planned_action] += self.alpha  * td_error


            # Reset the env for those that are done
            self.env.reset(done)

        return cumulative_reward

class DynaQ_plus:
    def __init__(self,device, no_runs, no_states, no_actions ,env: Gridworld, kappa= 0.001, alpha= 0.1, epsilon= 0.1,gamma = 0.95) -> None:
        self.device = device
        self.no_runs = no_runs
        self.runs = torch.arange(no_runs, device=self.device)
        self.no_states = no_states
        self.no_actions = no_actions
        self.env = env

        self.Q = torch.zeros((no_runs,no_states,no_actions), device=device)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.kappa = kappa

        self.model = torch.zeros((no_runs,no_states,no_actions,2), device=device) # (next state, reward)

        state_indices = torch.arange(no_states, device=device)
        self.model[..., 0] = state_indices.view(1, no_states, 1) 


        self.visited_mask = torch.zeros((no_runs, no_states, no_actions), dtype=torch.bool, device=device)

    def _select_action(self, state):
        random_actions = torch.randint(0, self.no_actions, (self.no_runs,), device=self.device)
        greedy_actions = self.Q[self.runs, state].argmax(dim=-1)
        mask = torch.rand(self.no_runs, device=self.device) < self.epsilon
        return torch.where(mask, random_actions, greedy_actions)

    def train(self, max_steps,planning_steps):
        cumulative_reward = torch.zeros(max_steps, device=self.device)
        tau = torch.zeros(self.no_runs, self.no_states,self.no_actions,device=self.device)
        goal_state = self.env.get_processed_goal_state()


        done = torch.zeros(self.no_runs, device=self.device, dtype=torch.bool)

        for i in range(max_steps):

            # Real experience
            state = self.env.get_processed_state()
            action = self._select_action(state)
            next_state, r, done = self.env.step(action)

            td_target = r + (self.gamma * self.Q[self.runs, next_state].max(dim=-1).values) * ~done
            td_error = td_target - self.Q[self.runs, state, action]
            self.Q[self.runs,state, action] += self.alpha  * td_error

            # Reset tau for performed transitions
            tau[self.runs, state, action] = -1
            tau += 1

            prev = cumulative_reward[i-1] if i > 0 else 0.0
            cumulative_reward[i] = prev + r.float().mean()

            # Model learning
            self.model[self.runs, state, action] = torch.stack([next_state, r], dim=1).float()

            self.visited_mask[self.runs, state, action] = True
            self.visited_mask[self.runs, goal_state] = False # disallow sampling from terminal state

            if self.visited_mask.any():
                # Sample unique states from the model
                planned_state = torch.randint(0, self.no_states, (no_runs, planning_steps))
                planned_action = torch.randint(0, self.no_actions, (no_runs, planning_steps))
                planned_transition = self.model[self.runs[:, None], planned_state, planned_action]
                planned_next_state = planned_transition[..., 0].long()
                planned_r = planned_transition[..., 1]

                # Planning
                planned_done = planned_next_state == goal_state

                bonus =  self.kappa *  tau[self.runs[:,None], planned_state,planned_action].sqrt()
                td_target = planned_r + bonus + (self.gamma * self.Q[self.runs[:,None], planned_next_state].max(dim=-1).values) * ~planned_done
                td_error = td_target - self.Q[self.runs[:,None], planned_state, planned_action]
                self.Q[self.runs[:,None],planned_state, planned_action] += self.alpha  * td_error


            # Reset the env for those that are done
            self.env.reset(done)

        return cumulative_reward

class DynaQ_plus_action_bonus:
    def __init__(self,device, no_runs, no_states, no_actions ,env: Gridworld, kappa= 0.001, alpha= 0.1, epsilon= 0.1,gamma = 0.95) -> None:
        self.device = device
        self.no_runs = no_runs
        self.runs = torch.arange(no_runs, device=self.device)
        self.no_states = no_states
        self.no_actions = no_actions
        self.env = env

        self.Q = torch.zeros((no_runs,no_states,no_actions), device=device)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.kappa = kappa

        self.model = torch.zeros((no_runs,no_states,no_actions,2), device=device) # (next state, reward)

        state_indices = torch.arange(no_states, device=device)
        self.model[..., 0] = state_indices.view(1, no_states, 1) 


        self.visited_mask = torch.zeros((no_runs, no_states, no_actions), dtype=torch.bool, device=device)
        self.tau = torch.zeros(self.no_runs, self.no_states,self.no_actions,device=self.device)

    def _select_action(self, state):
        random_actions = torch.randint(0, self.no_actions, (self.no_runs,), device=self.device)
        greedy_actions = (self.Q[self.runs, state] + self.kappa *  self.tau[self.runs, state].sqrt()).argmax(dim=-1)
        mask = torch.rand(self.no_runs, device=self.device) < self.epsilon
        return torch.where(mask, random_actions, greedy_actions)
    def train(self, max_steps,planning_steps):
        cumulative_reward = torch.zeros(max_steps, device=self.device)
        goal_state = self.env.get_processed_goal_state()


        done = torch.zeros(self.no_runs, device=self.device, dtype=torch.bool)

        for i in tqdm(range(max_steps)):
            # Real experience
            state = self.env.get_processed_state()
            action = self._select_action(state)
            next_state, r, done = self.env.step(action)

            td_target = r + (self.gamma * self.Q[self.runs, next_state].max(dim=-1).values) * ~done
            td_error = td_target - self.Q[self.runs, state, action]
            self.Q[self.runs,state, action] += self.alpha  * td_error

            prev = cumulative_reward[i-1] if i > 0 else 0.0
            cumulative_reward[i] = prev + r.float().mean()

            # Reset tau for performed transitions
            self.tau[self.runs, state, action] = -1
            self.tau += 1

            # Model learning
            self.model[self.runs, state, action] = torch.stack([next_state, r], dim=1).float()

            self.visited_mask[self.runs, state, action] = True
            self.visited_mask[self.runs, goal_state] = False # disallow sampling from terminal state

            if self.visited_mask.any():
                # Sample unique states from the model
                flat_visited = self.visited_mask.view(self.no_runs, -1).float() # use an encoding for (s,a)
                planned_idx = torch.multinomial(flat_visited, planning_steps, replacement=True)
                planned_state = planned_idx // self.Q.size(-1)
                planned_action = planned_idx % self.Q.size(-1)
                planned_transition = self.model[self.runs[:, None], planned_state, planned_action]
                planned_next_state = planned_transition[..., 0].long()
                r = planned_transition[..., 1]

                # Planning
                planned_done = planned_next_state == goal_state
                td_target = r + (self.gamma * self.Q[self.runs[:,None], planned_next_state].max(dim=-1).values) * ~planned_done
                td_error = td_target - self.Q[self.runs[:,None], planned_state, planned_action]
                self.Q[self.runs[:,None],planned_state, planned_action] += self.alpha  * td_error


            # Reset the env for those that are done
            self.env.reset(done)

        return cumulative_reward


gridsize = 6, 9
no_states = gridsize[0] * gridsize[1]
no_actions = 4
max_steps_envA = 3000
no_runs = 20
max_steps_envA = 5000
max_steps_envB = 5000
planning_steps = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(device)

env = Gridworld(device, no_runs, gridsize, goal_state=(0, 8))

def set_wall_a(env):
    env.reset_all()
    env.reset_wall()
    env.wall[3, 1:9] = True

def set_wall_b(env):
    env.reset_all()
    env.reset_wall()
    env.wall[3, 1:8] = True

def run_and_plot(env, name):
    # DynaQ plus action bonus
    print("Working on: DynaQ plus action bonus")
    agent = DynaQ_plus_action_bonus(device,no_runs, no_states, no_actions, env, gamma=0.8)
    set_wall_a(agent.env)
    dynaQ_plus_act_a = agent.train(max_steps_envA, planning_steps)
    set_wall_b(agent.env)
    dynaQ_plus_act_b = agent.train(max_steps_envB, planning_steps)

    # DynaQ plus
    print("Working on: DynaQ plus")
    agent = DynaQ_plus(device,no_runs, no_states, no_actions, env, gamma=0.8)
    set_wall_a(agent.env)
    dynaQ_plus_a = agent.train(max_steps_envA, planning_steps)
    set_wall_b(agent.env)
    dynaQ_plus_b = agent.train(max_steps_envB, planning_steps)

    # DynaQ 
    print("Working on: DynaQ ")
    agent = DynaQ(device,no_runs, no_states, no_actions, env, gamma=0.8)
    set_wall_a(agent.env)
    dynaQ_left_env_r = agent.train(max_steps_envA, planning_steps)
    set_wall_b(agent.env)
    dynaQ_right_env_r = agent.train(max_steps_envB, planning_steps)

    # DynaQ plus selective
    print("Working on: DynaQ plus selective")
    agent = DynaQ_plus_selective_sample(device,no_runs, no_states, no_actions, env, gamma=0.8)
    set_wall_a(agent.env)
    dynaQ_plus_left_env_r = agent.train(max_steps_envA, planning_steps)
    set_wall_b(agent.env)
    dynaQ_plus_right_env_r = agent.train(max_steps_envB, planning_steps)


    styles = [('default', 'light_imgs'), ('dark_background', 'dark_imgs')]
        
    for i, (style, folder) in enumerate(styles):
        plt.style.use(style)
        plt.figure(figsize=(12, 4.5))
        
        # Plotting logic
        plt.plot(torch.concat((dynaQ_left_env_r,dynaQ_right_env_r + dynaQ_left_env_r[-1])).cpu(), label="DynaQ")
        plt.plot(torch.concat((dynaQ_plus_left_env_r,dynaQ_plus_right_env_r + dynaQ_plus_left_env_r[-1])).cpu(), label="DynaQ+ Selective")
        plt.plot(torch.concat((dynaQ_plus_a,dynaQ_plus_b + dynaQ_plus_a[-1])).cpu(), label="DynaQ+")
        plt.plot(torch.concat((dynaQ_plus_act_a,dynaQ_plus_act_b + dynaQ_plus_act_a[-1])).cpu(), label="DynaQ+ Action Bonus")
        
        plt.xlabel("Steps", fontsize=18)
        plt.ylabel("Cumulative Reward", fontsize=18)
        plt.legend(fontsize=16)
        
        save_path = f"content/posts/Reinforcement Learning/Planning and learning/programming_task/{folder}/{name}.svg"
        plt.savefig(save_path, bbox_inches="tight", transparent=True)

        if i == len(styles) -1:
            plt.show()

        plt.close() # Close after saving each version





run_and_plot(env, "less_planning")

