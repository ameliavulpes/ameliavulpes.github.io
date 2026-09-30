import torch
from tqdm import tqdm


def _argmax_with_random_tie_break(values):
    noise = torch.rand_like(values) * values.abs().max() * torch.finfo(values.dtype).eps
    return (values + noise).argmax(dim=-1)


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

    def reset_index(self, index):
        self.state[index] = self.start_state

    def reset(self):
        self.state[:] = self.start_state

class DynaQ:
    def __init__(self,device, no_runs, no_states, no_actions,env: Gridworld ,kappa=0.001, alpha= 0.1, epsilon= 0.1,gamma = 0.95) -> None:
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
        greedy_actions = _argmax_with_random_tie_break(self.Q[self.runs, state])
        mask = torch.rand(self.no_runs, device=self.device) < self.epsilon
        return torch.where(mask, random_actions, greedy_actions)

    def train(self, max_steps,planning_steps):
        avg_reward = torch.zeros(max_steps, device=self.device)
        goal_state = self.env.get_processed_goal_state()


        done = torch.zeros(self.no_runs, device=self.device, dtype=torch.bool)

        for i in tqdm(range(max_steps), leave=False, desc="DynaQ"):
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
            self.env.reset_index(done)

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
        self.tau = torch.zeros(self.no_runs, self.no_states,self.no_actions,device=self.device)

        self.visited_mask = torch.zeros((no_runs, no_states, no_actions), dtype=torch.bool, device=device)

    def _select_action(self, state):
        random_actions = torch.randint(0, self.no_actions, (self.no_runs,), device=self.device)
        greedy_actions = _argmax_with_random_tie_break(self.Q[self.runs, state])
        mask = torch.rand(self.no_runs, device=self.device) < self.epsilon
        return torch.where(mask, random_actions, greedy_actions)

    def train(self, max_steps,planning_steps):
        cumulative_reward = torch.zeros(max_steps, device=self.device)
        goal_state = self.env.get_processed_goal_state()


        done = torch.zeros(self.no_runs, device=self.device, dtype=torch.bool)

        for i in tqdm(range(max_steps), leave=False, desc="DynaQ_plus_selective_sample"):

            # Real experience
            state = self.env.get_processed_state()
            action = self._select_action(state)
            next_state, r, done = self.env.step(action)

            td_target = r + (self.gamma * self.Q[self.runs, next_state].max(dim=-1).values) * ~done
            td_error = td_target - self.Q[self.runs, state, action]
            self.Q[self.runs,state, action] += self.alpha  * td_error

            # Reset tau for performed transitions
            self.tau[self.runs, state, action] = -1
            self.tau += 1

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

                bonus =  self.kappa *  self.tau[self.runs[:,None], planned_state,planned_action].sqrt()
                td_target = planned_r + bonus + (self.gamma * self.Q[self.runs[:,None], planned_next_state].max(dim=-1).values) * ~planned_done
                td_error = td_target - self.Q[self.runs[:,None], planned_state, planned_action]
                self.Q[self.runs[:,None],planned_state, planned_action] += self.alpha  * td_error


            # Reset the env for those that are done
            self.env.reset_index(done)

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
        self.tau = torch.zeros(self.no_runs, self.no_states,self.no_actions,device=self.device)


        state_indices = torch.arange(no_states, device=device)
        self.model[..., 0] = state_indices.view(1, no_states, 1) 


        self.visited_mask = torch.zeros((no_runs, no_states, no_actions), dtype=torch.bool, device=device)

    def _select_action(self, state):
        random_actions = torch.randint(0, self.no_actions, (self.no_runs,), device=self.device)
        greedy_actions = _argmax_with_random_tie_break(self.Q[self.runs, state])
        mask = torch.rand(self.no_runs, device=self.device) < self.epsilon
        return torch.where(mask, random_actions, greedy_actions)

    def train(self, max_steps,planning_steps):
        cumulative_reward = torch.zeros(max_steps, device=self.device)
        goal_state = self.env.get_processed_goal_state()

        done = torch.zeros(self.no_runs, device=self.device, dtype=torch.bool)

        for i in tqdm(range(max_steps), leave=False, desc="DynaQ_plus"):
            # Real experience
            state = self.env.get_processed_state()
            action = self._select_action(state)
            next_state, r, done = self.env.step(action)

            td_target = r + (self.gamma * self.Q[self.runs, next_state].max(dim=-1).values) * ~done
            td_error = td_target - self.Q[self.runs, state, action]
            self.Q[self.runs,state, action] += self.alpha  * td_error

            # Reset tau for performed transitions
            self.tau[self.runs, state, action] = -1
            self.tau += 1

            prev = cumulative_reward[i-1] if i > 0 else 0.0
            cumulative_reward[i] = prev + r.float().mean()

            # Model learning
            self.model[self.runs, state, action] = torch.stack([next_state, r], dim=1).float()

            self.visited_mask[self.runs, state, action] = True
            self.visited_mask[self.runs, goal_state] = False # disallow sampling from terminal state

            if self.visited_mask.any():
                # Sample unique states from the model
                planned_state = torch.randint(0, self.no_states, (self.no_runs, planning_steps))
                planned_action = torch.randint(0, self.no_actions, (self.no_runs, planning_steps))
                planned_transition = self.model[self.runs[:, None], planned_state, planned_action]
                planned_next_state = planned_transition[..., 0].long()
                planned_r = planned_transition[..., 1]

                # Planning
                planned_done = planned_next_state == goal_state

                bonus =  self.kappa *  self.tau[self.runs[:,None], planned_state,planned_action].sqrt()
                td_target = planned_r + bonus + (self.gamma * self.Q[self.runs[:,None], planned_next_state].max(dim=-1).values) * ~planned_done
                td_error = td_target - self.Q[self.runs[:,None], planned_state, planned_action]
                self.Q[self.runs[:,None],planned_state, planned_action] += self.alpha  * td_error


            # Reset the env for those that are done
            self.env.reset_index(done)

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
        greedy_values = self.Q[self.runs, state] + self.kappa *  self.tau[self.runs, state].sqrt()
        greedy_actions = _argmax_with_random_tie_break(greedy_values)
        mask = torch.rand(self.no_runs, device=self.device) < self.epsilon
        return torch.where(mask, random_actions, greedy_actions)
    def train(self, max_steps,planning_steps):
        cumulative_reward = torch.zeros(max_steps, device=self.device)
        goal_state = self.env.get_processed_goal_state()


        done = torch.zeros(self.no_runs, device=self.device, dtype=torch.bool)

        for i in tqdm(range(max_steps), leave=False, desc="DynaQ_plus_action_bonus"):
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
            self.env.reset_index(done)

        return cumulative_reward




