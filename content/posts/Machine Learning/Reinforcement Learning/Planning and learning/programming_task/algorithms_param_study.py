import torch
from tqdm import tqdm


def _as_param_tensor(value, no_params, device, dtype):
    if torch.is_tensor(value):
        tensor = value.to(device=device, dtype=dtype).reshape(-1)
    else:
        tensor = torch.tensor(value, device=device, dtype=dtype).reshape(-1)

    if tensor.numel() == 1:
        return tensor.repeat(no_params)
    if tensor.numel() != no_params:
        raise ValueError(f"Expected 1 or {no_params} values, got {tensor.numel()}")
    return tensor


def _argmax_with_random_tie_break(values):
    noise = torch.rand_like(values) * values.abs().max() * torch.finfo(values.dtype).eps
    return (values + noise).argmax(dim=-1)


class Gridworld:
    def __init__(
        self,
        device,
        no_runs,
        no_params,
        gridsize,
        start_state=(5, 3),
        goal_state=(0, 8),
    ) -> None:
        self.device = device
        self.no_runs = no_runs
        self.no_params = no_params
        self.param_idx = torch.arange(no_params, device=device)[:, None]
        self.run_idx = torch.arange(no_runs, device=device)[None, :]

        self.gridsize = gridsize
        self.max_row_idx = gridsize[0] - 1
        self.max_col_idx = gridsize[1] - 1

        self.start_state = torch.tensor(start_state, device=device)
        self.goal_state = torch.tensor(goal_state, device=device)

        self.wall = torch.zeros((no_params, *gridsize), device=device, dtype=torch.bool)
        self.state = torch.tile(self.start_state, (no_params, no_runs, 1))

    def step(self, action):
        prev_state = self.state.clone()

        self.state[..., 0] -= (action == 0).int()
        self.state[..., 0] += (action == 1).int()
        self.state[..., 1] -= (action == 2).int()
        self.state[..., 1] += (action == 3).int()

        self.state[..., 0].clamp_(0, self.max_row_idx)
        self.state[..., 1].clamp_(0, self.max_col_idx)

        wall_mask = self.wall[
            self.param_idx,
            self.state[..., 0],
            self.state[..., 1],
        ]
        self.state[wall_mask] = prev_state[wall_mask]

        terminal_mask = (prev_state == self.goal_state).all(dim=-1)
        self.state[terminal_mask] = self.goal_state

        done = (self.state == self.goal_state).all(dim=-1)
        reward = (done & ~terminal_mask).long()

        return self.get_processed_state(), reward, done

    def get_processed_state(self):
        return self.state[..., 0] * self.gridsize[1] + self.state[..., 1]

    def get_processed_goal_state(self):
        return self.goal_state[0] * self.gridsize[1] + self.goal_state[1]

    def reset_wall(self):
        self.wall.zero_()

    def reset_index(self, index):
        self.state[index] = self.start_state

    def reset(self):
        self.state[:] = self.start_state


def set_wall_a(env):
    env.reset_wall()
    env.wall[:, 3, 1:9] = True


def set_wall_b(env):
    env.reset_wall()
    env.wall[:, 3, 1:8] = True


class _BaseDynaQ:
    def __init__(self,device,no_runs,no_params,no_states,no_actions,env: Gridworld,kappa=0.001,alpha=0.1,epsilon=0.1,gamma=0.95,) -> None:
        self.device = device
        self.no_runs = no_runs
        self.no_params = no_params
        self.no_states = no_states
        self.no_actions = no_actions
        self.env = env

        self.param_idx = torch.arange(no_params, device=device)[:, None]
        self.run_idx = torch.arange(no_runs, device=device)[None, :]

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.kappa = _as_param_tensor(kappa, no_params, device, torch.float32).view(no_params, 1, 1)

        self.Q = torch.zeros((no_params, no_runs, no_states, no_actions), device=device)
        self.visited_mask = torch.zeros(
            (no_params, no_runs, no_states, no_actions),
            dtype=torch.bool,
            device=device,
        )

    def _select_action(self, state):
        random_actions = torch.randint(0,
            self.no_actions,
            (self.no_params, self.no_runs),
            device=self.device,
        )
        greedy_actions = _argmax_with_random_tie_break(
            self.Q[self.param_idx, self.run_idx, state]
        )
        mask = torch.rand(self.no_params, self.no_runs, device=self.device) < self.epsilon
        return torch.where(mask, random_actions, greedy_actions)

    def _planning_mask(self, planning_steps):
        planning_steps = _as_param_tensor(
            planning_steps, self.no_params, self.device, torch.long
        )
        max_planning_steps = int(planning_steps.max().item())
        planning_range = torch.arange(max_planning_steps, device=self.device)
        active_mask = planning_range[None, :] < planning_steps[:, None]
        return max_planning_steps, active_mask[:, None, :]


class DynaQ(_BaseDynaQ):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = torch.full(
            (self.no_params, self.no_runs, self.no_states, self.no_actions, 2),
            -1,
            device=self.device,
            dtype=torch.int8,
        )

    def train(self, max_steps, planning_steps):
        cumulative_reward = torch.zeros((self.no_params, max_steps), device=self.device)
        goal_state = self.env.get_processed_goal_state()
        max_planning_steps, planning_mask = self._planning_mask(planning_steps)

        done = torch.zeros((self.no_params, self.no_runs), device=self.device, dtype=torch.bool)

        for i in tqdm(range(max_steps), leave=False, desc="DynaQ Param Study"):
            state = self.env.get_processed_state()
            action = self._select_action(state)
            next_state, reward, done = self.env.step(action)

            next_q = self.Q[self.param_idx, self.run_idx, next_state].max(dim=-1).values
            td_target = reward.float() + self.gamma * next_q * ~done
            td_error = td_target - self.Q[self.param_idx, self.run_idx, state, action]
            self.Q[self.param_idx, self.run_idx, state, action] += self.alpha * td_error

            prev = cumulative_reward[:, i - 1] if i > 0 else 0.0
            cumulative_reward[:, i] = prev + reward.float().mean(dim=1)

            self.model[self.param_idx, self.run_idx, state, action] = torch.stack(
                [next_state, reward], dim=-1
            ).to(torch.int8)

            self.visited_mask[self.param_idx, self.run_idx, state, action] = True
            self.visited_mask[:, :, goal_state] = False

            if max_planning_steps > 0 and self.visited_mask.any():
                flat_visited = self.visited_mask.view(self.no_params * self.no_runs, -1).float()
                planned_idx = torch.multinomial(
                    flat_visited, max_planning_steps, replacement=True
                ).view(self.no_params, self.no_runs, max_planning_steps)

                planned_state = planned_idx // self.no_actions
                planned_action = planned_idx % self.no_actions
                planned_transition = self.model[
                    self.param_idx[:, :, None],
                    self.run_idx[:, :, None],
                    planned_state,
                    planned_action,
                ]
                planned_next_state = planned_transition[..., 0].long()
                planned_reward = planned_transition[..., 1].float()

                planned_done = planned_next_state == goal_state
                next_q = self.Q[
                    self.param_idx[:, :, None],
                    self.run_idx[:, :, None],
                    planned_next_state,
                ].max(dim=-1).values
                td_target = planned_reward + self.gamma * next_q * ~planned_done
                td_error = td_target - self.Q[
                    self.param_idx[:, :, None],
                    self.run_idx[:, :, None],
                    planned_state,
                    planned_action,
                ]
                self.Q[
                    self.param_idx[:, :, None],
                    self.run_idx[:, :, None],
                    planned_state,
                    planned_action,
                ] += self.alpha * td_error * planning_mask

            self.env.reset_index(done)

        return cumulative_reward


class DynaQ_plus_selective_sample(_BaseDynaQ):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = torch.full(
            (self.no_params, self.no_runs, self.no_states, self.no_actions, 2),
            -1,
            device=self.device,
            dtype=torch.int8,
        )
        self.tau = torch.zeros(
            (self.no_params, self.no_runs, self.no_states, self.no_actions),
            device=self.device,
        )

    def train(self, max_steps, planning_steps):
        cumulative_reward = torch.zeros((self.no_params, max_steps), device=self.device)
        goal_state = self.env.get_processed_goal_state()
        max_planning_steps, planning_mask = self._planning_mask(planning_steps)

        done = torch.zeros((self.no_params, self.no_runs), device=self.device, dtype=torch.bool)

        for i in tqdm(range(max_steps), leave=False, desc="DynaQ+ Selective Param Study"):
            state = self.env.get_processed_state()
            action = self._select_action(state)
            next_state, reward, done = self.env.step(action)

            next_q = self.Q[self.param_idx, self.run_idx, next_state].max(dim=-1).values
            td_target = reward.float() + self.gamma * next_q * ~done
            td_error = td_target - self.Q[self.param_idx, self.run_idx, state, action]
            self.Q[self.param_idx, self.run_idx, state, action] += self.alpha * td_error

            self.tau[self.param_idx, self.run_idx, state, action] = -1
            self.tau += 1

            prev = cumulative_reward[:, i - 1] if i > 0 else 0.0
            cumulative_reward[:, i] = prev + reward.float().mean(dim=1)

            self.model[self.param_idx, self.run_idx, state, action] = torch.stack(
                [next_state, reward], dim=-1
            ).to(torch.int8)

            self.visited_mask[self.param_idx, self.run_idx, state, action] = True
            self.visited_mask[:, :, goal_state] = False

            if max_planning_steps > 0 and self.visited_mask.any():
                flat_visited = self.visited_mask.view(self.no_params * self.no_runs, -1).float()
                planned_idx = torch.multinomial(
                    flat_visited, max_planning_steps, replacement=True
                ).view(self.no_params, self.no_runs, max_planning_steps)

                planned_state = planned_idx // self.no_actions
                planned_action = planned_idx % self.no_actions
                planned_transition = self.model[
                    self.param_idx[:, :, None],
                    self.run_idx[:, :, None],
                    planned_state,
                    planned_action,
                ]
                planned_next_state = planned_transition[..., 0].long()
                planned_reward = planned_transition[..., 1].float()

                planned_done = planned_next_state == goal_state
                bonus = self.kappa * self.tau[
                    self.param_idx[:, :, None],
                    self.run_idx[:, :, None],
                    planned_state,
                    planned_action,
                ].sqrt()
                next_q = self.Q[
                    self.param_idx[:, :, None],
                    self.run_idx[:, :, None],
                    planned_next_state,
                ].max(dim=-1).values
                td_target = planned_reward + bonus + self.gamma * next_q * ~planned_done
                td_error = td_target - self.Q[
                    self.param_idx[:, :, None],
                    self.run_idx[:, :, None],
                    planned_state,
                    planned_action,
                ]
                self.Q[
                    self.param_idx[:, :, None],
                    self.run_idx[:, :, None],
                    planned_state,
                    planned_action,
                ] += self.alpha * td_error * planning_mask

            self.env.reset_index(done)

        return cumulative_reward


class DynaQ_plus(_BaseDynaQ):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = torch.zeros(
            (self.no_params, self.no_runs, self.no_states, self.no_actions, 2),
            device=self.device,
            dtype=torch.int8,
        )
        self.tau = torch.zeros(
            (self.no_params, self.no_runs, self.no_states, self.no_actions),
            device=self.device,
        )

        state_indices = torch.arange(self.no_states, device=self.device, dtype=torch.int8)
        self.model[..., 0] = state_indices.view(1, 1, self.no_states, 1)

    def train(self, max_steps, planning_steps):
        cumulative_reward = torch.zeros((self.no_params, max_steps), device=self.device)
        goal_state = self.env.get_processed_goal_state()
        max_planning_steps, planning_mask = self._planning_mask(planning_steps)

        done = torch.zeros((self.no_params, self.no_runs), device=self.device, dtype=torch.bool)

        for i in tqdm(range(max_steps), leave=False, desc="DynaQ+ Param Study"):
            state = self.env.get_processed_state()
            action = self._select_action(state)
            next_state, reward, done = self.env.step(action)

            next_q = self.Q[self.param_idx, self.run_idx, next_state].max(dim=-1).values
            td_target = reward.float() + self.gamma * next_q * ~done
            td_error = td_target - self.Q[self.param_idx, self.run_idx, state, action]
            self.Q[self.param_idx, self.run_idx, state, action] += self.alpha * td_error

            self.tau[self.param_idx, self.run_idx, state, action] = -1
            self.tau += 1

            prev = cumulative_reward[:, i - 1] if i > 0 else 0.0
            cumulative_reward[:, i] = prev + reward.float().mean(dim=1)

            self.model[self.param_idx, self.run_idx, state, action] = torch.stack(
                [next_state, reward], dim=-1
            ).to(torch.int8)

            self.visited_mask[self.param_idx, self.run_idx, state, action] = True
            self.visited_mask[:, :, goal_state] = False

            if max_planning_steps > 0:
                planned_state = torch.randint(
                    0,
                    self.no_states,
                    (self.no_params, self.no_runs, max_planning_steps),
                    device=self.device,
                )
                planned_action = torch.randint(
                    0,
                    self.no_actions,
                    (self.no_params, self.no_runs, max_planning_steps),
                    device=self.device,
                )
                planned_transition = self.model[
                    self.param_idx[:, :, None],
                    self.run_idx[:, :, None],
                    planned_state,
                    planned_action,
                ]
                planned_next_state = planned_transition[..., 0].long()
                planned_reward = planned_transition[..., 1]

                planned_done = planned_next_state == goal_state
                bonus = self.kappa * self.tau[
                    self.param_idx[:, :, None],
                    self.run_idx[:, :, None],
                    planned_state,
                    planned_action,
                ].sqrt()
                next_q = self.Q[
                    self.param_idx[:, :, None],
                    self.run_idx[:, :, None],
                    planned_next_state,
                ].max(dim=-1).values
                td_target = planned_reward + bonus + self.gamma * next_q * ~planned_done
                td_error = td_target - self.Q[
                    self.param_idx[:, :, None],
                    self.run_idx[:, :, None],
                    planned_state,
                    planned_action,
                ]
                self.Q[
                    self.param_idx[:, :, None],
                    self.run_idx[:, :, None],
                    planned_state,
                    planned_action,
                ] += self.alpha * td_error * planning_mask

            self.env.reset_index(done)

        return cumulative_reward


class DynaQ_plus_action_bonus(_BaseDynaQ):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = torch.zeros(
            (self.no_params, self.no_runs, self.no_states, self.no_actions, 2),
            device=self.device,
            dtype=torch.int8,
        )
        self.tau = torch.zeros(
            (self.no_params, self.no_runs, self.no_states, self.no_actions),
            device=self.device,
        )

        state_indices = torch.arange(self.no_states, device=self.device, dtype=torch.int8)
        self.model[..., 0] = state_indices.view(1, 1, self.no_states, 1)

    def _select_action(self, state):
        random_actions = torch.randint(
            0,
            self.no_actions,
            (self.no_params, self.no_runs),
            device=self.device,
        )
        action_bonus = self.kappa * self.tau[self.param_idx, self.run_idx, state].sqrt()
        greedy_values = self.Q[self.param_idx, self.run_idx, state] + action_bonus
        greedy_actions = _argmax_with_random_tie_break(greedy_values)
        mask = torch.rand(self.no_params, self.no_runs, device=self.device) < self.epsilon
        return torch.where(mask, random_actions, greedy_actions)

    def train(self, max_steps, planning_steps):
        cumulative_reward = torch.zeros((self.no_params, max_steps), device=self.device)
        goal_state = self.env.get_processed_goal_state()
        max_planning_steps, planning_mask = self._planning_mask(planning_steps)

        done = torch.zeros((self.no_params, self.no_runs), device=self.device, dtype=torch.bool)

        for i in tqdm(range(max_steps), leave=False, desc="DynaQ+ Action Bonus Param Study"):
            state = self.env.get_processed_state()
            action = self._select_action(state)
            next_state, reward, done = self.env.step(action)

            next_q = self.Q[self.param_idx, self.run_idx, next_state].max(dim=-1).values
            td_target = reward.float() + self.gamma * next_q * ~done
            td_error = td_target - self.Q[self.param_idx, self.run_idx, state, action]
            self.Q[self.param_idx, self.run_idx, state, action] += self.alpha * td_error

            prev = cumulative_reward[:, i - 1] if i > 0 else 0.0
            cumulative_reward[:, i] = prev + reward.float().mean(dim=1)

            self.tau[self.param_idx, self.run_idx, state, action] = -1
            self.tau += 1

            self.model[self.param_idx, self.run_idx, state, action] = torch.stack(
                [next_state, reward], dim=-1
            ).to(torch.int8)

            self.visited_mask[self.param_idx, self.run_idx, state, action] = True
            self.visited_mask[:, :, goal_state] = False

            if max_planning_steps > 0 and self.visited_mask.any():
                flat_visited = self.visited_mask.view(self.no_params * self.no_runs, -1).float()
                planned_idx = torch.multinomial(
                    flat_visited, max_planning_steps, replacement=True
                ).view(self.no_params, self.no_runs, max_planning_steps)

                planned_state = planned_idx // self.no_actions
                planned_action = planned_idx % self.no_actions
                planned_transition = self.model[
                    self.param_idx[:, :, None],
                    self.run_idx[:, :, None],
                    planned_state,
                    planned_action,
                ]
                planned_next_state = planned_transition[..., 0].long()
                planned_reward = planned_transition[..., 1]

                planned_done = planned_next_state == goal_state
                next_q = self.Q[
                    self.param_idx[:, :, None],
                    self.run_idx[:, :, None],
                    planned_next_state,
                ].max(dim=-1).values
                td_target = planned_reward + self.gamma * next_q * ~planned_done
                td_error = td_target - self.Q[
                    self.param_idx[:, :, None],
                    self.run_idx[:, :, None],
                    planned_state,
                    planned_action,
                ]
                self.Q[
                    self.param_idx[:, :, None],
                    self.run_idx[:, :, None],
                    planned_state,
                    planned_action,
                ] += self.alpha * td_error * planning_mask

            self.env.reset_index(done)

        return cumulative_reward
