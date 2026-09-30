import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class Gridworld:
    def __init__(self, gridsize, start_state=(5, 3), goal_state=(0, 8)) -> None:


        # --- Static ENV for simplicity ---
        self.gridsize = gridsize
        self.max_row_idx = self.gridsize[0] - 1
        self.max_col_idx = self.gridsize[1] - 1

        self.start_state = np.array(start_state, dtype=np.int64)
        self.goal_state = np.array(goal_state, dtype=np.int64)
        self.actions = np.array((0, 1, 2, 3), dtype=np.int64)

        self.wall = np.zeros(self.gridsize, dtype=bool)
        self.state = self.start_state

    def step(self, action):
        prev_state = self.state.copy()

        # update state for up, down, left, right
        self.state[0] -= (action == 0).astype(np.int64)
        self.state[0] += (action == 1).astype(np.int64)
        self.state[1] -= (action == 2).astype(np.int64)
        self.state[1] += (action == 3).astype(np.int64)

        # do not let agent exit grid
        self.state[0] = np.clip(self.state[0], 0, self.max_row_idx)
        self.state[1] = np.clip(self.state[1], 0, self.max_col_idx)

        # wall
        if self.wall[self.state[0], self.state[1]]:
            self.state = prev_state.copy()

        # Do not let agent leave terminal state
        terminated = (prev_state == self.goal_state).all()
        if terminated:
            self.state = self.goal_state

        # Compute rewards
        done = (self.state == self.goal_state).all() & ~terminated
        reward = done.astype(np.int64)
        if (reward < 0).any():
            raise ValueError("reward less than 0")

        processed_state = self.get_processed_state()
        return processed_state, reward, done

    def get_processed_state(self):
        # State is a number 0 - (6 * 9 - 1)
        return self.state[0] * self.gridsize[1] + self.state[1]

    def set_wall(self, index):
        self.wall[index] = True

    def reset_wall(self):
        self.wall = np.zeros(self.gridsize, dtype=bool)

    def reset(self):
        self.state = self.start_state.copy()
        return self.state


class DynaQ:
    def __init__(self, no_states, no_actions=4, kappa=0.1, alpha=0.1, epsilon=0.1, gamma=0.95) -> None:

        self.Q = np.zeros((no_states, no_actions), dtype=np.float64)

        self.kappa = kappa
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.visited = set()
        self.model = np.full((no_states, no_actions, 2), -1, dtype=np.int64)

    def _select_action(self, state):
        random_actions = np.random.randint(0, 4)
        greedy_actions = self.Q[state].argmax(axis=-1)
        mask = np.random.rand() < self.epsilon
        return np.where(mask, random_actions, greedy_actions)

    def train(self, max_steps, planning_steps, env: Gridworld):
        cumulative_reward = []

        for no_steps in tqdm(range(max_steps)):
            # Real experience
            state = env.get_processed_state()
            action = self._select_action(state)
            next_state, reward, done = env.step(action)

            td_target = reward + self.gamma * self.Q[next_state].max(axis=-1) * (1 - int(done))
            td_error = td_target - self.Q[state, action]
            self.Q[state, action] += self.alpha * td_error


            # Model learning
            self.model[state, action] = np.stack((next_state, reward))
            self.visited.add((int(state), int(action)))

            # Planning
            visited_entries = tuple(self.visited)
            sample_indices = np.random.randint(0, len(visited_entries), size=planning_steps)
            sampled_entries = np.array([visited_entries[idx] for idx in sample_indices], dtype=np.int64)
            sampled_states = sampled_entries[:, 0]
            sampled_actions = sampled_entries[:, 1]

            transition_entry = self.model[sampled_states, sampled_actions]
            planned_next_state = transition_entry[..., 0]
            planned_reward = transition_entry[..., 1]

            planned_td_target = planned_reward + self.gamma * self.Q[planned_next_state].max(axis=-1)
            planned_td_error = planned_td_target - self.Q[sampled_states, sampled_actions]
            self.Q[sampled_states, sampled_actions] += self.alpha * planned_td_error

            prev = cumulative_reward[-1] if cumulative_reward else 0
            cumulative_reward.append(prev + reward)

            if done:
                env.reset()

        return np.array(cumulative_reward)

    def reset(self):
        self.Q = np.zeros_like(self.Q)
        self.model.fill(-1)
        self.visited.clear()

gridsize = 6, 9
no_states = gridsize[0] * gridsize[1]
no_actions = 4
max_steps = 6000
planning_steps = 5

env = Gridworld(gridsize)
env.wall[3, 1:9] = True

agent = DynaQ(no_states, no_actions, gamma=0.8)
left_env_reward = agent.train(max_steps, planning_steps, env)


env = Gridworld(gridsize)
env.wall[3, 1:8] = True
agent.reset()
right_env_reward = agent.train(max_steps, planning_steps, env)




plt.plot(np.concat((left_env_reward,right_env_reward + left_env_reward[-1])))
plt.xlabel("Steps")
plt.ylabel("Cummulative Reward")
plt.show()
