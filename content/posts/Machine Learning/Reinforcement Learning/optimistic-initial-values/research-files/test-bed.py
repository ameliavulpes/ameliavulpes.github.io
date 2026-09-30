import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
runs = 5_000
steps = 2000
k = 10
alpha = 0.1
epsilon = 0.1

def run_bandit(Q_init, epsilon):

    q_true = np.random.normal(0, 1, (runs, k))
    Q = np.full((runs, k), Q_init, dtype=np.float32)

    optimal_arm = np.argmax(q_true, axis=1)
    optimal_action = np.zeros(steps)

    for t in tqdm(range(steps)):

        explore = np.random.rand(runs) < epsilon

        greedy_actions = np.argmax(Q, axis=1)
        random_actions = np.random.randint(k, size=runs)

        actions = np.where(explore, random_actions, greedy_actions)

        rewards = np.random.normal(q_true[np.arange(runs), actions], 1)

        Q[np.arange(runs), actions] += alpha * (
            rewards - Q[np.arange(runs), actions]
        )

        optimal_action[t] = np.mean(actions == optimal_arm)

        # if epsilon != 0:
        #     return

        # if 8 < t < 13:
        #     print(t)
        #     optimal_actions = np.argmax(Q, axis=1)
        #     is_optimal = optimal_actions == optimal_arm
        #     print(f"Mean optimal action: {Q[np.arange(runs), optimal_arm].mean()}")
        #     print(f"Mean non-optimal action: {Q[~is_optimal].mean()}")
        # if t > 20:
        #     return

    return optimal_action


realistic = run_bandit(Q_init=0, epsilon=0.1)
optimistic = run_bandit(Q_init=5.0, epsilon=0)

with open('content/posts/sutton-barto/Optimistic-Initial-Values/realistic.pkl', 'wb') as f:
    pickle.dump(realistic, f)

with open('content/posts/sutton-barto/Optimistic-Initial-Values/optimistic.pkl', 'wb') as f:
    pickle.dump(optimistic, f)

