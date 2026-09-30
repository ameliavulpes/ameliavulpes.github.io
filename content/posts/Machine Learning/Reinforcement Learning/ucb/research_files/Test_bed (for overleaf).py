import numpy as np
from tqdm import tqdm
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor

class Bandit:
    def __init__(self,Q_init, runs, steps, k, alpha):
        """
        :param Q_init: Initial action-values
        :type Q_init: int
        :param runs: Number of runs to average over
        :type runs: int
        :param steps: time steps to simulate before stopping
        :type steps: int
        :param k: Number of arms
        :type k: int
        :param alpha: Step size
        :type alpha: int
        """
        self.Q_init = Q_init
        self.q_true = np.random.normal(0, 1, (runs, k))
        self.Q = np.full((runs, k), self.Q_init, dtype=np.float32)

        self.optimal_arm = np.argmax(self.q_true, axis=1)
        self.optimal_action = np.zeros(steps)

        self.runs = runs
        self.steps = steps
        self.k = k
        self.alpha = alpha

        self.avg_rewards = np.zeros(self.steps)
        
    def train(self,c,epsilon) -> None:
        """ Run the bandit simulation using a hybrid ε-greedy + UCB action selection strategy.

        At each time step, the agent either explores with probability `epsilon`
        by selecting a random arm, or exploits by selecting the arm with the
        highest Upper Confidence Bound (UCB) score. Arms that have not yet been
        selected are forced to be tried by assigning them infinite priority in
        the argmax step.

        The method records:
            - the average reward obtained at each step (`self.rewards`)
            - the fraction of optimal actions selected (`self.optimal_action`)

        The simulation runs across `self.runs` independent bandit problems,
        each with `self.k` arms, for `self.steps` time steps.
        """

        n_action_selected = np.zeros((self.runs,self.k))

        for t in tqdm(range(self.steps)):

            explore = np.random.rand(self.runs) < epsilon

            
            picked = n_action_selected != 0

            action_scores = self.Q + c * np.sqrt(np.log(t+1)/n_action_selected)

            # If we have not picked an option, we want to pick it next
            Actions = np.argmax(np.where(picked, action_scores, np.inf), axis=-1)

            random_actions = np.random.randint(self.k, size=self.runs)
            actions = np.where(explore, random_actions, Actions)


            n_action_selected[np.arange(self.runs), actions] += 1


            rewards = np.random.normal(self.q_true[np.arange(self.runs), actions], 1)

            self.avg_rewards[t]  = rewards.mean()

            self.Q[np.arange(self.runs), actions] += self.alpha * (
                rewards - self.Q[np.arange(self.runs), actions]
            )

            self.optimal_action[t] = np.mean(actions == self.optimal_arm)
            
    
    def pickle_data(self, path):
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load_model(self,path):
        path = Path(path)
        with open(path, "rb") as f:
            self = pickle.load(f)

    def reset(self):
        """
        Reset the bandit to its initial state by reinitializing all parameters
        and statistics using the original constructor arguments.
        """
        self = Bandit(self.Q_init,self.runs,self.steps,self.k, self.alpha)


   

# probably not worth it lol, just run sequentially
if __name__ == "__main__": 

    runs = 5000
    steps = 500
    k = 10
    alpha = 0.1

    model = Bandit(Q_init=0, runs=runs, steps=steps, k = k, alpha=alpha)

    # UCB mode
    model.train(c = 2, epsilon=0)

    model.reset()

    # epsilon-grdy mode
    model.train(c=0, epsilon=0.1)



