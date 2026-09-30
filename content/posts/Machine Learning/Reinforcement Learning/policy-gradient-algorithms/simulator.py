import numpy as np
import torch
from tqdm import tqdm
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Self, Optional, Dict


class Bandit:
    def __init__(self,mean, H_init, runs, steps, k, alpha, device = "cpu"):
        """
        :param mean: Mean of distribution which each arm's expected value is drawn from  
        :type mean: int
        :param H_init: Initial action-prefrence
        :type H_init: int
        :param runs: Number of runs to average over
        :type runs: int
        :param steps: time steps to simulate before stopping
        :type steps: int
        :param k: Number of arms
        :type k: int
        :param alpha: Step size
        :type alpha: int
        """
        self.device = device

        self.H_init = H_init
        self.mean = mean
        self.q_true = torch.normal(self.mean, 1, (runs, k), dtype=torch.float32, device=self.device)
        self.H = torch.full((runs, k), self.H_init, dtype=torch.float32, device=self.device)

        self.optimal_arm = torch.argmax(self.q_true, dim=1)
        self.optimal_action = torch.zeros(steps, device=self.device)

        self.runs = runs
        self.steps = steps
        self.k = k
        self.alpha = alpha        

        self.avg_rewards = torch.zeros(self.steps, device=self.device)
        
    def train(self) -> Self:
        """
        The method records:
            - the average reward obtained at each step (`self.rewards`)
            - the fraction of optimal actions selected (`self.optimal_action`)

        The simulation runs across `self.runs` independent bandit problems,
        each with `self.k` arms, for `self.steps` time steps.
        """

        print(f"Running on {self.device}")
        idx = torch.arange(self.runs, device=self.device)
        avg_rewards = torch.zeros(self.runs, device=self.device)

        for t in tqdm(range(self.steps)):

            probs = torch.softmax(self.H, dim=1)
            actions = torch.distributions.Categorical(probs).sample()

            rewards = torch.normal(self.q_true[idx, actions], 1)

            advantage = rewards - avg_rewards

            self.H -= self.alpha * advantage.unsqueeze(1) * probs
            self.H[idx, actions] += self.alpha * advantage

            avg_rewards += (rewards - avg_rewards) / (t + 1)

            self.avg_rewards[t]  = rewards.mean()
            self.optimal_action[t] = torch.mean((actions == self.optimal_arm).float())

        return self
    
    def train_without_baseline(self) -> Self:
        """
        The method records:
            - the average reward obtained at each step (`self.rewards`)
            - the fraction of optimal actions selected (`self.optimal_action`)

        The simulation runs across `self.runs` independent bandit problems,
        each with `self.k` arms, for `self.steps` time steps.
        """

        print(f"Running on {self.device}")
        idx = torch.arange(self.runs, device=self.device)

        for t in tqdm(range(self.steps)):

            probs = torch.softmax(self.H, dim=1)
            actions = torch.distributions.Categorical(probs).sample()

            rewards = torch.normal(self.q_true[idx, actions], 1)

            advantage = rewards 

            self.H -= self.alpha * advantage.unsqueeze(1) * probs
            self.H[idx, actions] += self.alpha * advantage

            self.avg_rewards[t]  = rewards.mean()
            self.optimal_action[t] = torch.mean((actions == self.optimal_arm).float())

        return self
            
    
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
        self = Bandit(self.H_init,self.mean,self.runs,self.steps,self.k, self.alpha)


def plot_and_save(
    data: Dict[str, list], 
    xlabel: str = "Steps", 
    ylabel: str = "y", 
    filename: Optional[str] = None, 
    show: bool = True
):
    """
    Plots multiple datasets and optionally saves them in light and dark styles.

    Parameters:
    -----------
    data : dict
        Dictionary where keys are labels and values are lists of data points.
        Example: {'experiment1': [1,2,3], 'experiment2': [4,5,6]}
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    filename : str, optional
        Base filename to save the plots. Saved in both light and dark styles.
    show : bool
        Whether to display the plot.
    """

    def _plot(style: str):
        """Internal function to plot the data with a given style."""
        plt.figure(figsize=(12, 6))
        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        for label, d in data.items():
            plt.plot(d, label=label)
        plt.legend(fontsize=16)


    # Light style
    plt.style.use("default")
    _plot("light")

    # Dark style
    plt.style.use("dark_background")
    _plot("dark")

    # Show plot if requested
    if show:
        plt.show()


if __name__ == "__main__": 
    # NOTE: The order of parameters matters!!
    conditions = {
    "mean": 4,
    "H_init" : 0,
    "runs" : 1_000_000,
    "steps" : 1000,
    "k" : 10,
    "alpha" : 0.1
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_with_baseline = Bandit(**conditions, device=device)
    model_without_baseline = Bandit(**conditions, device=device)
    
    model_with_baseline.train()
    model_without_baseline.train_without_baseline()

    data = {
        "With baseline": model_with_baseline.avg_rewards.tolist(),
        "Without baseline": model_without_baseline.avg_rewards.tolist()
    }


    plot_and_save(data,ylabel="Average reward", filename="avg_reward_bandit_baseline_comparison")

    
    data = {
        "With baseline": (model_with_baseline.optimal_action * 100).tolist() ,
        "Without baseline": (model_without_baseline.optimal_action * 100).tolist()
    }


    plot_and_save(data,ylabel="% Optimal action", filename="optimal_action_bandit_baseline_comparison")

