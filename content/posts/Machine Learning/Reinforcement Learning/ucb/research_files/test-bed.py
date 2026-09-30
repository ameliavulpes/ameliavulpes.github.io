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
        
    def train(self,c,epsilon) -> Bandit:
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
        self = Bandit(self.Q_init,self.runs,self.steps,self.k, self.alpha)


      
def plot_and_save(data, xlabel="Steps", ylabel="y", filename=None, show=True):
    """
    data should be in format

    (label: data)
    """

    def plot():
        plt.figure(figsize=(12,6))

        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        for label, d in data.items():
            print(label)
            plt.plot(d, label= label)
        plt.legend(fontsize=16)


    plt.style.use('default') 
    plot()
    if filename:
        plt.savefig(f"content/posts/sutton-barto/ucb/blog_imgs/light/{filename}.svg", bbox_inches="tight",transparent=True)
        plt.savefig(f"content/posts/sutton-barto/ucb/pdf_imgs/light/{filename}.pdf", bbox_inches="tight",transparent=True)


    plt.style.use("dark_background")
    plot()
    if filename:
        plt.savefig(f"content/posts/sutton-barto/ucb/blog_imgs/dark/{filename}.svg", bbox_inches="tight" ,transparent=True)
        plt.savefig(f"content/posts/sutton-barto/ucb/pdf_imgs/dark/{filename}.pdf", bbox_inches="tight",transparent=True)

    if show: plt.show()



def run_model(Q_init = 0, runs = 5000, steps = 500, k = 10, alpha = 0.1, c = 0, epsilon = 0.1):
    print(f"Q_init {Q_init}", f"runs {runs}", f"steps {steps}", f"k {k}", f"alpha {alpha}" , f"c  {c}" , f"epsilon {epsilon}")

    model = Bandit(Q_init, runs, steps, k, alpha)
    model.train(c, epsilon)
    return model


# probably not worth it lol, just run sequentially
if __name__ == "__main__": 

    # NOTE: The order of parameters matters!!
    conditions = {
    "Q_init" : 0,
    "runs" : 50_000,
    "steps" : 10_000,
    "k" : 10,
    "alpha" : 0.1
    }

    eps_grdy_conds = {
        "c" : 0,
        "epsilon" : 0.1
    }

    ucb_conds = {
        "c" : 2,
        "epsilon" : 0
    }

    experiments = [tuple(x) for x in [(conditions | eps_grdy_conds).values(), (conditions | ucb_conds).values()]]


    with Pool(2) as p:
        eps_greedy, ucb = p.starmap(run_model,experiments)
        

        plot_and_save( 
                {r"UCB ($c=2$)": ucb.avg_rewards,
                r"$\epsilon$-greedy ($\epsilon=0.1$)": eps_greedy.avg_rewards},
                ylabel="Average reward",
                # filename="avg_reward_c2_long_term"  
        )

        plot_and_save({
            r"UCB ($c=2$)": ucb.optimal_action*100,
            r"$\epsilon$-greedy ($\epsilon=0.1$)": eps_greedy.optimal_action*100
        },
        ylabel="% Optimal Action",
        filename="optimal_action_c2_long_term"  
        )



    eps_grdy = Bandit(**conditions)
    ucb = Bandit(**conditions)
    with ProcessPoolExecutor() as e:
        f1 = e.submit(eps_grdy.train, **eps_grdy_conds)
        f2 = e.submit(ucb.train, **ucb_conds)

        eps_grdy = f1.result()
        ucb = f2.result()


    raise NotImplementedError
    runs = 50000
    steps = 500
    k = 10
    alpha = 0.1


    # with ProcessPoolExecutor() as e:
    #     f1 = e.submit(run_model)
    #     f2 = e.submit(run_ucb)

    #     eps_greedy = f1.result()
    #     ucb = f2.result()

    plot_and_save( 
            {r"UCB ($c=2$)": ucb.avg_rewards,
            r"$\epsilon$-greedy ($\epsilon=0.1$)": eps_greedy.avg_rewards},
            ylabel="Average reward",
            filename="avg_reward_c2"  
    )

    plot_and_save({
        r"UCB ($c=2$)": ucb.optimal_action*100,
        r"$\epsilon$-greedy ($\epsilon=0.1$)": eps_greedy.optimal_action*100
    },
    ylabel="Average reward",
    filename="optimal_action_c2"  
    )






    import sys; sys.exit()
    runs = 5000
    steps = 500
    k = 10
    alpha = 0.1

    plot_and_save( 
        {r"UCB ($c=1$)": ucb.avg_rewards[:50],
        r"$\epsilon$-greedy ($\epsilon=0.1$)": eps_greedy.avg_rewards[:50]},
        ylabel="Average reward",
        filename="avg_reward_zoomed"  
)

    plot_and_save({
        r"UCB ($c=1$)": ucb.avg_rewards[:500],
        r"$\epsilon$-greedy ($\epsilon=0.1$)": eps_greedy.avg_rewards[:500]
    },
    ylabel="Average reward",
    filename="avg_reward"  
    )



