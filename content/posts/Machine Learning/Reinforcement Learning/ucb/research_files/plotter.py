import matplotlib.pyplot as plt
import pickle

with open('content/posts/sutton-barto/ucb/data/ucb_optimal_action.pkl', 'rb') as f:
    ucb_optimal_action = pickle.load(f)

with open('content/posts/sutton-barto/ucb/data/eps_optimal_action.pkl', 'rb') as f:
    eps_optimal_action = pickle.load(f)

with open('content/posts/sutton-barto/ucb/data/ucb_reward.pkl', 'rb') as f:
    ucb_reward = pickle.load(f)

with open('content/posts/sutton-barto/ucb/data/eps_reward.pkl', 'rb') as f:
    eps_reward = pickle.load(f)

print(ucb_reward[-10:])
print(eps_reward[-10:])
# 


    def plot_and_save(data1, data2, data1_label, data2_label, xlabel = "Steps", ylabel = "y", filename=None, show=True, helper=True):

        def plot():
            plt.figure(figsize=(12,6))

            plt.xlabel(xlabel, fontsize=18)
            plt.ylabel(ylabel, fontsize=18)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            plt.plot(data1[:10_000], label= data1_label)
            plt.plot(data2[:10_000], label=data2_label)
            plt.legend(fontsize=16)

        plot()
        if filename:
            plt.savefig(f"content/posts/sutton-barto/ucb/{filename}_light.svg", bbox_inches="tight")
            # plt.savefig(f"content/posts/sutton-barto/ucb/{filename}._light.pdf", bbox_inches="tight")

    
        plt.style.use("dark_background")
        plot()
        if filename:
            plt.savefig(f"content/posts/sutton-barto/ucb/{filename}_dark.svg", bbox_inches="tight")
            # plt.savefig(f"content/posts/sutton-barto/ucb/{filename}._dark.pdf", bbox_inches="tight")

        if show: plt.show()



# plot_and_save(ucb_optimal_action, eps_optimal_action, r"UCB ($c=2$)", r"$\epsilon$-greedy ($\epsilon=0.1$)", 
#               ylabel="% Optimal Action", filename="ucb_optimal_action_long_term")#_c_2_long_term")

plot_and_save(ucb_reward, eps_reward, r"UCB ($c=2$)", r"$\epsilon$-greedy ($\epsilon=0.1$)", 
              ylabel="Average reward", filename="Avg_reward_zoomed_in") #_long_term")



# plt.figure(figsize=(12,4.5))

# plt.xlabel("Steps", fontsize=16)
# plt.ylabel("% Optimal Action", fontsize=16)

# plt.plot(ucb_optimal_action[:100] * 100, label=r"$\UCB ($\c=2$)")
# plt.plot(eps_optimal_action[:100] * 100, label=r"$\epsilon$-greedy ($\epsilon=0.1$)")
# plt.legend(fontsize=14)
# plt.savefig("content/posts/sutton-barto/ucb/ucb_zoomed.svg", bbox_inches="tight")
# plt.savefig("content/posts/sutton-barto/ucb/ucb_zoomed.pdf", bbox_inches="tight")
# plt.show()
