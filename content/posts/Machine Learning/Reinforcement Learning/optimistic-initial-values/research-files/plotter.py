import matplotlib.pyplot as plt
import pickle

with open('content/posts/sutton-barto/optimistic-initial-values/research-files/realistic.pkl', 'rb') as f:
    realistic = pickle.load(f)

with open('content/posts/sutton-barto/optimistic-initial-values/research-files/optimistic.pkl', 'rb') as f:
    optimistic = pickle.load(f)

print(realistic)
print(optimistic)
# plt.style.use('dark_background')

plt.figure(figsize=(12,4.5))

plt.xlabel("Steps", fontsize=18)
plt.ylabel("% Optimal Action", fontsize=18)

plt.plot(realistic * 100, label=r"$\epsilon$-greedy ($\epsilon=0.1$)")
plt.plot(optimistic * 100, label=r"Optimistic $Q_0=5$")
plt.legend(fontsize=16)
plt.savefig(r"content\posts\sutton-barto\optimistic-initial-values\blog-files\optimistic_initial_values_light.svg", bbox_inches="tight", transparent= True)
plt.savefig(r"content\posts\sutton-barto\optimistic-initial-values\blog-files\optimistic_initial_values_light.pdf", bbox_inches="tight", transparent= True)

plt.show()


plt.figure(figsize=(12,4.5))

plt.xlabel("Steps", fontsize=16)
plt.ylabel("% Optimal Action", fontsize=16)

plt.plot(realistic[:100] * 100, label=r"$\epsilon$-greedy ($\epsilon=0.1$)")
plt.plot(optimistic[:100] * 100, label=r"Optimistic $Q_0=5$")
plt.legend(fontsize=14)
plt.savefig(r"content\posts\sutton-barto\optimistic-initial-values\blog-files\optimistic_initial_values_zoomed_light.svg", bbox_inches="tight", transparent= True)
plt.savefig(r"content\posts\sutton-barto\optimistic-initial-values\blog-files\optimistic_initial_values_zoomed_light.pdf", bbox_inches="tight", transparent= True)

plt.show()

