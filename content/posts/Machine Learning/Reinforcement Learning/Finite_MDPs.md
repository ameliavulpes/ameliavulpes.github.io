---
title: Finite MDPs
date: 2026-03-12
lastmod: 2026-03-23
tags:
categories:
  - Machine Learning
  - Reinforcement Learning
  - Action-Value
draft: false
math: true
mathEngine: mathjax
summary:
---
# Characteristics of MDPs 
In a *finite* MDP, the set of states, actions, and rewards ($\mathcal S$, $\mathcal A$, and $\mathcal R$) all have a finite number of elements. For particular values of $S_t$ and $A_t$, there is a joint probability that the next state and reward will be $s^\prime$ and $r$, respectively.
$$\newcommand{\coloneqq}{\mathrel{\mathop:}=} \begin{equation}
\boxed{p(s^\prime,r \mid s,a) \doteq P(S_{t+1} = s^\prime, R_{t+1} = r, \mid  S_{t}=s, A_{t}= a)}
\label{four_value_prob}
\end{equation}$$

where the probabilities given by $p$ completely characterise the *dynamics* of the environment.

A state representation satisfies the *Markov property* if the probability of each possible value of $S_{t+1}$ and $R_{t+1}$ must depend only on the current state and action, $S_{t}$ and $A_{t}$, and completely independent on all earlier states and actions. This can be interpreted as a restriction on the state to include all information required to determine the probability distribution of the next state and reward, given the current action. In most Reinforcement Learning tasks, the state is assumed to hold the Markov property.

# Returns and episodes
The goal of the agent is to maximise the *expected return*, denoted $G_t$, is defined as some specific function of future rewards. In the simplest case, the return is is the sum of future rewards.
$$\begin{equation}
G_t \doteq  R_{t+1} +  R_{t+2} +  R_{t+3} +\cdots + R_T
\label{return}
\end{equation}$$
where $T$ is the final time step. This approach makes sense when there is a natural notion of a final time step, in which the sequence of agent–environment interactions naturally terminates. This sequence of interactions is called *episodes*.

However, not all tasks naturally terminate and can be broken down into discrete episodes. We call these *continuing* tasks. In these cases, the return formulation in \eqref{return} is problematic as the return, which we are trying to maximise, could be unbounded.

To address this issue, we introduce the concept of discounting.

$$
\begin{align} 
G_t &\doteq  R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots \notag\\ 
&= R_{t+1} + \gamma G_{t+1} \label{discounted_return}\\
&= \sum_{k=0}^\infty \gamma^k R_{t+k+1} \notag
\end{align}
$$ 
where $\gamma \in [0,1]$, a hyper parameter, is the *discount rate*.
The discount rate determines the present value of future rewards. If $\gamma < 1$, the infinite sum in \eqref{discounted_return} has a finite value as long as the reward sequence ${R_k}$ is bounded. 

## Exercises from Sutton Barto

<span id="exercise:2.1"></span>**Exercise *2.1***  Imagine that you are designing a robot to run a maze. You decide to give it a reward of $+1$ for escaping from the maze and a reward of zero at all other times. The task seems to break down naturally into episodes—the successive runs through the maze—so you decide to treat it as an episodic task, where the goal is to maximize expected total reward \eqref{return}. After running the learning agent for a while, you find that it is showing no improvement in escaping from the maze. What is going wrong? Have you effectively communicated to the agent what you want it to achieve?

# Policies and Value functions

## Formal definitions

Most reinforcement learning algorithms involving estimating *value functions*, which estimates the expected return for an agent in a given state. As the expected return is dependent on action taken, value functions are defined with respect to particular ways of acting, called *policies*.

Formally, a policy is a mapping from states to probabilities of selecting each possible
action. 
$$\begin{gather*}
\pi(a \mid s) \doteq P(A_t = a \mid S_{t} = s ) 
\end{gather*}$$
<span id="exercise:3.1"></span>**Exercise *3.1***  If the current state is $S_t$, and actions are selected according to a stochastic policy $\pi$, what is the expectation of $R_{t+1}$ in terms of $\pi$ and the four-argument
function $p$ \eqref{four_value_prob}?  
[See solution](#solution:3.1)

The *value function* of a state $s$ under a policy $\pi$, denoted $v_\pi(s)$, is the expected return when starting in $s$ and following $\pi$. For MDPs, $v_{\pi}$ is formally defined as 
$$v_{\pi}(s) \doteq \mathbb{E}_{\pi} [G_t \mid S_t = s] = \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \;\middle| \; S_t = s \right], \text{for all } s \in \mathcal{S}$$
Similarly, we define the value of taking action $a$ in state $s$ under a policy $\pi$, denoted
$q_\pi(s, a)$, as the expected return starting from $s$, taking the action $a$, and thereafter
following policy $\pi$, as the *action-value* function for policy $\pi$.
$$q_{\pi}(s,a) \doteq \mathbb{E}_{\pi} [G_t \mid S_t = s, A_{t} = a] = \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \;\middle| \; S_t = s , A_{t} = a\right]$$
<span id="exercise:3.2"></span>**Exercise *3.2***  Find $v_\pi$ in terms of $q_\pi$ and $\pi$.   
<span id="exercise:3.3"></span>**Exercise *3.3***  Find $q_\pi$ in terms of $v_\pi$ and the four-argument function $p$ 
\eqref{four_value_prob}  
[See solutions](#solution:3.2)

## Estimating value functions

The value functions $v_\pi$ and $q_{\pi}$ can be estimated from experience. By taking a sample averages of the returns in each state, an approximation can be found for $v_*$, or the true value function. Likewise, sample averages can be taken in each state-action pair to find an approximation of $q_\pi$. These are called *Monte Carlo methods*.

A fundamental property of value functions used throughout reinforcement learning problems is that they satisfy a recursive relationship, like the return in \eqref{discounted_return}. The recursive relations for $v_\pi$ and $q_\pi$ are also their *Bellman equations*.
$$\begin{align*}
v_\pi(s) &\doteq \mathbb{E}_\pi [G_t \mid S_t = s] \\
&= \mathbb{E}_\pi [R_{t+1} + \gamma G_{t+1} \mid S_t = s] \tag{by \eqref{discounted_return}}\\
&= \mathbb{E}_\pi [R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s]  \\
% &= \sum_a \pi(a \mid s) \sum_{s'} \sum_r p(s', r \mid s, a) \left[ r + \gamma \mathbb{E}_\pi [G_{t+1} \mid S_{t+1} = s'] \right] \\
&= \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma v_\pi(s') \right], \quad \text{for all } s \in \mathcal{S}
\end{align*}$$
<span id="exercise:3.4"></span>**Exercise *3.4***  Find a recursive relation for $q_\pi$.   
[See solution](#solution:3.4)

By solving the Bellman equations, we can find the value functions $v_\pi$ and $q_\pi$ of a policy. (See [Appendix](#Appendix) for more details).
## Optimal policies

One way to obtain the optimal policy $\pi_*$ is by computing its corresponding value functions $v_*$ and $q_*$. Since an optimal policy assigns probability only to optimal actions, and all optimal actions have the same action-value, we can rewrite the value function as
$$\begin{align*}
v_{*}(s) &= \sum_{a} \pi_*(a \mid s)\, q_{*}(s, a) \tag{by \eqref{ex:3.2}}\\
&=  \underset{a}{\max} q_*(s,a)\\
&=  \underset{a}{\max} \sum_{s^\prime, r} \, p(s^\prime, r \mid s,a) \,[ r + \gamma v_{*}(s^\prime)] \tag{by \eqref{ex:3.3}}
\end{align*}$$
Likewise, we can rewrite the action-value function as
$$\begin{align*}
q_{*}(s,a)  
&= \sum_{s^\prime, r} \, p(s^\prime, r \mid s,a) \,[ r + \gamma \sum_{a^\prime \in \mathcal{A}} *(a^\prime \mid s^\prime) \, q_{*}(s^\prime, a^\prime)] \tag{by \eqref{ex:3.4}} \\
&= \sum_{s^\prime, r} \, p(s^\prime, r \mid s,a) \,[ r + \gamma  \,\underset{a^\prime}{\max q_*(s,a)}] \notag
\end{align*}$$
# Solutions to Exercises

<span id="solution:3.1"></span>[**Exercise *3.1*** ](#exercise:3.1)$\;$To compute the expected reward given the current state $S_{t} = s$, we weigh each action by the policy $\pi(a \mid s)$, and for each action, we take the expected reward under the environment dynamics.
$$\begin{align*}
\mathbb{E}[R_{t+1} \mid S_t = s] 
&= \sum_{a} \pi(a \mid s) \, \mathbb{E}[R_{t+1} \mid s, a] \\
&= \sum_{a} \pi(a \mid s) \, \sum_{r} r p(r \mid s, a)\\
&= \sum_{a} \pi(a \mid s) \sum_{s^\prime} \sum_{r} r \, p(s^\prime, r \mid s,a)
\end{align*}$$
<span id="solution:3.2"></span>[**Exercise *3.2*** ](#exercise:3.2)  
$$\begin{align}
v_{\pi}(s) &= \mathbb{E} [G_t \mid S_t = s] \notag \\
&= \sum_{a} \pi(a \mid s)\, \mathbb{E} [G_t \mid S_t = s, A_t = a] \notag \\
&= \sum_{a} \pi(a \mid s)\, q_{\pi}(s, a) 
\label{ex:3.2}
\end{align}$$

<span id="solution:3.3"></span>[**Exercise *3.3*** ](#exercise:3.3)  
$$\begin{align}
q_{\pi}(s,a)  
&= \mathbb{E} [G_t \mid S_t = s, A_t = a] \notag \\ 
&= \sum_{s^\prime} \sum_{r} \, p(s^\prime, r \mid s,a) \, \mathbb{E} [G_{t}]\notag \\
&= \sum_{s^\prime, r} \, p(s^\prime, r \mid s,a) \,[ r + \gamma v_{\pi}(s^\prime)]
\label{ex:3.3}
\end{align}$$
<span id="solution:3.4"></span>[**Exercise *3.4*** ](#exercise:3.4) $\;$Substituting the result from Exercise 3.2 into Exercise 3.3
$$\begin{align}
q_{\pi}(s,a)  
&= \sum_{s^\prime, r} \, p(s^\prime, r \mid s,a) \,[ r + \gamma v_{\pi}(s^\prime)]\notag \\
&= \sum_{s^\prime, r} \, p(s^\prime, r \mid s,a) \,[ r + \gamma \sum_{a^\prime \in \mathcal{A}} \pi(a^\prime \mid s^\prime) \, q_{\pi}(s^\prime, a^\prime)] \label{ex:3.4}
\end{align}$$

# Summary
- Most reinforcement learning tasks assume the markov property.
- Returns are typically discounted to 
	- Bound the return in continuing tasks[^1]
	- Increase agent myopia 


# Appendix

**Proof.** The Bellman equations guarantees a unique solution to $v_\pi$ and $q_\pi$. It is helpful to specify notation as follows:
- A value vector $V_\pi$ of length $|\mathcal S|$
- The action-value and reward $Q_\pi$ and $\mathbf r$ as vectors of length $|\mathcal{S} \cdot \mathcal{A}|$
- A matrix $P$ of size $(|\mathcal S|\cdot|\mathcal A| \times |\mathcal S|)$, where the entry $P_{(s,a),s^\prime} \coloneqq P(s^\prime \mid s,a)$ 
- A transition matrix induced by a deterministic stationary policy $\pi$, where the entry $P^\pi_{(s,a),(s^\prime, a^\prime)} \coloneqq P(s^\prime \mid s,a)\,\pi(a^\prime \mid s^\prime)$ 

Using the above notation

$$\begin{gather*}
Q_\pi = \mathbf r + \gamma P\,V _\pi\\
\\
Q_\pi = \mathbf r + \gamma P^\pi Q_\pi \\
Q_\pi (\mathbf{I}- \gamma P^\pi)= \mathbf r\\
Q_\pi = (\mathbf{I}- \gamma P^\pi)^{-1} \, \mathbf r
\end{gather*}$$
If $\mathbf{I}- \gamma P^\pi$ is invertible, then $Q_\pi$ has a unique solution. Observe that for any non-zero vector $x \in \mathbb R^{|S||A|}$
$$\begin{align*}
\|(I - \gamma P^\pi)x\|_\infty &= \|x - \gamma P^\pi x\|_\infty \\
&\ge \|x\|_\infty - \gamma \|P^\pi x\|_\infty & \text{(triangle inequality for norms)} \\
&\ge \|x\|_\infty - \gamma \|x\|_\infty & (P^\pi \text{ is row-stochastic}) \\
&= (1 - \gamma)\|x\|_\infty > 0, & (\gamma < 1, x \neq 0)
\end{align*}$$
which implies $\mathbf{I}- \gamma P^\pi$ is full rank and thus invertible.

**Corollary.**$\;$If $q_\pi$ has a unique solution, then so does $v_\pi$.
$$v_\pi(s) = \sum_a \pi(a \mid s)\,q_\pi(s,a)$$


[^1]: Discounting is not necessary to obtain bounded returns in continuing tasks. Using the differential return formulation, the long-run return remains bounded without discounting.


