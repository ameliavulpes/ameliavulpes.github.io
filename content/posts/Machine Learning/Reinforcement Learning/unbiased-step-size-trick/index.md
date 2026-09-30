---
title: Unbiased Step Size trick
date: 2026-09-26
lastmod: 2026-09-26
tags:
  - Medium
categories:
  - Machine Learning
  - Reinforcement Learning
  - Math
math: true
mathEngine: mathjax
draft: False
summary: The Unbiased Step Size Trick used to eliminate initial biases
---
In many cases, we initialise Q-values optimistically (e.g., setting all Q(s, a) to high values) to encourage exploration in greedy algorithms. However, with a constant step size $\alpha$, there is a bias toward the initial values, which may be detrimental in non-stationary tasks.  

Hence, we create two new variables $\beta, \bar{O}_{n}\in \mathbb R$, where $\beta$ is the final step size, and $\bar{O}_{n}$ is a helper variable. To start, we define our step size, $\alpha$:
$$\alpha_n = \frac{\beta}{\bar{O}_n} \quad \forall n > 0 ,\bar{O}_0 =  0$$
Where $\bar O_n$ follows the recursive relation:
$$ \bar{O}_{n} = \bar{O}_{n-1} + \beta(1 - \bar{O}_{n-1}) , \bar{O}_0 = 1$$
There are two important properties of this formulation:
- $n \rightarrow \infty$, $\bar{O}_{n} \rightarrow 1$ ,$\alpha \rightarrow \beta$.
- $\bar{O}_1$ = $\beta$

# How the bias term is removed
The bias can be removed for all recursive relations with the following form, where $\hat{x}$ is the estimate.
$$\hat{x}_{t+1} = \hat{x}_{t} +\alpha \left( U_{t}  - \hat{x}_{t} \right) $$
In this example, we use $Q_{t}$ instead of $\hat{x}_{t}$
$$\begin{aligned} Q_{t+1} &= Q_t + \alpha_t (U_t - Q_{t}) \\ &= \alpha_t U_t + (1 - \alpha_t)Q_t \\ &= \alpha_t U_t + \alpha_{t-1}(1-\alpha_t)U_{t-1} + (1-\alpha_t)(1-\alpha_{t-1})Q_{t-1} \\ &= \sum_{k=1}^{t} \alpha_k U_k \prod_{i=k+1}^{t}(1-\alpha_i) + Q_1 \prod_{i=1}^{t}(1-\alpha_i) \end{aligned}$$

When $n = 1$, $\bar{O}_n = \beta$, $\alpha_n = 1$. The coefficient of the $Q_1$ term is $0$, hence, there is no initial bias.

Proof that the recursive relation follows the two important properties.
$$\begin{align*}
\bar{O}_n &= \bar{O}_{n-1} + \beta(1 - \bar{O}_{n-1}) \\
&= \beta + (1-\beta)\bar{O}_{n-1} \\
&= \beta + \beta(1-\beta) + (1-\beta)^2\bar{O}_{n-2} \\
&= \beta\sum_{i=0}^{n-1}(1-\beta)^i + (1-\beta)^n\bar{O}_0 
\end{align*}$$
**Corollary:**
$$
\begin{align*}
\lim_{ n \to \infty }
   \beta\sum_{i=0}^{\infty}(1-\beta)^i 
&= \frac{\beta}{1-(1-\beta)} \\
&= 1.
\end{align*} $$
