---
title: Importance Sampling in Off Policy learning
date: 2026-07-23
lastmod: 2026-07-23
tags:
  - Medium
categories:
  - Machine Learning
  - Reinforcement Learning
  - Math
math: true
draft: false
mathEngine: mathjax
summary: A summary of Importance Sampling in the context of Reinforcement Learning
---
**Summary of Notation**  
Let $x$ denote the random variable of interest (the *state* $s$, *state-action* pair $(s,a)$, or *trajectory* $\tau$).   
Let $p(x)$ and $q(x)$ be the probability distribution of $x$ while following target policy $\pi$ or behaviour policy $\mu$.  

Let $f(x)$ be any quantity computed at state $x$, such as the reward $R(x)$, the Value function $V(x)$, or the advantage $A(x)$.  

Through Importance Sampling, we can draw samples from a different distribution other than the target policy.  
$$\begin{align*}
\mathbb{E}_p [f(x)] &= \int p(x) f(x) \, dx \\
&= \int \frac{q(x)}{q(x)} p(x) f(x) \, dx & \big(q(x) \gt 0 \text{ when }p(x) \gt 0\big)\\
&= \mathbb{E}_{q} \left[ \frac{p(x)}{q(x)} f(x) \right]
\end{align*}
$$
 
