---
title: "Loss functions"
date: 2026-03-13
lastmod: 2026-03-13
tags: ["Linear Algebra", "Jacobian","Matrix Calculus"]
categories: ["Machine Learning",Math]
math: true
summary: A list of loss functions and their main uses.
---
# Introduction

A loss function defines how prediction error is measured in a machine learning model by quantifying the discrepancy between predicted outputs and the ground truth. Different loss functions reflect different objectives.
## MSE
MSE, or Mean Squared Error, is commonly used for regression tasks. It is square of the difference between the predicted output and the ground truth, averaged over all samples.

$$\boxed{\text {MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }$$
where $n \in \mathbb R$ is the number of data points, $y_i\in \mathbb R$ represents the ground truth, and $\hat{y}_i\in \mathbb R$ represents the predicted values.

It can be rewritten in the matrix form as
$$\boxed{\text{MSE} = \frac{1}{n} \|\vec{y} - \hat{\vec{y}}\|^2}$$

## Cross entropy loss
Cross entropy loss, or *CE loss*, is commonly used for classification tasks. It originates from the KL divergence.
$$\boxed{ L = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)}$$
where $C\in \mathbb R$ is the number of classes, $y_c\in \mathbb R$ is a binary indicator (0 or 1) if class $c$ is the correct classification for the observation, and $\hat{y}_c$ is the predicted probability for class $c$.

It can be rewritten in the matrix form as
$$\boxed{L(\vec{y}, \hat{\vec{y}}) = -\vec{y}^T \log(\hat{\vec{y}})}$$