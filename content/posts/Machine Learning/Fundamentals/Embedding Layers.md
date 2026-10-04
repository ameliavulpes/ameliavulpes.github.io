---
title: "Embedding Layers"
date: 2026-10-01
lastmod: 2026-10-01
tags: ["Linear Algebra", "Backpropagation"]
categories: ["Machine Learning"]
math: true
summary: How Embedding Layers are trained
---

# The problem
One hot embeddings are sparse and inefficient.

# Summary

- Embedding layers are used to turn known objects (typically indices), into semantic vectors.
- As they are a one to one mapping from the input to the vectors , they have cleaner convergence guarantees. However, this one to one mapping also means that they are unable to generalise.
