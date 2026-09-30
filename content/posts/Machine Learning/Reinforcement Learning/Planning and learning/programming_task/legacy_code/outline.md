## env class

the `state space` is a grid of 6 by 9.
so, a state is a, b, where  a is 0 - 5 and b is 0 - 8
state => Discrete(2)

has methods
- `env.reset`
- `env.step(action)`

actions are 
| Up | down | left |right |
| --- |--- |---|---|
|0 | 1 | 2 | 3|

`env.step` returns s', r, done
if done, return the same terminal state, with `r = 0`


## agent class

The action-values at the terminal state should be zero, during initialisation.
