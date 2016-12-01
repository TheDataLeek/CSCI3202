# CSCI 3202 - Homework 7
## Will Farmer

# Purpose

In this assignment we will experiment with Markov Decision Processes and examine
various solutions that numerical solutions yield.

To do this, we will initially establish a Markov Decision Process for a provided
sample case with a horse navigating treacherous terrain. This model has the
following format.

```python
[[0   , 0   , 0   , 0   , -1  , 0   , -1  , -1 , 0   , 50],
 [None, None, -1  , -1  , 0   , -.5 , None, 0  , None, 0],
 [0   , 0   , 0   , 0   , 0   , -.5 , None, 0  , 0   , 0],
 [None, 2   , None, None, None, -.5 , 0   , 2  , None, 0],
 [None, 0   , 0   , 0   , 0   , None, -1  , -.5, -1  , 0],
 [0   , -.5 , None, 0   , 0   , None, 0   , 0  , None, 0],
 [0   , -.5 , None, 0   , -1  , None, 0   , -1 , None, None],
 [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0  , 0   , 0]],
```

With terminal state in the upper right hand corner.

The initial solution is of the following form.

```python
---------------------------------
| >  >  >  >  >  >  >  >  >  .  |
|       ^  >  >  ^     ^     ^  |
| >  v  >  >  ^  ^     >  >  ^  |
|    >           >  >  ^     ^  |
|    ^  <  <  <     ^  ^  >  ^  |
| >  ^     ^  ^     ^  ^     ^  |
| ^  ^     ^  v     ^  ^        |
| ^  ^  >  >  >  >  ^  ^  <  <  |
---------------------------------
```

# Results

## Living Reward

We first experiment with adjusting the values of the blank states to be positive
or negative 1.

1. We can set all blank states to `+1` and examine the results
```
---------------------------------
| 1  1  1  1  M  1  M  M  1  50 |
|       M  M  1  S     1     0  |
| 1  1  1  1  1  S     1  1  0  |
|    B           S  1  B     0  |
|    1  1  1  1     M  S  M  0  |
| 1  S     1  1     1  1     0  |
| 1  S     1  M     1  M        |
| 1  1  1  1  1  1  1  1  1  0  |
---------------------------------
---------------------------------
| >  >  >  >  >  >  >  >  >  .  |
|       ^  >  >  ^     ^     ^  |
| >  >  >  >  ^  ^     ^  >  ^  |
|    >           >  >  ^     ^  |
|    ^  <  <  <     ^  ^  >  ^  |
| >  ^     ^  ^     >  ^     ^  |
| ^  ^     v  v     ^  ^        |
| >  >  >  >  >  >  ^  ^  <  <  |
---------------------------------
```
We note that the results do indeed change.
2. That being said, no matter what changes are made, a solution is always found.

## Value of Gamma

The default value for the initial solution is `0.9`, by changing this to `0.1`,
we obtain a radically different answer.

```
---------------------------------
| 0  0  0  0  M  0  M  M  0  50 |
|       M  M  0  S     0     0  |
| 0  0  0  0  0  S     0  0  0  |
|    B           S  0  B     0  |
|    0  0  0  0     M  S  M  0  |
| 0  S     0  0     0  0     0  |
| 0  S     0  M     0  M        |
| 0  0  0  0  0  0  0  0  0  0  |
---------------------------------
---------------------------------
| >  <  ^  <  v  ^  <  >  >  .  |
|       v  >  v  <     v     ^  |
| >  v  <  v  <  <     v  >  ^  |
|    >           >  >  >     ^  |
|    ^  <  <  <     ^  ^  >  >  |
| <  ^     ^  ^     v  <     ^  |
| <  <     <  <     <  <        |
| >  v  >  >  v  >  v  v  >  >  |
---------------------------------
```
