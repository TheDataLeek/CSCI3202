Title: Solving Political Boundaries Through Simulation
Date: 2016-10-14
Category: Homework
tags: homework, ai, python, numerics
Author: Will Farmer
Summary: How to generate solutions of hard problems with simulations.

# An Introduction to Simulated Annealing and Genetic Algorithms

Simulated Annealing and Genetic Algorithms are both methods of finding solutions
to problems through simulations. They are similar in many ways, but also very
different.

## What is a "Fitness Function"?

Before we dive into what exactly these algorithms are, let's talk about "fitness
functions". These are simply functions that "score" a system and give you a
number back. These functions are usually then maximized.

## What is Simulated Annealing?

```python
def simulated_annealing(g):
    s, c = generate_solution(g)
    T = 1
    Tmin = 1e-9
    alpha = 0.99
    k = 1
    i = 0
    while T > Tmin:
        sp, cp = generate_solution_neighbor(g, s, c)
        DE = cp - c
        if DE < 0:
            s = sp
            c = cp
        elif random.random() < math.exp(-DE / (k * T)):
            s = sp
            c = cp
        T *= alpha
        i += 1
    print(s, c, i)
```

## What are Genetic Algorithms?

```python
for i in range(10):
    print('Starting with {}'.format(str(get_value(solutions))))
    new_solutions = gen_new(solutions)
    print('Birthed {}'.format(str(get_value(new_solutions))))
    full_solutions = solutions + new_solutions
    solutions = get_top3(full_solutions)
    print('Evolved to {}'.format(str(get_value(solutions))))
    print('---')
```

# Drawing Political District Boundaries

Now that we know what these monsters are, we can dig into how they can be
applied to solving a system.

Let's say we're interested in determining how to section off a two-party system
of voters into "equal" districts, for some definition of equal. Our system is
defined in a provided file that simply denotes, for every index, the type of
voter in that location. It looks like this

```
D R D R D R R R
D D R D R R R R
D D D R R R R R
D D R R R R D R
R R D D D R R R
R D D D D D R R
R R R D D D D D
D D D D D D R D
```

Which can be plotted for readability.

![png](./smallState_initial.png)

## Purpose

We wish to segment the above district (or any like it) into a certain number of
voting districts. Sounds easy enough.

## Procedure

### Fitness Function

### Neighbor Detection

### Generating New Solutions

## Data

## Results
