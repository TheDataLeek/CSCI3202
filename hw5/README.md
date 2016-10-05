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

In a nutshell, simulated annealing can be defined as follows.

1. Generate a random solution
2. Generate a "neighboring solution" to our generated solution
3. Keep whichever is better, or (with decaying probability) take the new one
   regardless
4. Go back to 2

Incomplete python code for this is below.

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

Genetic algorithms are very similar, and the algorithm can be defined as
follows.

1. Randomly generate an initial population of solutions
2. Use our solution population to generate some large number of children (note,
   these children should inherit properties from their parents)
3. Keep the best of our total population
4. Go back to 2

Again, incomplete code is below.

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

## Procedure

So in the context of our problem, we can examine how the code actually works
here.

### Simulated Annealing

### Genetic Algorithm

### Fitness Function

### Neighbor Detection

### Generating New Solutions

## Data

## Results
