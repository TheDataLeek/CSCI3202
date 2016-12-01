#!/usr/bin/env python2.7

#Artificial Intelligence: A Modern Approach

# Search AIMA
#AIMA Python file: mdp.py

"""Markov Decision Processes (Chapter 17)

First we define an MDP, and the special case of a GridMDP, in which
states are laid out in a 2-dimensional grid.  We also represent a policy
as a dictionary of {state:action} pairs, and a Utility function as a
dictionary of {state:number} pairs.  We then define the value_iteration
and policy_iteration algorithms."""

from __future__ import print_function
import sys
from utils import *  # what the fuck

'''

AI: A Modern Approach by Stuart Russell and Peter Norvig
Modified: Jul 18, 2005

The data matrix you will need for the assignment:

[0, 0, 0, 0, -1, 0, -1, -1, 0, 50],
[None, None, -1, -1, 0, -.5, None, 0, None, 0],
[0, 0, 0, 0, 0, -.5, None, 0, 0, 0],
[None, 2, None, None, 0, -.5, 0, 2, None, 0],
[0, 0, None, 0, 0, None, -1, -.5, -1, 0],
[0, -.5, None, 0, 0, None, 0, 0, None, 0],
[0, -.5, None, 0, -1, None, 0, -1, None, None],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

'''


def main():
    """
    Answers to questions:

    1. Exit, since they're terminal
    2. Defined in the T function, 80% chance of following the action, 10% chance
    of turning left or right.
    3. value_iteration
    4. {(0, 1): 0.3984432178350045,
        (1, 2): 0.649585681261095,
        (3, 2): 1.0,
        (0, 0): 0.2962883154554812,
        (3, 0): 0.12987274656746342,
        (3, 1): -1.0,
        (2, 1): 0.48644001739269643,
        (2, 0): 0.3447542300124158,
        (2, 2): 0.7953620878466678,
        (1, 0): 0.25386699846479516,
        (0, 2): 0.5093943765842497}
    5. Actions are represented by grid directions that we can move to at any
    given moment in time. At any given point we can move in any of those
    directions.
    """
    myMDP = GridMDP([[-0.04, -0.04, -0.04, +1],
                     [-0.04, None,  -0.04, -1],
                     [-0.04, -0.04, -0.04, -0.04]],
                    terminals=[(3,1),(3,2)])

    U = value_iteration(myMDP, .001)

    # Initial
    horseMDP = GridMDP(
                [[0   , 0   , 0   , 0   , -1  , 0   , -1  , -1 , 0   , 50],
                 [None, None, -1  , -1  , 0   , -.5 , None, 0  , None, 0],
                 [0   , 0   , 0   , 0   , 0   , -.5 , None, 0  , 0   , 0],
                 [None, 2   , None, None, None, -.5 , 0   , 2  , None, 0],
                 [None, 0   , 0   , 0   , 0   , None, -1  , -.5, -1  , 0],
                 [0   , -.5 , None, 0   , 0   , None, 0   , 0  , None, 0],
                 [0   , -.5 , None, 0   , -1  , None, 0   , -1 , None, None],
                 [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0  , 0   , 0]],
                terminals=[(9, 7)])
    value_iteration_utility = value_iteration(horseMDP, epsilon=1e-5)
    value_iteration_policy = best_policy(horseMDP, value_iteration_utility)
    print("Initial Solution")
    printgrid(horseMDP.grid[::-1], replace={-1:'M', -0.5:'S', 2:'B', None:' '})
    printgrid(horseMDP.to_arrows(value_iteration_policy), replace={None: '  '})

    # Changing zeros
    horseMDP = GridMDP(
                [[1   , 1   , 1   , 1   , -1  , 1   , -1  , -1 , 1   , 50],
                 [None, None, -1  , -1  , 1   , -.5 , None, 1  , None, 0],
                 [1   , 1   , 1   , 1   , 1   , -.5 , None, 1  , 1   , 0],
                 [None, 2   , None, None, None, -.5 , 1   , 2  , None, 0],
                 [None, 1   , 1   , 1   , 1   , None, -1  , -.5, -1  , 0],
                 [1   , -.5 , None, 1   , 1   , None, 1   , 1  , None, 0],
                 [1   , -.5 , None, 1   , -1  , None, 1   , -1 , None, None],
                 [1   , 1   , 1   , 1   , 1   , 1   , 1   , 1  , 1   , 0]],
                terminals=[(9, 7)])
    value_iteration_utility = value_iteration(horseMDP, epsilon=1e-5)
    value_iteration_policy = best_policy(horseMDP, value_iteration_utility)
    print("Living Reward Solution")
    printgrid(horseMDP.grid[::-1], replace={-1:'M', -0.5:'S', 2:'B', None:' '})
    printgrid(horseMDP.to_arrows(value_iteration_policy), replace={None: '  '})

    # Adjusting gamma
    horseMDP = GridMDP(
                [[0   , 0   , 0   , 0   , -1  , 0   , -1  , -1 , 0   , 50],
                 [None, None, -1  , -1  , 0   , -.5 , None, 0  , None, 0],
                 [0   , 0   , 0   , 0   , 0   , -.5 , None, 0  , 0   , 0],
                 [None, 2   , None, None, None, -.5 , 0   , 2  , None, 0],
                 [None, 0   , 0   , 0   , 0   , None, -1  , -.5, -1  , 0],
                 [0   , -.5 , None, 0   , 0   , None, 0   , 0  , None, 0],
                 [0   , -.5 , None, 0   , -1  , None, 0   , -1 , None, None],
                 [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0  , 0   , 0]],
                terminals=[(9, 7)], gamma=0.1)
    value_iteration_utility = value_iteration(horseMDP, epsilon=1e-5)
    value_iteration_policy = best_policy(horseMDP, value_iteration_utility)
    print("Changing Gamma to 0.1")
    printgrid(horseMDP.grid[::-1], replace={-1:'M', -0.5:'S', 2:'B', None:' '})
    printgrid(horseMDP.to_arrows(value_iteration_policy), replace={None: '  '})


def printgrid(grid, replace=None):
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if replace is not None:
                try:
                    new_char = replace[grid[i][j]]
                except KeyError:
                    new_char = str(grid[i][j])
                grid[i][j] = new_char
            else:
                grid[i][j] = str(grid[i][j])
    col_width = 0
    for l in grid:
        for v in l:
            if len(v) > col_width:
                col_width = len(v)
    col_width += 1
    print('-' * (3 + col_width * len(grid[0])))
    for l in grid:
        print('| ', end='')
        for v in l:
            print('{}{}'.format(v, ' ' * (col_width - len(v))), end="")
        print('|')
    print('-' * (3 + col_width * len(grid[0])))


class MDP:
    """A Markov Decision Process, defined by an initial state, transition model,
    and reward function. We also keep track of a gamma value, for use by
    algorithms. The transition model is represented somewhat differently from
    the text.  Instead of T(s, a, s') being  probability number for each
    state/action/state triplet, we instead have T(s, a) return a list of (p, s')
    pairs.  We also keep track of the possible states, terminal states, and
    actions for each state. [page 615]"""

    def __init__(self, init, actlist, terminals, gamma=.9):
        update(self, init=init, actlist=actlist, terminals=terminals,
               gamma=gamma, states=set(), reward={})

    def R(self, state):
        "Return a numeric reward for this state."
        return self.reward[state]

    def T(state, action):
        """Transition model.  From a state and an action, return a list
        of (result-state, probability) pairs."""
        raise NotImplementedError

    def actions(self, state):
        """Set of actions that can be performed in this state.  By default, a
        fixed list of actions, except for terminal states. Override this
        method if you need to specialize by state."""
        if state in self.terminals:
            return [None]
        else:
            return self.actlist

class GridMDP(MDP):
    """A two-dimensional grid MDP, as in [Figure 17.1].  All you have to do is
    specify the grid as a list of lists of rewards; use None for an obstacle
    (unreachable state).  Also, you should specify the terminal states.
    An action is an (x, y) unit vector; e.g. (1, 0) means move east."""
    def __init__(self, grid, terminals, init=(0, 0), gamma=.9):
        grid.reverse() ## because we want row 0 on bottom, not on top
        MDP.__init__(self, init, actlist=orientations,
                     terminals=terminals, gamma=gamma)
        update(self, grid=grid, rows=len(grid), cols=len(grid[0]))
        for x in range(self.cols):
            for y in range(self.rows):
                self.reward[x, y] = grid[y][x]
                if grid[y][x] is not None:
                    self.states.add((x, y))

    def T(self, state, action):
        if action == None:
            return [(0.0, state)]
        else:
            return [(0.8, self.go(state, action)),
                    (0.1, self.go(state, turn_right(action))),
                    (0.1, self.go(state, turn_left(action)))]

    def go(self, state, direction):
        "Return the state that results from going in this direction."
        state1 = vector_add(state, direction)
        return if_(state1 in self.states, state1, state)

    def to_grid(self, mapping):
        """Convert a mapping from (x, y) to v into a [[..., v, ...]] grid."""
        return list(reversed([[mapping.get((x,y), None)
                               for x in range(self.cols)]
                              for y in range(self.rows)]))

    def to_arrows(self, policy):
        chars = {(1, 0):'>', (0, 1):'^', (-1, 0):'<', (0, -1):'v', None: '.'}
        return self.to_grid(dict([(s, chars[a]) for (s, a) in policy.items()]))


def value_iteration(mdp, epsilon=0.001):
    "Solving an MDP by value iteration. [Fig. 17.4]"
    U1 = dict([(s, 0) for s in mdp.states])
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    while True:
        U = U1.copy()
        delta = 0
        for s in mdp.states:
            U1[s] = R(s) + gamma * max([sum([p * U[s1] for (p, s1) in T(s, a)])
                                        for a in mdp.actions(s)])
            delta = max(delta, abs(U1[s] - U[s]))
        if delta < epsilon * (1 - gamma) / gamma:
             return U

def best_policy(mdp, U):
    """Given an MDP and a utility function U, determine the best policy,
    as a mapping from state to action. (Equation 17.4)"""
    pi = {}
    for s in mdp.states:
        pi[s] = argmax(mdp.actions(s), lambda a:expected_utility(a, s, U, mdp))
    return pi

def expected_utility(a, s, U, mdp):
    "The expected utility of doing a in state s, according to the MDP and U."
    # for p, s1 in mdp.T(s, a):
    #     print('{}\t{}\t{}'.format(p, s1, U[s1]))
    return sum([p * U[s1] for (p, s1) in mdp.T(s, a)])


def policy_iteration(mdp):
    "Solve an MDP by policy iteration [Fig. 17.7]"
    U = dict([(s, 0) for s in mdp.states])
    pi = dict([(s, random.choice(mdp.actions(s))) for s in mdp.states])
    while True:
        U = policy_evaluation(pi, U, mdp)
        unchanged = True
        for s in mdp.states:
            a = argmax(mdp.actions(s), lambda a: expected_utility(a,s,U,mdp))
            if a != pi[s]:
                pi[s] = a
                unchanged = False
        if unchanged:
            return pi

def policy_evaluation(pi, U, mdp, k=20):
    """Return an updated utility mapping U from each state in the MDP to its
    utility, using an approximation (modified policy iteration)."""
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    for i in range(k):
        for s in mdp.states:
            U[s] = R(s) + gamma * sum([p * U[s1] for (p, s1) in T(s, pi[s])])
    return U


if __name__ == '__main__':
    sys.exit(main())
