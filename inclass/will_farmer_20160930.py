#!/usr/bin/env python3.5

import sys
import random
import itertools

item_count = 10
weight_limit = 50

# weight, value
items = [(random.randint(1, 15), random.randint(1, 10)) for _ in range(item_count)]
print(items)
print('===')

solutions = [''.join([str(random.randint(0, 1)) for i in range(len(items))]) for j in range(3)]

def fitness(items, choices):
    weight = 0
    value = 0
    for choice, item in zip(choices, items):
        if choice == '1':
            weight += item[0]
            value += item[1]
    if weight > weight_limit:
        return 0
    else:
        return value

def combine(choice1, choice2):
    # i = random.randint(1, item_count - 1)
    i = int(item_count / 2)
    # Randomly combine
    if random.random() < .5:
        new_solution = choice1[:i] + choice2[i:]
    else:
        new_solution = choice2[:i] + choice1[i:]
    # 10% of time mutate
    if random.random() < .1:
        new_solution = list(new_solution)
        j = random.randint(0, item_count - 1)
        if new_solution[j] == '0':
            new_solution[j] = '1'
        else:
            new_solution[j] = '0'
        new_solution = ''.join(new_solution)
    return new_solution

def gen_new(top3):
    sols = []
    for _ in range(10):
        perms = list(itertools.permutations(top3))
        random.shuffle(perms)
        i, j = perms[0][:2]
        sols.append(combine(i, j))
    return sols

def get_top3(sols):
    return [s for s, v in sorted(get_value(sols), key=lambda tup: -tup[1])[:3]]

def get_value(sols):
    return [(item, fitness(items, item)) for item in sols]

for i in range(10):
    print('Starting with {}'.format(str(get_value(solutions))))
    new_solutions = gen_new(solutions)
    print('Birthed {}'.format(str(get_value(new_solutions))))
    full_solutions = solutions + new_solutions
    solutions = get_top3(full_solutions)
    print('Evolved to {}'.format(str(get_value(solutions))))
    print('---')

############################3
# This will be different than the traveling salesman in that we don't need to
# assert a much more strict function, and instead we can just make sure the
# weight is low. That being said, we need to be careful that our mutations don't
# make the problem unsolvable.
