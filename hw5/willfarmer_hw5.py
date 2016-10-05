#!/usr/bin/env python2.7

import sys
import argparse
import math
import random
import re
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import moviepy
from moviepy import editor
import queue
from tqdm import tqdm
import time


def main():
    args = get_args()
    system = System(args.filename)
    if args.annealing:
        simulated_annealing(system, args.numdistricts, args.precision, args.animate)
    elif args.genetic:
        genetic_algorithm(system, args.numdistricts, args.precision, args.animate)
    else:
        print('Running in Demo Mode!!!')
        print('First we\'ll use Simulated Annealing')
        simulated_annealing(system, args.numdistricts, args.precision, False)
        print('Now we\'ll try the Genetic Algorithm')
        genetic_algorithm(system, args.numdistricts, args.precision, False)


def simulated_annealing(system, numdistricts, precision, animate):
    solution = Solution(system, numdistricts)
    solution.generate_random_solution()
    history = [solution]
    k = 0.5
    Tvals = np.arange(1, 1e-12, -1.0/precision)
    for i, T in tqdm(enumerate(Tvals), total=len(Tvals)):
        new_solution = solution.copy()
        new_solution.mutate()
        dv = new_solution.value - solution.value
        if (dv > 0 or random.random() < math.exp(dv / (k * T))):
            solution = new_solution
            history.append(new_solution)

    solution.count = len(Tvals)
    solution.algo = 'Simulated Annealing'
    print(solution)
    print(solution.summary())

    if animate:
        print('Saving to File')
        fig, axarr = plt.subplots(1, 2, figsize=(8, 8))
        axarr[0].imshow(systemdata, interpolation='nearest')
        sol = axarr[1].imshow(history[0].full_mask, interpolation='nearest')
        axarr[1].set_title('value {}'.format(history[0].value))
        plt.axis('off')

        def update_plot(i):
            sol.set_data(history[i].full_mask)
            axarr[1].set_title('value {}'.format(history[i].value))
            plt.suptitle('Solution {}'.format(i))
            return sol,

        interval = int(60000.0 / len(history))
        if interval == 0:
            interval = 1
        ani = animation.FuncAnimation(fig, update_plot, len(history),
                                      interval=interval, blit=True)
        filename = 'simulated_annealing_solution_{}'.format(filename.split('.')[0])
        ani.save(filename + '.mp4')

        editor.VideoFileClip(filename + '.mp4')\
                .write_gif(filename + '.gif')


def genetic_algorithm(system, numdistricts, precision, animate):
    solutions = [Solution(system, numdistricts) for _ in range(3)]
    for s in solutions:
        s.generate_random_solution()
    top3_history = [solutions]
    children_history = []
    for i in tqdm(range(precision)):
        new_solutions = []
        for _ in range(10):
            s1, s2 = np.random.choice(solutions, size=2)
            new_solutions.append(s1.combine(s2))
        full_solutions = new_solutions + solutions
        children_history.append(new_solutions)
        solutions = [_[0] for _ in
                    sorted([(s, s.value) for s in full_solutions],
                           key=lambda tup: -tup[1])[:3]]
        top3_history.append(solutions)

    solution = solutions[0]
    solution.count = precision
    solution.algo = 'Genetic Algorithm'
    print(solution)
    print(solution.summary())

    if animate:
        print('Saving to File')

        fig = plt.figure(figsize=(8, 8))
        ax1 = plt.subplot2grid((10, 10), (3, 0), colspan=4, rowspan=3)
        top3_axes = [plt.subplot2grid((10, 10), (2, 4 + i), colspan=3, rowspan=2)
                     for i in range(0, 5, 2)]
        children_axes = [plt.subplot2grid((10, 10), (i, 7), colspan=3)
                         for i in range(10)]

        ax1.imshow(system.matrix, interpolation='nearest')

        top3 = [ax.imshow(top3_history[0][i].full_mask, interpolation='nearest')
                for i, ax in enumerate(top3_axes)]
        children = [ax.imshow(children_history[0][i].full_mask, interpolation='nearest')
                    for i, ax in enumerate(children_axes)]

        # axarr[1].set_title('value {}'.format(history[0].value))

        def update_plot(i):
            for j, ax in enumerate(top3):
                ax.set_data(top3_history[i][j].full_mask)
            for j, ax in enumerate(children):
                ax.set_data(children_history[i][j].full_mask)
            # axarr[1].set_title('value {}'.format(history[i].value))
            plt.suptitle('Solution {}'.format(i))
            return top3[0],

        interval = int(60000.0 / len(top3_history))
        if interval == 0:
            interval = 1
        ani = animation.FuncAnimation(fig, update_plot, len(top3_history),
                                      interval=interval, blit=True)
        filename = 'simulated_annealing_solution_{}'.format(system.filename.split('.')[0])
        ani.save(filename + '.mp4')

        # editor.VideoFileClip(filename + '.mp4')\
        #         .write_gif(filename + '.gif')


class Solution(object):
    def __init__(self, system, numdistricts):
        self.system = system
        self.numdistricts = numdistricts
        if numdistricts is None:
            self.numdistricts = system.width + 1
        self.full_mask = np.zeros((system.height, system.width))
        self.algo = None
        self.count = 0

    def __getitem__(self, key):
        if key < 1 or key > self.numdistricts:
            raise KeyError('District does not exist!')
        else:
            new_mask = Mask()
            new_mask.parse_list(self.get_solution(key))
            return new_mask

    def __str__(self):
        return str(self.full_mask)

    def summary(self):
        sep = (40 * '-') + '\n'
        summary_string = ''
        summary_string += sep
        total_size, percents = self.system.stats
        summary_string += 'Total Population Size: {}\n'.format(total_size)
        summary_string += sep
        summary_string += 'Party Division in Population\n'
        for k, v in percents.items():
            summary_string += '{}: {:05f}\n'.format(k, v)
        summary_string += sep

        majorities = {k:0 for k in self.system.names.keys()}
        locations = []
        for i in range(1, self.numdistricts + 1):
            majorities[self.system._name_arr[self.majority(i)]] += 1
            locations.append(self[i].location)
        summary_string += 'Number of Districts with Majority by Party\n'
        for k, v in majorities.items():
            summary_string += '{}: {}\n'.format(k, v)
        summary_string += sep

        summary_string += 'District Locations (zero-indexed, [y, x])\n'
        for i, loc in enumerate(locations):
            loc_string = '\n\t'.join(str(tup) for tup in loc)
            summary_string += 'District {}:\n\t{}\n'.format(i + 1, loc_string)
        summary_string += sep

        summary_string += 'Algorithm: {}\n'.format(self.algo)
        summary_string += sep

        summary_string += 'Valid Solution States Explored: {}\n'.format(self.count)
        summary_string += sep

        return summary_string[:-1]

    def majority(self, i):
        district = self.system.matrix[self[i].mask.astype(bool)]
        if district.sum() > (len(district) / 2.0):
            return 1
        else:
            return 0

    def copy(self):
        new_sol = Solution(self.system, self.numdistricts)
        new_sol.full_mask = np.copy(self.full_mask)
        return new_sol

    def show(self, save=False, name='out.png'):
        fig, axarr = plt.subplots(1, 2, figsize=(8, 8))
        axarr[0].imshow(self.system.matrix, interpolation='nearest')
        axarr[1].imshow(self.full_mask, interpolation='nearest')
        axarr[1].set_title('Value: {}'.format(self.value))
        axarr[0].axis('off')
        axarr[1].axis('off')
        if save:
            plt.savefig(name)
        else:
            plt.show()

    @property
    def is_valid(self):
        """
        A valid solution is one that covers everything
        """
        valid = True
        if (self.full_mask == 0).any():
            valid = False
            return valid
        for i in range(1, self.numdistricts + 1):
            valid &= self[i].is_valid
        return valid

    @property
    def value(self):
        """
        This is our fitness function.

        We're trying to maximize similarity in districts, as well as make sure
        that the size of each district is at least 1.
        """
        value = 0
        if not self.is_valid:
            return value
        for i in range(1, self.numdistricts + 1):
            values = self.system.matrix[self[i].mask.astype(bool)]
            if len(values) == 0:
                value = 0
                return value
            else:
                value += np.abs(len(values[values == 0]) - len(values[values == 1]))
        return value

    def get_solution(self, i):
        return (self.full_mask == i).astype(int)

    def get_openspots(self, value):
        openspots = np.where(self.full_mask == value)
        if len(openspots[0]) == 1:
            choice = 0
        elif len(openspots[0]) == 0:
            return None, None
        else:
            choice = np.random.randint(0, len(openspots[0]) - 1)
        y = openspots[0][choice]
        x = openspots[1][choice]
        return y, x

    def generate_random_solution(self):
        """
        Solutions are not guaranteed to be equal in size, as if one gets boxed
        off it will stay small...
        """
        i = 1
        j = 0
        while (self.full_mask == 0).any():
            if j < self.numdistricts:
                y, x = self.get_openspots(0)
                self.full_mask[y, x] = i
            else:
                y, x = self.get_openspots(i)
                traversed = {(y, x)}
                while True:
                    neighbors = [(y + yi, x + xi)
                                 for xi in range(-1, 2)
                                 for yi in range(-1, 2)
                                 if (0 <= y + yi < self.system.height) and
                                    (0 <= x + xi < self.system.width) and
                                    not (x == 0 and y == 0) and
                                    (y + yi, x + xi) not in traversed and
                                    self.full_mask[y + yi, x + xi] in [i, 0]]
                    if len(neighbors) == 0:
                        break
                    for ii, jj in neighbors:
                        traversed.add((ii, jj))
                        if self.full_mask[ii, jj] == 0:
                            self.full_mask[ii, jj] = i
                            break
            i = (i % self.numdistricts) + 1
            j += 1

    def mutate(self):
        i = np.random.randint(1, self.numdistricts)
        y, x = self.get_openspots(i)
        if y is None:
            raise IndexError('No open spots? Something is real bad')
        traversed = set()
        q = queue.Queue()
        q.put((y, x))
        while not q.empty():
            y, x = q.get()
            traversed.add((y, x))

            if (self.full_mask[y, x] != i and
                    self[self.full_mask[y, x]].size > 1):
                self.full_mask[y, x] = i
                break

            neighbors = [(y + yi, x + xi)
                         for xi in range(-1, 2)
                         for yi in range(-1, 2)
                         if (0 <= y + yi < self.system.height) and
                            (0 <= x + xi < self.system.width) and
                            not (xi == 0 and yi == 0) and
                            (y + yi, x + xi) not in traversed]
            for ii, jj in neighbors:
                q.put((ii, jj))

    def combine(self, other_solution):
        """
        This is the combining function. We have a couple of options here

        1. Naively just mash things together until something looks right
            * Probably the approach most will use
        2. Use a procedural method to use elements of the two parents to produce
        a child
            * This is more likely to work out, as each individual solution has a
            much higher chance of success
            * Markov methods
            * Waveform collapsing (mmmmm this could be amazing)

        What we're going to do here is basically along the same lines as the
        "generate random" solution.
        """
        new_solution = Solution(self.system, self.numdistricts)
        initial_spots = [_ for _ in range(1, self.numdistricts + 1)]
        random.shuffle(initial_spots)
        districts = list(range(1, self.numdistricts + 1))
        i = 1
        j = 0
        while (new_solution.full_mask == 0).any():
            # First set all initial locations for spawns
            if len(initial_spots) != 0:
                district = initial_spots.pop()
                locations = (self[district].location +
                             other_solution[district].location)
                y, x = locations[random.randint(0, len(locations) - 1)]
                while new_solution.full_mask[y, x] != 0:
                    y, x = locations[random.randint(0, len(locations) - 1)]
                new_solution.full_mask[y, x] = district
            # Then we can expand each thing. Note, no new assignment, only
            # taking parts from parents
            else:
                y, x = self.get_openspots(i)
                traversed = {(y, x)}
                while True:
                    neighbors = [(y + yi, x + xi)
                                 for xi in range(-1, 2)
                                 for yi in range(-1, 2)
                                 if (0 <= y + yi < self.system.height) and
                                    (0 <= x + xi < self.system.width) and
                                    not (xi == 0 and yi == 0) and
                                    (y + yi, x + xi) not in traversed and
                                    new_solution.full_mask[y + yi, x + xi] in [i, 0]]
                    if len(neighbors) == 0:
                        break
                    for ii, jj in neighbors:
                        traversed.add((ii, jj))
                        if new_solution.full_mask[ii, jj] == 0:
                            new_solution.full_mask[ii, jj] = i
                            break
            i = (i % self.numdistricts) + 1
            j += 1

        if random.random() < 0.1:
            new_solution.mutate()
        return new_solution


class System(object):
    """
    Solely for reading in the file and keeping track of where things are
    """
    def __init__(self, filename):
        self.filename = filename
        self.matrix = None
        self.names = dict()
        self.num_names = 0
        self._read_file()

    def __getitem__(self, key):
        if key not in list(self.names.keys()):
            raise KeyError('{} does not exist'.format(key))
        raw_spots = np.where(self.matrix == self.names[key])
        spots = []
        for i in range(len(raw_spots[0])):
            spots.append([raw_spots[0][i], raw_spots[1][i]])
        return spots

    @property
    def width(self):
        return self.matrix.shape[1]

    @property
    def height(self):
        return self.matrix.shape[0]

    @property
    def _name_arr(self):
        return [_[0] for _ in
                sorted(self.names.items(),
                       key=lambda tup: tup[1])]

    @property
    def stats(self):
        size = self.width * self.height
        percents = {}
        for k in self.names.keys():
            percents[k] = len(self[k]) / float(size)
        return size, percents

    def _read_file(self):
        """
        We read in the file here. The input file needs to be of a very specific
        format, where there are m rows and n columns, with fields separated by a
        space.

        D R D R D R R R
        D D R D R R R R
        D D D R R R R R
        D D R R R R D R
        R R D D D R R R
        R D D D D D R R
        R R R D D D D D
        D D D D D D R D
        """
        width = 0
        height = 0
        system = []
        with open(self.filename, 'r') as fileobj:
            i = 0
            for line in [re.sub('\n', '', _) for _ in fileobj.readlines()]:
                items = line.split(' ')
                system.append(items)
                width = len(items)
                i += 1
            height = i
        self.matrix = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                try:
                    num = self.names[system[i][j]]
                except KeyError:
                    self.names[system[i][j]] = self.num_names
                    self.num_names += 1
                self.matrix[i, j] = self.names[system[i][j]]

    def empty_state(self):
        return np.zeros(self.matrix.shape)


class Mask(object):
    """
    This is the class that tracks each solution

    Solutions are easy, as they're in the form of a bitmask
    """
    def __init__(self, height=0, width=0):
        self.mask = np.zeros((height, width))
        self.width, self.height = width, height

    def __str__(self):
        return str(self.mask)

    def parse_list(self, listvals):
        self.mask = np.array(listvals)
        self.height, self.width = self.mask.shape

    @property
    def size(self):
        return self.mask.sum()

    @property
    def location(self):
        raw_spots = np.where(self.mask == 1)
        spots = []
        for i in range(len(raw_spots[0])):
            spots.append([raw_spots[0][i], raw_spots[1][i]])
        return spots

    @property
    def is_valid(self):
        """
        https://en.wikipedia.org/wiki/Connected-component_labeling
        """
        curlab = 1
        labels = np.zeros(self.mask.shape)
        q = queue.Queue()
        def unlabelled(i, j):
            return self.mask[i, j] == 1 and labels[i, j] == 0
        for i in range(self.height):
            for j in range(self.width):
                if unlabelled(i, j):
                    labels[i, j] = curlab
                    q.put((i, j))
                    while not q.empty():
                        y0, x0 = q.get()
                        neighbors = [(y0 + y, x0 + x)
                                     for x in range(-1, 2)
                                     for y in range(-1, 2)
                                     if (0 <= y0 + y < self.height) and
                                        (0 <= x0 + x < self.width) and
                                        not (x == 0 and y == 0)]
                        for ii, jj in neighbors:
                            if unlabelled(ii, jj):
                                labels[ii, jj] = curlab
                                q.put((ii, jj))
                    curlab += 1
        if curlab > 2:
            return False
        else:
            return True

    def overlap(self, mask):
        if ((self.mask + mask.mask) > 1).any():
            return True
        else:
            return False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', metavar='F', type=str, nargs=1,
                        help='File to load')
    parser.add_argument('-a', '--annealing', action='store_true',
                        default=False,
                        help='Use Simulated Annealing Algorithm?')
    parser.add_argument('-g', '--genetic', action='store_true',
                        default=False,
                        help='Use Genetic Algorithm?')
    parser.add_argument('-n', '--numdistricts', type=int, default=None,
                        help='Number of districts to form.')
    parser.add_argument('-z', '--animate', action='store_true', default=False,
                        help='Animate algorithms?')
    parser.add_argument('-p', '--precision', type=int, default=1000,
                        help='Tweak precision, lower is less.')
    args= parser.parse_args()
    args.filename = args.filename[0]
    return args


if __name__ == '__main__':
    sys.exit(main())
