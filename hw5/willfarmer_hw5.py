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
    if args.report:
        generate_report_assets(system, args.numdistricts, args.precision, args.gif)
    elif args.annealing:
        simulated_annealing(system, args.numdistricts, args.precision,
                args.animate, args.gif)
    elif args.genetic:
        genetic_algorithm(system, args.numdistricts, args.precision,
                args.animate, args.gif)
    else:
        print('Running in Demo Mode!!!')
        print('First we\'ll use Simulated Annealing')
        simulated_annealing(system, args.numdistricts, args.precision, False,
                False)
        print('Now we\'ll try the Genetic Algorithm')
        genetic_algorithm(system, args.numdistricts, args.precision, False,
                False)


def generate_report_assets(system, numdistricts, precision, makegif):
    # First just plot initial map
    plt.figure(figsize=(8, 8))
    plt.imshow(system.matrix, interpolation='nearest',
                    cmap=plt.get_cmap('cool'))
    plt.axis('off')
    plt.title(system.filename)
    plt.savefig(system.filename.split('.')[0] + '_initial.png')

    # Now generate random solution
    solution = Solution(system, numdistricts)
    solution_history = solution.generate_random_solution(history=True)
    animate_history(system.filename, system.matrix,
                    solution_history, solution.numdistricts, makegif,
                    algo_name='generate_random')
    # Now show mutation
    backup = solution.copy()
    fig, axarr = plt.subplots(1, 3, figsize=(8, 8))
    axarr[0].imshow(solution.full_mask, interpolation='nearest',
                    cmap=plt.get_cmap('gnuplot'))
    axarr[0].axis('off')
    axarr[0].set_title('Initial Solution')
    solution.mutate()
    axarr[1].imshow(solution.full_mask, interpolation='nearest',
                    cmap=plt.get_cmap('gnuplot'))
    axarr[1].axis('off')
    axarr[1].set_title('Mutated Solution')
    axarr[2].imshow(np.abs(backup.full_mask - solution.full_mask),
                    interpolation='nearest',
                    cmap=plt.get_cmap('gnuplot'))
    axarr[2].axis('off')
    axarr[2].set_title('Difference in Solutions')
    plt.savefig('mutation.png')

    # Now show combination
    solution.full_mask[:] = 0
    solution.generate_random_solution()
    fig, axarr = plt.subplots(1, 3, figsize=(8, 8))
    axarr[0].imshow(backup.full_mask, interpolation='nearest',
                    cmap=plt.get_cmap('gnuplot'))
    axarr[0].axis('off')
    axarr[0].set_title('Parent 1')
    axarr[1].imshow(solution.full_mask, interpolation='nearest',
                    cmap=plt.get_cmap('gnuplot'))
    axarr[1].axis('off')
    axarr[1].set_title('Parent 2')
    child = backup.combine(solution)
    axarr[2].imshow(child.full_mask, interpolation='nearest',
                    cmap=plt.get_cmap('gnuplot'))
    axarr[2].axis('off')
    axarr[2].set_title('Child')
    plt.savefig('combine.png')


def simulated_annealing(system, numdistricts, precision, animate, makegif):
    solution = Solution(system, numdistricts)
    solution.generate_random_solution()
    history = [solution]
    k = 0.8
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
        animate_history(system.filename, system.matrix,
                        history, solution.numdistricts,
                        makegif)


def genetic_algorithm(system, numdistricts, precision, animate, makegif):
    solutions = [Solution(system, numdistricts) for _ in range(3)]
    for s in solutions:
        s.generate_random_solution()
    top_history = []
    for i in tqdm(range(precision)):
        new_solutions = []
        for _ in range(10):
            s1, s2 = np.random.choice(solutions, size=2)
            new_solutions.append(s1.combine(s2))
        full_solutions = new_solutions + solutions
        solutions = [_[0] for _ in
                    sorted([(s, s.value) for s in full_solutions],
                           key=lambda tup: -tup[1])[:3]]
        top_history.append(solutions[0])

    solution = solutions[0]
    solution.count = precision
    solution.algo = 'Genetic Algorithm'
    print(solution)
    print(solution.summary())

    if animate:
        animate_history(system.filename, system.matrix,
                        top_history, solution.numdistricts,
                        makegif)


def animate_history(filename, systemdata, history, numdistricts, makegif, algo_name=None):
    print('Saving to File')
    fig, axarr = plt.subplots(1, 2, figsize=(8, 8))
    systemplot = axarr[0].imshow(systemdata, interpolation='nearest',
                                 cmap=plt.get_cmap('cool'))
    axarr[0].axis('off')
    sol = axarr[1].imshow(history[0].full_mask, interpolation='nearest',
                          cmap=plt.get_cmap('gnuplot'),
                          vmin=0,
                          vmax=numdistricts)
    axarr[1].set_title('value {}'.format(history[0].value))
    axarr[1].axis('off')

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
    if not algo_name:
        algo_name = re.sub(' ', '_', history[-1].algo.lower())
    filename = '{}_solution_{}'.format(algo_name, filename.split('.')[0])
    ani.save(filename + '.mp4')

    if makegif:
        editor.VideoFileClip(filename + '.mp4')\
                .write_gif(filename + '.gif')

    if history[-1].algo is not None:
        plt.figure(figsize=(8, 8))
        plt.imshow(history[-1].full_mask, interpolation='nearest',
                              cmap=plt.get_cmap('gnuplot'),
                              vmin=0,
                              vmax=numdistricts)
        plt.title(history[-1].algo + ' Final Solution')
        plt.axis('off')
        plt.savefig(filename + '.png')


class Solution(object):
    def __init__(self, system, numdistricts):
        self.system = system
        self.numdistricts = numdistricts
        if numdistricts is None:
            self.numdistricts = system.width
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
        summary_string += 'Score: {}\n'.format(self.value)
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
            loc_string = ','.join(str(tup) for tup in loc)
            summary_string += 'District {}:{}\n'.format(i + 1, loc_string)
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

    def get_full_openspots(self, value):
        openspots = np.where(self.full_mask == value)
        spots = []
        for i in range(len(openspots[0])):
            spots.append([openspots[0][i], openspots[1][i]])
        return spots

    def get_neighbors(self, y, x):
        neighbors = [(y + yi, x + xi)
                     for xi in range(-1, 2)
                     for yi in range(-1, 2)
                     if (0 <= y + yi < self.system.height) and
                        (0 <= x + xi < self.system.width) and
                        not (xi == 0 and yi == 0)]
        return neighbors

    def get_district_neighbors(self, i):
        y, x = self.get_openspots(i)
        q = queue.Queue()
        q.put((y, x))
        edges = []
        labels = np.zeros(self.full_mask.shape)
        while not q.empty():
            y, x = q.get()
            labels[y, x] = 1
            if self.full_mask[y, x] == i:
                for yi, xi in self.get_neighbors(y, x):
                    if labels[yi, xi] == 0:
                        q.put((yi, xi))
                        labels[yi, xi] = 1
            else:
                edges.append((y, x))
        return edges

    def get_filtered_district_neighbors(self, i, filter_list):
        return [(y, x) for y, x in self.get_district_neighbors(i)
                if self.full_mask[y, x] in filter_list]

    def fill(self, keep_history=False):
        districts = list(range(1, self.numdistricts + 1))
        history = []
        while (self.full_mask == 0).any():
            i = districts[random.randint(0, len(districts) - 1)]
            neighbors = self.get_filtered_district_neighbors(i, [0])
            if len(neighbors) == 0:
                districts.remove(i)
            else:
                y, x = neighbors[random.randint(0, len(neighbors) - 1)]
                self.full_mask[y, x] = i
                if keep_history:
                    history.append(self.copy())
        return history

    def generate_random_solution(self, history=False):
        """
        Solutions are not guaranteed to be equal in size, as if one gets boxed
        off it will stay small...
        """
        solution_history = [self.copy()]
        for i in range(1, self.numdistricts + 1):
            y, x = self.get_openspots(0)
            self.full_mask[y, x] = i
            if history:
                solution_history.append(self.copy())
        solution_history += self.fill(keep_history=history)
        if history:
            return solution_history

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

            neighbors = [_ for _ in self.get_neighbors(y, x)
                         if _ not in traversed]
            for ii, jj in neighbors:
                q.put((ii, jj))

    def combine(self, other_solution):
        new_solution = Solution(self.system, self.numdistricts)

        pick_order = [self, other_solution]
        random.shuffle(pick_order)
        cursor = 0
        for i in range(1, self.numdistricts + 1):
            parent_locations = pick_order[cursor][i].location
            open_locations = new_solution.get_full_openspots(0)
            combined = [(y, x) for y, x in parent_locations
                        if [y, x] in open_locations]
            for y, x in combined:
                new_solution.full_mask[y, x] = i
            cursor ^= 1
        for i in range(1, self.numdistricts + 1):
            y, x = new_solution.get_openspots(i)
            if y is None:
                y, x = new_solution.get_openspots(0)
                new_solution.full_mask[y, x] = i
        new_solution.fill()
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
    parser.add_argument('-r', '--report', action='store_true', default=False,
                        help='Generate Animations for the report')
    parser.add_argument('-j', '--gif', action='store_true', default=False,
                        help='Generate gif versions')
    args= parser.parse_args()
    args.filename = args.filename[0]
    return args


if __name__ == '__main__':
    sys.exit(main())
