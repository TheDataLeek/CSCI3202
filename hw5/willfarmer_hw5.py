#!/usr/bin/env python2.7

"""
Simulated Annealing and Genetic Algorithm solutions to districting problem.

Notes on implementation:
    * I like properties and I use them in a couple spots...
    * Use `pip install --user -r requirements.txt` on the requirements file
    available in the root of this git repository.
    * If by some strange happenstance you only have this file, go to the
    following url to get the entire repo.
    https://github.com/willzfarmer/CSCI3202/tree/master/hw5
    * Test coverage is around 80% which I'm happy with. All the super important
    things are tested.
"""

# stdlib imports
import sys
import argparse
import math
import random
import re
import itertools
import queue
import time

# pypi imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import moviepy
from moviepy import editor
from tqdm import tqdm


FIGSIZE = (4, 4)


def main():
    args = get_args()
    system = System(args.filename)
    if args.full:
        generate_report_assets(system, args.numdistricts, 1000, True)
        simulated_annealing(system, args.numdistricts, 1000, True, True)
        genetic_algorithm(system, args.numdistricts, 1000, True, True)
    elif args.report:
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
        simulated_annealing(system, args.numdistricts, args.precision,
                            False, False)
        print('Now we\'ll try the Genetic Algorithm')
        genetic_algorithm(system, args.numdistricts, args.precision,
                          False, False)


def simulated_annealing(system, numdistricts, precision, animate, makegif):
    """
    Perform simulated annealing on our system with a series of progressively
    improving solutions.
    """
    solution = Solution(system, numdistricts)
    solution.generate_random_solution()  # start with random solution
    history = [solution]  # Keep track of our history
    k = 0.4  # Larger k => more chance of randomly accepting
    Tvals = np.arange(1, 1e-12, -1.0 / precision)
    for i, T in tqdm(enumerate(Tvals), total=len(Tvals)):
        new_solution = solution.copy()  # copy our current solution
        new_solution.mutate()  # Mutate the copy
        # TODO: Speed this up by keeping current value
        dv = new_solution.value - solution.value  # Look at delta of values
        # If it's better, or random chance, we accept it
        if dv > 0 or random.random() < math.exp(dv / (k * T)):
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
    """
    Use a genetic algorithm to find a good solution to our district problem
    """
    # Start with random initial solution space (3)
    solutions = [Solution(system, numdistricts) for _ in range(3)]
    for s in solutions:
        s.generate_random_solution()  # Initialize our solutions
    top_history = []  # Keep history of our top solution from each "frame"
    for i in tqdm(range(precision)):
        new_solutions = []
        for _ in range(10):  # Create 10 children per frame
            s1, s2 = np.random.choice(solutions, size=2)
            # Randomly combine two parents
            new_solutions.append(s1.combine(s2))
        # Combine everything, giving 13 total solutions
        full_solutions = new_solutions + solutions
        # Keep the top 3 for next generation
        solutions = [_[0] for _ in
                     sorted([(s, s.value) for s in full_solutions],
                            key=lambda tup: -tup[1])[:3]]
        # Only record top from generation, and only if it's changed
        if len(top_history) == 0 or solutions[0] != top_history[-1]:
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


def generate_report_assets(system, numdistricts, precision, makegif):
    """
    Responsible for generating all plots and animations specific to the writeup.
    In order this includes the following.

    1. Basic initial voting areas
    2. Random solution progression
    3. Mutation demonstration
    4. Genetic algorithm combination demonstration
    """
    # First just plot initial map
    plt.figure(figsize=FIGSIZE)
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
    fig, axarr = plt.subplots(1, 3, figsize=FIGSIZE)
    axarr[0].imshow(solution.full_mask, interpolation='nearest',
                    cmap=plt.get_cmap('gnuplot'))
    axarr[0].axis('off')
    axarr[0].set_title('Initial')
    solution.mutate()
    axarr[1].imshow(solution.full_mask, interpolation='nearest',
                    cmap=plt.get_cmap('gnuplot'))
    axarr[1].axis('off')
    axarr[1].set_title('Mutant')
    axarr[2].imshow(np.abs(backup.full_mask - solution.full_mask),
                    interpolation='nearest',
                    cmap=plt.get_cmap('gnuplot'))
    axarr[2].axis('off')
    axarr[2].set_title('Difference')
    plt.savefig('mutation.png')

    # Now show combination
    solution.full_mask[:] = 0
    solution.generate_random_solution()
    fig, axarr = plt.subplots(1, 3, figsize=FIGSIZE)
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


def animate_history(filename, systemdata, history, numdistricts, makegif, algo_name=None):
    """
    Take our given solution history, and animate it using matplotlib.animate.
    Save to gif if asked.
    """
    print('Saving to File')
    fig, axarr = plt.subplots(1, 2, figsize=FIGSIZE)
    # Plot our "field"
    systemplot = axarr[0].imshow(systemdata, interpolation='nearest',
                                 cmap=plt.get_cmap('cool'))
    axarr[0].axis('off')
    # Plot our first solution
    sol = axarr[1].imshow(history[0].full_mask, interpolation='nearest',
                          cmap=plt.get_cmap('gnuplot'),
                          vmin=0,
                          vmax=numdistricts)
    axarr[1].set_title('value {}'.format(history[0].value))
    axarr[1].axis('off')

    def update_plot(i):
        """Animation loop"""
        sol.set_data(history[i].full_mask)
        axarr[1].set_title('value {}'.format(history[i].value))
        plt.suptitle('Solution {}'.format(i))
        return sol,

    # Set interval so that things always last 60s or to 100
    interval = max(int(60000.0 / len(history)), 100)
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

    # Save final solution as separate image
    if history[-1].algo is not None:
        plt.figure(figsize=FIGSIZE)
        plt.imshow(history[-1].full_mask, interpolation='nearest',
                   cmap=plt.get_cmap('gnuplot'),
                   vmin=0,
                   vmax=numdistricts)
        plt.title(history[-1].algo + ' Final Solution')
        plt.axis('off')
        plt.savefig(filename + '.png')


class Solution(object):
    """This is our unique solution class"""
    def __init__(self, system, numdistricts):
        self.system = system
        self.numdistricts = numdistricts
        if numdistricts is None:  # If user doesn't specify
            self.numdistricts = system.width
        # Our solution is simply a numpy array
        self.full_mask = np.zeros((system.height, system.width))
        self.algo = None
        self.count = 0

    def __getitem__(self, key):
        """Allows us to easily index each district and get a Mask back"""
        if key < 1 or key > self.numdistricts:
            raise KeyError('District does not exist!')
        else:
            new_mask = Mask()  # initialize new empty mask
            # Set mask from district
            new_mask.parse_list(self.get_solution(key))
            return new_mask

    def __str__(self):
        """String version is just the string version of numpy array"""
        return str(self.full_mask)

    def summary(self):
        """This is literally only here for the grading..."""
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
        """
        Tell us who has majority in the specified district
        """
        district = self.system.matrix[self[i].mask.astype(bool)]
        if district.sum() > (len(district) / 2.0):
            return 1
        else:
            return 0

    def copy(self):
        """
        So... Numpy uses memory instances of arrays, meaning you need to tell it
        to actually copy the damn thing otherwise messing with the first will
        mess with all of its successors

        This was a bad bug...
        """
        new_sol = Solution(self.system, self.numdistricts)
        new_sol.full_mask = np.copy(self.full_mask)
        return new_sol

    def show(self, save=False, name='out.png'):
        """Debug function for individual plotting. Deprecated."""
        fig, axarr = plt.subplots(1, 2, figsize=FIGSIZE)
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
        A valid solution is one that covers everything. So we do two things
        here, first of which is to make sure that no element in the mask is
        zero, and second check that each district is valid.
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
        if not self.is_valid:  # if we don't have a valid solution, return 0
            return value
        # Sum up values of each district
        for i in range(1, self.numdistricts + 1):
            values = self.system.matrix[self[i].mask.astype(bool)]
            if len(values) == 0:
                value = 0
                return value
            else:
                # District value is simply abs(num_red - num_blue)
                subvalue = np.abs(len(values[values == 0]) - len(values[values == 1]))
                if subvalue < len(values):
                    # For any non-uniform values, add 10% their value to account
                    # for independent voter turnout
                    subvalue += (len(values) - subvalue) * 0.1
                value += subvalue
        return value

    def get_solution(self, i):
        """
        Return array just showing district

        If our full_mask looks like this
            [[1, 1, 2],
             [1, 2, 3],
             [2, 2, 3]]
        This function returns the following when i=2
            [[0, 0, 1],
             [0, 1, 0],
             [1, 1, 0]]
        """
        return (self.full_mask == i).astype(int)

    def get_openspots(self, value):
        """
        Return a random location where our full mask is equal to value

        If our full_mask is
            [[1, 1, 2],
             [1, 2, 3],
             [2, 2, 3]]
        self.get_openspots(1) could return any of
            [[0, 0], [0, 1], [1, 0]]
        """
        openspots = np.where(self.full_mask == value)
        if len(openspots[0]) == 1:
            choice = 0
        elif len(openspots[0]) == 0:
            return None, None  # if no spots exist, return None
        else:
            choice = np.random.randint(0, len(openspots[0]) - 1)
        y = openspots[0][choice]
        x = openspots[1][choice]
        return y, x

    def get_full_openspots(self, value):
        """
        Instead of just returning one random openspot, return all of them.

        If our full_mask is
            [[1, 1, 2],
             [1, 2, 3],
             [2, 2, 3]]
        self.get_full_openspots(1) will return (not necessarily sorted)
            [[0, 0], [0, 1], [1, 0]]
        """
        openspots = np.where(self.full_mask == value)
        spots = []
        for i in range(len(openspots[0])):
            spots.append([openspots[0][i], openspots[1][i]])
        return spots

    def get_neighbors(self, y, x):
        """
        Get all neighbors of a point that fall within boundary

        If our full_mask is
            [[1, 1, 2],
             [1, 2, 3],
             [2, 2, 3]]
        self.get_neighbors(0, 1) will return (not necessarily sorted)
            [[0, 0], [1, 0], [1, 1], [1, 2], [0, 2]]
        """
        neighbors = [(y + yi, x + xi)
                     for xi in range(-1, 2)
                     for yi in range(-1, 2)
                     if (0 <= y + yi < self.system.height) and
                     (0 <= x + xi < self.system.width) and
                     not (xi == 0 and yi == 0)]
        return neighbors

    def get_district_neighbors(self, i):
        """
        Get all points on the edge of a district

        If our full_mask is
            [[1, 1, 2],
             [1, 2, 3],
             [2, 2, 3]]
        self.get_district_neighbors(1) will return (not necessarily sorted)
            [[2, 0], [2, 1], [1, 1], [1, 2], [0, 2]]
        """
        y, x = self.get_openspots(i)
        q = queue.Queue()
        q.put((y, x))
        edges = []
        labels = np.zeros(self.full_mask.shape)
        labels[y, x] = 1
        while not q.empty():
            y, x = q.get()
            if self.full_mask[y, x] == i:
                for yi, xi in self.get_neighbors(y, x):
                    if labels[yi, xi] == 0:
                        q.put((yi, xi))
                        labels[yi, xi] = 1
            else:
                edges.append((y, x))
        return edges

    def get_filtered_district_neighbors(self, i, filter_list):
        """
        Simply a handy filter on get_district_neighbors. Only includes values
        that fall into the filter list

        If our full_mask is
            [[1, 1, 2],
             [1, 2, 3],
             [2, 2, 3]]
        self.get_filtered_district_neighbors(1, [2]) will return (not necessarily
        sorted)
            [[2, 0], [2, 1], [1, 1], [0, 2]]
        """
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
        Generate a random solution by picking spawn points and filling around
        them.

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
        """
        Pick a random district, find a random neighbor, and if the other
        district is at least size 2, replace the point with our district
        """
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
        """
        Look at both solutions, alternate between them randomly, and try to
        basically inject one side at a time. Afterwards fill the gaps in with
        fill()
        """
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
        """
        Again, lets us access with self[i], and just return every index where
        our matrix is equal to 'D' or 'R'
        """
        if key not in list(self.names.keys()):
            raise KeyError('{} does not exist'.format(key))
        raw_spots = np.where(self.matrix == self.names[key])
        spots = []
        for i in range(len(raw_spots[0])):
            spots.append([raw_spots[0][i], raw_spots[1][i]])
        return spots

    @property
    def width(self):
        """Just the width of the system"""
        return self.matrix.shape[1]

    @property
    def height(self):
        """Just the height of the system"""
        return self.matrix.shape[0]

    @property
    def _name_arr(self):
        """Internal use, in order list of names ['D', 'R'] probably"""
        return [_[0] for _ in
                sorted(self.names.items(),
                       key=lambda tup: tup[1])]

    @property
    def stats(self):
        """For grading, returns size of system, percent of each party"""
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
        """Return an empty version of the system. Deprecated."""
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
        """Numpy string version of array"""
        return str(self.mask)

    def parse_list(self, listvals):
        """given some entry list, set our mask to be those vals"""
        self.mask = np.array(listvals)
        self.height, self.width = self.mask.shape

    @property
    def size(self):
        """Number of elements in mask"""
        return self.mask.sum()

    @property
    def location(self):
        """List of locations where mask == 1"""
        raw_spots = np.where(self.mask == 1)
        spots = []
        for i in range(len(raw_spots[0])):
            spots.append([raw_spots[0][i], raw_spots[1][i]])
        return spots

    @property
    def is_valid(self):
        """
        Valid masks have a single connected component.

        https://en.wikipedia.org/wiki/Connected-component_labeling

        This is what inspired much of the other code, this pattern is repeated
        throughout the code.
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
        """Tells us if two masks overlap. Deprecated"""
        if ((self.mask + mask.mask) > 1).any():
            return True
        else:
            return False


def get_args():
    """Get our arguments"""
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
                        help=('Number of districts to form. Defaults to the '
                              'width of the system'))
    parser.add_argument('-z', '--animate', action='store_true', default=False,
                        help='Animate algorithms?')
    parser.add_argument('-p', '--precision', type=int, default=1000,
                        help=('Tweak precision, lower is less. '
                              'In a nutshell, how many loops to run.'))
    parser.add_argument('-r', '--report', action='store_true', default=False,
                        help='Generate all assets for the report')
    parser.add_argument('-j', '--gif', action='store_true', default=False,
                        help='Generate gif versions of animations?')
    parser.add_argument('-F', '--full', action='store_true', default=False,
                        help='Generate everything. Report assets, SA, and GA.')
    args = parser.parse_args()
    args.filename = args.filename[0]  # We only allow 1 file at a time.
    return args


if __name__ == '__main__':
    sys.exit(main())
