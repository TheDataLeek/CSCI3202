#!/usr/bin/env python2.7

import sys
import argparse
import re
import numpy as np
import queue


def main():
    args = get_args()
    system = System(args.filename)
    if args.annealing:
        simulated_annealing(system)
    elif args.genetic:
        genetic_algorithm(system)


def simulated_annealing(system):
    pass


def genetic_algorithm(system):
    pass


class Solution(object):
    def __init__(self, system, numdistricts):
        self.system = system
        self.numdistricts = numdistricts
        if numdistricts is None:
            self.numdistricts = system.width + 1
        self.full_mask = np.zeros((system.height, system.width))

    @property
    def is_valid(self):
        """
        A valid solution is one that covers everything
        """
        return (self.full_mask == 1).all()

    def generate_random_solution(self):
        i = 1
        j = 0
        while (self.full_mask == 0).any():
            print(self.full_mask)
            if j < self.numdistricts:
                openspots = np.where(self.full_mask == 0)
                x = np.random.choice(openspots[0])
                y = np.random.choice(openspots[1])
                self.full_mask[x, y] = i
            else:
                openspots = np.where(self.full_mask == i)
                x = np.random.choice(openspots[0])
                y = np.random.choice(openspots[1])
                traversed = {(x, y)}
                while True:
                    neighbors = [(y + yi, x + xi)
                                 for xi in range(-1, 2)
                                 for yi in range(-1, 2)
                                 if (0 <= y + yi < self.system.height) and
                                    (0 <= x + xi < self.system.width) and
                                    not (x == 0 and y == 0) and
                                    (x + xi, y + yi) not in traversed and
                                    self.full_mask[y + yi, x + xi] in [i, 0]]
                    print(y, x)
                    print(neighbors)
                    if len(neighbors) == 0:
                        break
                    for ii, jj in neighbors:
                        traversed.add((jj, ii))
                        if self.full_mask[ii, jj] == 0:
                            self.full_mask[ii, jj] = i
                            break
            i = (i % self.numdistricts) + 1
            j += 1
        print(self.full_mask)


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

    def _read_file(self):
        """
        We read in the file here. The input file needs to be of a very specific
        format, where there are m rows and n columns, with fields separated by a
        space.
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

    def parse_list(self, listvals):
        self.mask = np.array(listvals)
        self.height, self.width = self.mask.shape

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
                        print(y0, x0)
                        print(neighbors)
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
    args= parser.parse_args()
    args.filename = args.filename[0]
    return args


if __name__ == '__main__':
    sys.exit(main())
