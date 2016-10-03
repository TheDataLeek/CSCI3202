#!/usr/bin/env python2.7

import sys
import argparse
import collections
import re
import numpy as np
import queue


def main():
    args = get_args()
    system = System(args.filename)
    if args.annealing:
        simulated_annealing(system, args.numdistricts)
    elif args.genetic:
        genetic_algorithm(system, args.numdistricts)


def simulated_annealing(system, numdistricts):
    pass


def genetic_algorithm(system, numdistricts):
    initial = Solution(system, numdistricts)
    initial.generate_random_solution()
    initial.value


class Solution(object):
    def __init__(self, system, numdistricts):
        self.system = system
        self.numdistricts = numdistricts
        if numdistricts is None:
            self.numdistricts = system.width + 1
        self.full_mask = np.zeros((system.height, system.width))

    def __getitem__(self, key):
        if key < 1 or key > self.numdistricts:
            raise KeyError('District does not exist!')
        else:
            new_mask = Mask()
            new_mask.parse_list(self.get_solution(key))
            return new_mask

    def __str__(self):
        return str(self.full_mask)

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
        value = 0
        if not self.is_valid:
            return value
        for i in range(1, self.numdistricts + 1):
            values = self.system.matrix[self[i].mask.astype(bool)]
            print(collections.Counter(values))
            break


    def get_solution(self, i):
        return (self.full_mask == i).astype(int)

    def generate_random_solution(self):
        """
        Solutions are not guaranteed to be equal in size, as if one gets boxed
        off it will stay small...
        """
        i = 1
        j = 0
        while (self.full_mask == 0).any():
            if j < self.numdistricts:
                openspots = np.where(self.full_mask == 0)
                y = np.random.choice(openspots[0])
                x = np.random.choice(openspots[1])
                self.full_mask[y, x] = i
                assert(self.full_mask.sum() == sum(range(i + 1)))
            else:
                openspots = np.where(self.full_mask == i)
                y = np.random.choice(openspots[0])
                x = np.random.choice(openspots[1])
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

    @property
    def width(self):
        return self.matrix.shape[1]

    @property
    def height(self):
        return self.matrix.shape[0]

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
