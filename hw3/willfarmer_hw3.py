#!/usr/bin/env python3

import sys
import argparse
import re

import numpy
import networkx as nx
import matplotlib.pyplot as plt
import seaborn

import random
import time


def main():
    args = get_args()

    g = Graph()
    for filename in args.file:
        with open(filename, 'r') as filebuff:
            # parse the shitty formatted file
            for line in [l[:-1] for l in filebuff.readlines() if re.match(r'[\[A-Za-z]', l)]:
                if line[0] == '[':
                    edge_data = line.split(',')
                    g.add_edge(edge_data[0][1:], edge_data[1], weight=float(edge_data[2][:-1]))
                else:
                    node_data = line.split('=')
                    g[node_data[0]].estimate = float(node_data[1])
    g.plot()
    print(g.shortest_path_dijkstra('S', 'F'))
    print(g.shortest_path_Astar('S', 'F'))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', metavar='F', type=str, nargs='+',
                        help='Graph file(s) to load')
    args = parser.parse_args()
    return args


class Node(object):
    def __init__(self, name, weight=1):
        self.name = name
        self.weight = weight
        self.edges = []

    @property
    def neighbors(self):
        return [e.end.name for e in self.edges]

    def get_edge_weight(self, end):
        weight = None
        for edge in self.edges:
            if edge.end.name == end:
                weight = edge.weight
        if weight is None:
            raise EdgeException('Edge does not exist!')
        else:
            return weight


class Edge(object):
    def __init__(self, start, end, weight=1):
        self.start = start
        self.end = end
        self.weight = weight


class Graph(object):
    """
    Directed graph with weighted edges (default weight of 1)

    Stored as adjacency list

    Nodes can be classes but int(node) needs to return a number
    """
    def __init__(self):
        self.graph = {}
        self.c = 5
        self.infinity = 1e8
        self.inf = self.infinity
        self.q = BMinHeap()

    def __getitem__(self, key):
        try:
            return self.graph[key]
        except KeyError:
            raise NodeException("Node does not exist")

    @property
    def size(self):
        return len(self.graph.keys())

    @property
    def nodes(self):
        return list(self.graph.keys())

    @property
    def edges(self):
        return list([(k, v) for k, v in self.graph.items()])

    def generate_erdosrenyi(self, node_total, c=5):
        """ Generate random graph """
        self.c = c
        p = c / (node_total - 1)
        letter = 96
        for n in range(node_total):
            self.add_node(chr(letter + n))
        for start in self.graph.keys():
            for end in self.graph.keys():
                if start[0] != end[0]:
                    if random.random() < p:
                        self.add_edge(start[0], end[0])

    def to_networkx(self):
        G = nx.DiGraph()
        for node in self.nodes:
            G.add_node(node)
        for node in self.nodes:
            for edge in self.graph[node].edges:
                G.add_edge(node, edge.end.name, weight=edge.weight)
        return G

    def plot(self):
        g = self.to_networkx()
        plt.figure()
        layout = nx.spectral_layout(g, weight=None, scale=5)
        nx.draw_networkx(g, layout)
        plt.savefig('./graph.png')

    def add_node(self, name):
        try:
            self.graph[name]
            raise NodeException('Node Already Exists!')
        except KeyError:
            node = Node(name)
            self.graph[node.name] = node

    def add_edge(self, start, end, weight=1):
        """ adds edge (and node if not exist) """
        for val in [start, end]:
            try:
                self.graph[val]
            except KeyError:
                node = Node(val)
                self.graph[val] = node
        edge = Edge(self.graph[start], self.graph[end], weight=weight)
        self.graph[start].edges.append(edge)

    def find_adjacent(self, name):
        """
        Return adjacent nodes
        """
        try:
            return self.graph[name].edges
        except KeyError:
            raise NodeException('Node Does not Exist!')

    def to_edge_list(self):
        """
        Return an edge list representation of graph
        """
        edges = []
        for name, cls in self.graph.items():
            for e in cls.edges:
                edges.append((name, e.end.name))
        return sorted(edges, key=lambda tup: tup[0])

    def shortest_path_Astar(self, s, e):
        q = self.q                                                        # setup queue
        distances = {name: self.inf for name in self.nodes}               # set distances
        previous = {name: None for name in self.nodes}                    # Establish previous: [None, ...]
        distances[s] = 0                                                  # Set source distance as zero
        heuristic = {name: self.inf for name in self.nodes}
        heuristic[s] = 0
        [q.push((k, v)) for k, v in heuristic.items()]                    # add everything to minheap by heuristic
        num_evaluated = 0
        while len(q.queue) > 0:                                           # while we have things to look for
            u = q.find_min()[0]                                           # find the smallest weight so far
            if u is None or u == e:                                       # if one doesn't exist, or we're at the end, we're done
                break
            qc = [x[0] for x in q.queue]                                  # everything in queue
            for v in self.graph[u].neighbors:                             # examine neighbors of u
                num_evaluated += 1
                if v in qc:                                               # if we haven't looked at this neighbor yet
                    alt = distances[u] + self.graph[u].get_edge_weight(v) # what's its distance?
                    if alt < distances[v]:                                # if this distance is smaller than encountered so far
                        distances[v] = alt                                # set to current
                        previous[v] = u
                        q.adjust(v, distances[v] + (heuristic[e] - heuristic[v]))                                  # adjust minheap
        # Reconstruct shortest path
        if not any([v for k, v in previous.items()]):
            r = []
        else:
            r = []
            ne = e
            while True:
                r.insert(0, ne)
                if previous[ne] is None:
                    break
                ne = previous[ne]

        if len(r) == 1 and r[len(r) - 1] != s:
            return [], distances[e], previous, distances
        return r, distances[e], previous, distances, num_evaluated

    def shortest_path_dijkstra(self, s, e):
        q = self.q                                                        # setup queue
        distances = {name: self.inf for name in self.nodes}               # set distances
        previous = {name: None for name in self.nodes}                    # Establish previous: [None, ...]
        distances[s] = 0                                                  # Set source distance as zero
        [q.push((k, v)) for k, v in distances.items()]                    # add everything to minheap
        num_evaluated = 0
        while len(q.queue) > 0:                                           # while we have things to look for
            u = q.find_min()[0]                                           # find the smallest weight so far
            if u is None or u == e:                                       # if one doesn't exist, or we're at the end, we're done
                break
            qc = [x[0] for x in q.queue]                                  # everything in queue
            for v in self.graph[u].neighbors:                             # examine neighbors of u
                num_evaluated += 1
                if v in qc:                                               # if we haven't looked at this neighbor yet
                    alt = distances[u] + self.graph[u].get_edge_weight(v) # what's its distance?
                    if alt < distances[v]:                                # if this distance is smaller than encountered so far
                        distances[v] = alt                                # set to current
                        previous[v] = u
                        q.adjust(v, alt)                                  # adjust minheap
        # Reconstruct shortest path
        if not any([v for k, v in previous.items()]):
            r = []
        else:
            r = []
            ne = e
            while True:
                r.insert(0, ne)
                if previous[ne] is None:
                    break
                ne = previous[ne]

        if len(r) == 1 and r[len(r) - 1] != s:
            return [], distances[e], previous, distances
        return r, distances[e], previous, distances, num_evaluated


class BMinHeap:
    """
    Priority Queue (Min-Heap)
    """
    def __init__(self):
        self.queue = []  # Defined queue
        self.infinity = 1e8

    def adjust(self, name, val):
        for n in range(len(self.queue)):
            if self.queue[n][0] == name:
                new = (self.queue[n][0], val)
                self.queue.pop(n)
                self.push(new)
                break
        self.__min_heapify__(0)

    def push(self, num):
        """
        Insert new node into min-heap
        """
        self.queue.append(num)                                          # Add to lowest level
        for i in range(len(self.queue) - 1, -1, -1):                     # Iterate through children to parents
            child = self.queue[i]
            parent = self.queue[int((i - 1) / 2)]
            if child[1] < parent[1]:
                self.__swap__(i, int((i - 1) / 2))                       # Swap the two

    def __swap__(self, i, j):
        self.queue[i], self.queue[j] = self.queue[j], self.queue[i]

    def find_min(self):
        if len(self.queue) == 0:
            return None
        heapmin = self.queue.pop(0)  # It will always be the first
        if len(self.queue) > 0:
            self.queue.insert(0, self.queue.pop(len(self.queue) - 1))
            self.__min_heapify__(0)      # Re-order tree
        return heapmin

    def __min_heapify__(self, i):
        parent = i   # Current number
        left   = 2 * i + 1  # Left child
        right  = 2 * i + 2  # Right Child
        if (left < len(self.queue) and
                self.queue[left][1] < self.queue[parent][1]):
            parent = left
        if (right < len(self.queue) and
                self.queue[right][1] < self.queue[parent][1]):
            parent = right
        if parent != i:                                              # If smallest changed, adjust
            self.__swap__(i, parent)
            self.__min_heapify__(parent)


class EdgeException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class NodeException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


if __name__ == '__main__':
    sys.exit(main())
