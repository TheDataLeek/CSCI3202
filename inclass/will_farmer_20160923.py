#!/usr/bin/env python3.5

import sys
import argparse
import re

import math

import random
import time


###############################################################
# From homework 3
###############################################################
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
        plt.axis('off')
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
###############################################################
# End from homework 3
###############################################################

g = Graph()
g.add_edge('A', 'B', weight=185)
g.add_edge('B', 'A', weight=185)
g.add_edge('A', 'C', weight=119)
g.add_edge('C', 'A', weight=119)
g.add_edge('A', 'D', weight=152)
g.add_edge('D', 'A', weight=152)
g.add_edge('A', 'E', weight=133)
g.add_edge('E', 'A', weight=133)
g.add_edge('B', 'C', weight=121)
g.add_edge('C', 'B', weight=121)
g.add_edge('B', 'D', weight=150)
g.add_edge('D', 'B', weight=150)
g.add_edge('B', 'E', weight=120)
g.add_edge('E', 'B', weight=120)
g.add_edge('C', 'D', weight=174)
g.add_edge('D', 'C', weight=174)
g.add_edge('C', 'E', weight=200)
g.add_edge('E', 'C', weight=200)
g.add_edge('D', 'E', weight=199)
g.add_edge('E', 'D', weight=199)

def generate_solution(g):
    nodes = g.nodes
    path = []
    cost = 0
    for i in range(len(nodes)):
        choice = random.choice(nodes)
        path.append(choice)
        nodes.remove(choice)
        if len(path) > 1:
            cost += g[path[-2]].get_edge_weight(path[-1])
    cost += g[path[-1]].get_edge_weight(path[0])
    return path, cost

def generate_solution_neighbor(g, s, c):
    flip = random.randint(0, len(s) - 2)
    new_path = s
    new_path[flip], new_path[flip + 1] = new_path[flip + 1], new_path[flip]
    new_cost = 0
    for i in range(len(s) - 1):
        new_cost += g[new_path[i]].get_edge_weight(new_path[i + 1])
    new_cost += g[new_path[-1]].get_edge_weight(new_path[0])
    return new_path, new_cost

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
        # print(s, c, math.exp(-DE / (k * T)))
        if DE < 0:
            s = sp
            c = cp
        elif random.random() < math.exp(-DE / (k * T)):
            s = sp
            c = cp
        T *= alpha
        i += 1
    print(s, c, i)

simulated_annealing(g)
