#!/usr/bin/env python3.5

import sys
from typing import List


# Problem 1 ###################################################################
class Stack(object):
    def __init__(self):
        self.stack = []

    def push(self, number: int) -> None:
        self.stack.append(number)

    def pop(self) -> int:
        return self.stack.pop()

    def check_size(self) -> int:
        return len(self.stack)


# Problem 2 ###################################################################
class Node(object):
    def __init__(self, value: int,
                 left_child=None, right_child=None, parent=None):
        self.value       = value
        self.left_child  = left_child
        self.right_child = right_child
        self.parent      = parent

    @property
    def children(self):
        return [self.left_child, self.right_child]

    @property
    def empty(self) -> bool:
        if self.left_child is None and self.right_child is None:
            return True
        else:
            return False

    def __str__(self):
        return (str(self.value) +
                (' -> ({})'.format(', '.join([str(c) for c in self.children
                                              if c is not None]))
                    if not self.empty
                    else ''))

    def getChildren(self) -> List:
        """Here for compatibility with bad code"""
        return [self.left_child, self.right_child]


# Problem 3 ###################################################################
class Tree(object):
    def __init__(self, rootkey: int):
        self.root = Node(rootkey)

    def __str__(self):
        return str(self.root)

    def add(self, node_key: int, parent_key: int) -> None:
        parent = self.checkTree(parent_key, self.root)
        if parent:
            if parent.left_child is None:
                parent.left_child = Node(node_key, parent=parent)
            elif parent.right_child is None:
                parent.right_child = Node(node_key, parent=parent)
            else:
                print('Parent has two children, not added.')
        else:
            print('Parent not in tree.')

    def checkTree(self, parentValue, root):
        """
        Recursive function that searches through tree to find if parentValue
        exists
        """
        if root is None:  #if there is no root in tree
            return False
        elif root.value == parentValue:
            if root.left_child is None or root.right_child is None:
                return root
            else:
                print("Parent has two children, node not added.")
                return False
        else:
            for child in root.getChildren():
                add_temp = self.checkTree(parentValue, child)
                if add_temp:
                    return add_temp

    def findNodeDelete(self, value, root):
        if root is None:
            return False
        elif value == root.value:
            if root.empty:
                if root.parent.left_child.value == value:
                    root.parent.left_child = None
                elif root.parent.right_child.value == value:
                    root.parent.right_child = None
                root = None
                return True
            else:
                print("Node not deleted, has children")
                return False
        else:
            for child in root.getChildren():
                delete_node = self.findNodeDelete(value, child)
                if delete_node:
                    return delete_node

    def delete(self, value: int) -> bool:
        if self.root is None:
            print('Empty Tree')
        elif value == self.root.value:
            if self.root.empty:
                #print("Deleting Root")
                self.root = None
                return True
            else:
                print("Node not deleted, has children")
                return False
        else:
            for child in self.root.children:
                delete_node = self.findNodeDelete(value, child)
                if delete_node:
                    return delete_node
        print("Parent not found.")
        return False


# Problem 4 ###################################################################
class Graph:
    def __init__(self):
        self.vertices = {}

    def __str__(self):
        printstr = ''
        for key, value in self.vertices.items():
            printstr += (str(key) +
                ('' if len(value) == 0
                    else ' -> ({})'.format(', '.join(value))) +
                '\n')
        return printstr[:-1]  # remove trailing newline

    @property
    def nodes(self):
        return sorted(list(self.vertices.keys()))

    def addVertex(self, value: int):
        #check if value already exists
        if value in self.vertices:
            print("Vertex already exists")
        else:
            self.vertices[value] = []

    def addEdge(self, node1: int, node2: int) -> None:
        if node1 not in self.nodes or node2 not in self.nodes:
            print('One or more vertices not found')
        else:
            self.vertices[node1] = sorted([self.vertices[node1] + [node2]])
            self.vertices[node2] = sorted([self.vertices[node2] + [node1]])

    def findVertex(self, value: int) -> List[int]:
        if value in self.nodes:
            print(self.vertices[value])
        else:
            print('Node does not exist')
