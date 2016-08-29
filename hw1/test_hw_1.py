#!/usr/bin/env python3.5

#Tree Test
import pytest
import will_farmer_hw1


def test_creation():
    will_farmer_hw1.Tree(10)


def test_printing():
    tree = will_farmer_hw1.Tree(10)
    assert(str(tree) == '10')
    tree.add(1, 10)
    print(tree)
    print(tree.root.value)
    print(tree.root.left_child.value)
    assert(str(tree) == '10 -> (1)')
    tree.add(2, 10)
    assert(str(tree) == '10 -> (1, 2)')


class TestTree(object):
    """
    Tree Test
    add 10 ints to tree, print In-Order, delete 2 ints, print In-Order
    """

    @pytest.fixture(scope='module')
    def tree(self):
        return will_farmer_hw1.Tree(5)

    def test_adding(self, tree):
        tree.add(6,5)
        tree.add(4,5)
        tree.add(7,4)
        tree.add(3,7)
        tree.add(8,4)
        tree.add(2,8)
        tree.add(9,7)
        tree.add(1,3)
        tree.add(10,3)

    def test_delete(self, tree):
        oldtree = str(tree)
        tree.delete(10)  # should delete 10 node
        assert(oldtree != str(tree))
        oldtree = str(tree)
        tree.delete(1)  # should delete 1 node
        assert(oldtree != str(tree))
        tree.add(18,3)


class TestGraph(object):
    """
    Graph Test
    Add 10 vertecies, make 20 edges, print edges of five vertecies
    """
    @pytest.fixture(scope='module')
    def g(self):
        return will_farmer_hw1.Graph()

    def test_add_vertices(self, g):
        g.addVertex(1)
        g.addVertex(11)
        g.addVertex(12)
        g.addVertex(13)
        g.addVertex(14)
        g.addVertex(15)
        g.addVertex(16)
        g.addVertex(17)
        g.addVertex(18)
        g.addVertex(19)
        g.addVertex(100)

    def test_add_edges(self, g):
        g.addEdge(1,12)
        g.addEdge(1,13)
        g.addEdge(11,14)
        g.addEdge(15,11)
        g.addEdge(16,100)
        g.addEdge(15,17)
        g.addEdge(15,12)
        g.addEdge(12,13)
        g.addEdge(12,14)
        g.addEdge(12,16)
        g.addEdge(12,17)
        g.addEdge(1,100)
        g.addEdge(12,100)
        g.addEdge(15,100)
        g.addEdge(19,12)
        g.addEdge(13,100)
        g.addEdge(14,100)
        g.addEdge(100,19)
        g.addEdge(19,18)
        g.addEdge(19,17)
        g.addEdge(52, 53)

    def test_find_vertices(self, g):
        g.findVertex(1)
        g.findVertex(12)
        g.findVertex(13)
        g.findVertex(14)
        g.findVertex(100)
        g.findVertex(52)
