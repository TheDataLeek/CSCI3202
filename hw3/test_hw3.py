from willfarmer_hw3 import *
import networkx

class TestHeap:
    def testpush(self):
        heap = BMinHeap()
        heap.push(('a', 10))
        assert([('a', 10)] == heap.queue)
        heap.push(('a', 5))
        assert([('a', 5), ('a', 10)] == heap.queue)
        heap.push(('a', 15))
        assert([('a', 5), ('a', 10), ('a', 15)] == heap.queue)
        heap.push(('a', 20))
        assert([('a', 5), ('a', 10), ('a', 15), ('a', 20)] == heap.queue)

    def testfind_min(self):
        heap = BMinHeap()
        [heap.push(('a', i)) for i in range(100)]
        for i in range(100):
            assert(heap.find_min() == ('a', i))
        assert(heap.queue == [])
        [heap.push(('a', i)) for i in range(100, 0, -1)]
        for i in range(1, 101):
            assert(heap.find_min() == ('a', i))
        assert(heap.queue == [])

    def testheap(self):
        for i in range(100):
            current = [('a', random.randint(0, 100)) for x in range(20)]
            heap = BMinHeap()
            [heap.push(x) for x in current]
            assert([heap.find_min() for x in range(20)] == sorted(current, key=lambda t:t[1]))

    def testadjust(self):
        heap = BMinHeap()
        heap.push(('a', 10))
        heap.push(('b', 5))
        heap.push(('c', 15))
        heap.push(('d', 20))
        assert(heap.queue == [('b', 5), ('a', 10), ('c', 15), ('d', 20)])
        heap.adjust('c', 1)
        print(heap.queue)
        assert(heap.queue == [('c', 1), ('b', 5), ('d', 20), ('a', 10)])


class TestGraph:
    def testinit(self):
        g = Graph()
        assert(g.size == 0)

    def testaddnode(self):
        g = Graph()
        g.add_node('A')
        g.add_node('B')
        g.add_node('C')
        g.add_edge('A', 'C')
        g.add_edge('B', 'C')
        assert(g.size == 3)
        g.add_node('Z')
        assert(g.size == 4)

    def testfindadjacent(self):
        g = Graph()
        g.add_node('A')
        g.add_node('B')
        g.add_node('C')
        g.add_edge('A', 'C')
        g.add_edge('B', 'C')
        assert(len(g.find_adjacent('A')) == 1)
        try:
            g.find_adjacent('Z')
            raise AssertionError('Error should have been thrown')
        except NodeException:
            pass

    def testgeneration(self):
        g = Graph()
        g.generate_erdosrenyi(10)

    def testtoedgelist(self):
        g = Graph()
        g.add_node('A')
        g.add_node('B')
        g.add_node('C')
        g.add_edge('A', 'C')
        g.add_edge('B', 'C')
        assert(g.to_edge_list() == [('A', 'C'), ('B', 'C')])

    def testshortestpath(self):
        g = Graph()
        g.add_node('A')
        g.add_node('B')
        g.add_node('C')
        g.add_node('D')
        g.add_node('E')
        g.add_node('F')
        g.add_edge('A', 'D')
        g.add_edge('A', 'F')
        g.add_edge('B', 'E')
        g.add_edge('D', 'C')
        g.add_edge('D', 'E')
        g.add_edge('E', 'A')
        g.add_edge('F', 'B')
        g['A'].estimate = 0
        g['B'].estimate = 2
        g['C'].estimate = 2
        g['D'].estimate = 3
        g['E'].estimate = 4
        g['F'].estimate = 10
        gx = networkx.DiGraph()
        gx.add_edges_from(g.to_edge_list())
        start = "B"
        end = "C"
        assert(len(g.shortest_path_dijkstra(start, end)[0]) ==
                len(g.shortest_path_Astar(start, end)[0]) ==
                len(networkx.shortest_path(gx, start, end)))
        for i in range(2, 100):
            g1 = Graph()
            g1.generate_erdosrenyi(i)
            g2 = g1.to_networkx()

            start = chr(96 + random.randint(0, i - 1))
            end = chr(96 + random.randint(0, i - 1))
            while start == end:
                end = chr(96 + random.randint(0, i - 1))

            print(start, '->', end)
            mypath = g1.shortest_path_dijkstra(start, end)
            print(mypath)
            try:
                npath  = networkx.shortest_path(g2, start, end)
            except networkx.exception.NetworkXNoPath:
                npath = []
            print(npath)
            try:
                print(1)
                assert(len(mypath[0]) == len(npath))
            except AssertionError:
                networkx.draw_circular(g2)
                matplotlib.pyplot.savefig('./ErrorGraph.png')
                raise
