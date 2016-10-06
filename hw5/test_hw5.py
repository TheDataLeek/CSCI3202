#!/usr/bin/env python3.5

import willfarmer_hw5
import pytest
import numpy as np


class TestEndToEnd(object):
    @pytest.fixture
    def system(self):
        return willfarmer_hw5.System('./hw5/smallState.txt')

    def test_simulated_annealing(self, system):
        willfarmer_hw5.simulated_annealing(system, 8, 10, False, False)
        willfarmer_hw5.simulated_annealing(system, 4, 10, False, False)

    def test_genetic(self, system):
        willfarmer_hw5.genetic_algorithm(system, 8, 10, False, False)
        willfarmer_hw5.genetic_algorithm(system, 4, 10, False, False)


class TestSystem(object):
    @pytest.fixture
    def system(self):
        return willfarmer_hw5.System('./hw5/smallState.txt')

    def test_file_parser(self, system):
        filesystem = np.array([
                        [0,1,0,1,0,1,1,1],
                        [0,0,1,0,1,1,1,1],
                        [0,0,0,1,1,1,1,1],
                        [0,0,1,1,1,1,0,1],
                        [1,1,0,0,0,1,1,1],
                        [1,0,0,0,0,0,1,1],
                        [1,1,1,0,0,0,0,0],
                        [0,0,0,0,0,0,1,0]])
        assert (system.matrix == filesystem).all()

    def test_width(self, system):
        assert system.width == 8

    def test_height(self, system):
        assert system.height == 8

    def test_stats(self, system):
        size, percents = system.stats
        assert size == 64
        assert percents['D'] == 0.5
        assert percents['R'] == 0.5


class TestMask(object):
    @pytest.fixture
    def mask(self):
        return willfarmer_hw5.Mask()

    def test_parse(self, mask):
        mask.parse_list([[1, 0, 0],
                         [1, 0, 0],
                         [0, 0, 0]])
        mask.parse_list([[1, 0, 0],
                         [0, 0, 0]])

    def test_valid(self, mask):
        mask.parse_list([[1, 0, 0],
                         [1, 0, 0],
                         [0, 0, 0]])
        assert mask.is_valid is True
        mask.parse_list([[1, 0, 0],
                         [0, 0, 0]])
        assert mask.is_valid is True
        mask.parse_list([[1, 0, 0],
                         [1, 0, 0],
                         [0, 0, 1]])
        assert mask.is_valid is False
        mask.parse_list([[1, 0, 0],
                         [1, 0, 1],
                         [0, 0, 1]])
        assert mask.is_valid is False
        mask.parse_list([[1, 0, 0],
                         [1, 1, 1],
                         [0, 0, 1]])
        assert mask.is_valid is True
        mask.parse_list([[1, 1, 0],
                         [0, 1, 1]])
        assert mask.is_valid is True

    def test_overlap(self, mask):
        mask.parse_list([[1, 0, 0],
                         [1, 1, 1],
                         [0, 0, 1]])
        mask0 = willfarmer_hw5.Mask()
        mask0.parse_list([[0, 0, 0],
                          [0, 1, 1],
                          [0, 0, 0]])
        assert mask.overlap(mask0) is True
        mask0.parse_list([[0, 0, 0],
                          [0, 0, 0],
                          [1, 1, 0]])
        assert mask.overlap(mask0) is False

    def test_parse_locations(self, mask):
        mask.parse_locations(3, 3, [[0, 0], [1, 1], [1, 0]])
        mask0 = willfarmer_hw5.Mask()
        mask0.parse_list([[1, 0, 0],
                          [1, 1, 0],
                          [0, 0, 0]])
        assert mask.overlap(mask0) is True

    def test_make_valid(self, mask):
        mask.parse_list([[1, 0, 0],
                         [1, 0, 1],
                         [0, 0, 1]])
        mask.make_valid()
        assert mask.is_valid
        assert mask.mask[0, 0] == 1 or mask.mask[2, 2] == 1


class TestSolution(object):
    @pytest.fixture
    def solution(self):
        return willfarmer_hw5.Solution(PseudoSystem(), numdistricts=8)

    def test_random(self, solution):
        solution.generate_random_solution()
        assert solution.is_valid is True

    def test_copy(self, solution):
        copy = solution.copy()
        assert (copy.full_mask == solution.full_mask).all()
        copy.full_mask[0, 0] = 999
        assert copy.full_mask[0, 0] != solution.full_mask[0, 0]

    def test_valid(self, solution):
        solution.full_mask[:] = 1
        assert solution.is_valid
        solution.full_mask[1, 1] = 0
        assert not solution.is_valid

    def test_value(self, solution):
        solution.full_mask[:] = 1
        solution.full_mask[1, 1] = 0
        assert solution.value == 0

    def test_openspots(self, solution):
        for _ in range(100):
            y, x = solution.get_random_openspot(0)
            assert 0 <= y < len(solution.full_mask)
            assert 0 <= x < len(solution.full_mask[0])
        assert not any(solution.get_random_openspot(5))
        solution.full_mask[0, 0] = 1
        for _ in range(10):
            y, x = solution.get_random_openspot(1)
            assert y == 0
            assert x == 0

    def test_mutate(self, solution):
        solution.generate_random_solution()
        copy = solution.copy()
        solution.mutate()
        assert len(np.where((copy.full_mask - solution.full_mask) != 0)[0]) == 1

    def test_combine(self, solution):
        solution.generate_random_solution()
        osol = willfarmer_hw5.Solution(PseudoSystem(), numdistricts=8)
        osol.generate_random_solution()
        new_sol = solution.combine(osol)

    def test_district_neighbors(self):
        solution = willfarmer_hw5.Solution(PseudoSystem(width=3, height=3), 2)
        solution.full_mask = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        assert len(solution.get_district_neighbors(1)) == 3

        solution = willfarmer_hw5.Solution(PseudoSystem(width=3, height=3), 2)
        solution.full_mask = np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0]])
        assert len(solution.get_district_neighbors(1)) == 4

        solution = willfarmer_hw5.Solution(PseudoSystem(width=3, height=3), 2)
        solution.full_mask = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        assert len(solution.get_district_neighbors(1)) == 4

        solution = willfarmer_hw5.Solution(PseudoSystem(width=3, height=3), 2)
        solution.full_mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 0]])
        assert len(solution.get_district_neighbors(1)) == 1

    def test_get_filtered_neighbors(self):
        solution = willfarmer_hw5.Solution(PseudoSystem(width=3, height=3), 2)
        solution.full_mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 0]])
        assert len(solution.get_filtered_district_neighbors(1, [0])) == 1

        solution = willfarmer_hw5.Solution(PseudoSystem(width=3, height=3), 2)
        solution.full_mask = np.array([[1, 1, 1], [1, 1, 1], [2, 1, 0]])
        assert len(solution.get_filtered_district_neighbors(1, [0])) == 1

        solution = willfarmer_hw5.Solution(PseudoSystem(width=3, height=3), 2)
        solution.full_mask = np.array([[2, 2, 2], [2, 1, 1], [1, 1, 0]])
        assert len(solution.get_filtered_district_neighbors(1, [0])) == 1


class PseudoSystem(object):
    def __init__(self, width=8, height=8):
        self.width = width
        self.height = height

