#!/usr/bin/env python3.5

import willfarmer_hw5
import pytest


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


class TestSolution(object):
    @pytest.fixture
    def solution(self):
        return willfarmer_hw5.Solution(PseudoSystem(), numdistricts=3)

    def test_random(self, solution):
        solution.generate_random_solution()
        assert solution.is_valid is True

class PseudoSystem(object):
    def __init__(self, width=5, height=5):
        self.width = width
        self.height = height
