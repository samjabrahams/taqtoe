"""
Tests for taqtoe.utils
"""
from unittest import TestCase
from nose.tools import assert_equal, assert_raises

from taqtoe import utils


class TestIdxRowCol(TestCase):
    def setUp(self):
        # The index in `cases` corresponds to the correct index of each
        # coordinate
        self.cases = [
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2),
        ]

    def test_idx_to_row_col(self):
        for i in range(9):
            assert_equal(utils.idx_to_row_col(i), self.cases[i])

    def test_row_col_to_idx(self):
        for i, case in enumerate(self.cases):
            row, col = case
            assert_equal(utils.row_col_to_idx(row, col), i)

    def test_row_col_moves_to_idx(self):
        expected = list(range(9))
        assert_equal(utils.row_col_moves_to_idx(self.cases), expected)

    def test_bad_idx(self):
        assert_raises(ValueError, utils.idx_to_row_col, -1)
        assert_raises(ValueError, utils.idx_to_row_col, 9)

    def test_bad_row_col(self):
        assert_raises(ValueError, utils.row_col_to_idx, 0, -1)
        assert_raises(ValueError, utils.row_col_to_idx, 0, 3)
        assert_raises(ValueError, utils.row_col_to_idx, -1, 0)
        assert_raises(ValueError, utils.row_col_to_idx, 3, 0)