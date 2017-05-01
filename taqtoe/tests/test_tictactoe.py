"""
Tests for taqtoe.tictactoe
"""
from unittest import TestCase

from nose.tools import assert_raises, assert_equal

from taqtoe.constants import *
from taqtoe.exceptions import BadMoveException
from taqtoe.tictactoe import TicTacToe

E = EMPTY


class TestTicTacToeGame(TestCase):
    def test_init(self):
        # Test basic initialization
        game = TicTacToe()
        assert_equal(game.board, [[E, E, E],
                                  [E, E, E],
                                  [E, E, E]])
        assert_equal(game.turn, X)
        assert_equal(game.status, CONTINUE)

    def test_init_board(self):
        # Test pre-set board state
        board = [[X, E, E],
                 [E, O, E],
                 [X, X, O]]
        game = TicTacToe(board=board)
        assert_equal(game.board, board)

    def test_init_move(self):
        # Test setting first move
        game = TicTacToe(first_move=O)
        assert_equal(game.turn, O)

    def test_init_end_game_state(self):
        # Test setting a winning state for X
        board = [[X, O, E],
                 [X, O, E],
                 [X, E, E]]
        game = TicTacToe(board=board)
        assert_equal(game.status, X_WIN)
        # Test setting a winning state for O
        board = [[O, X, X],
                 [X, O, X],
                 [E, O, O]]
        game = TicTacToe(board=board)
        assert_equal(game.status, O_WIN)
        # Test setting a cat's game
        board = [[X, O, X],
                 [O, O, X],
                 [X, X, O]]
        game = TicTacToe(board=board)
        assert_equal(game.status, CATS_GAME)

    def test_init_bad_cell_value(self):
        # Test setting a board with an invalid cell value
        board = [[E, E, E],
                 [E, 5, E],
                 [E, E, E]]
        assert_raises(ValueError, TicTacToe, board=board)

    def test_init_bad_board_value(self):
        # Test setting invalid board sizes
        # Too tall
        board = [[E, E, E],
                 [E, E, E],
                 [E, E, E],
                 [E, E, E]]
        assert_raises(ValueError, TicTacToe, board=board)
        # Too wide
        board = [[E, E, E, E],
                 [E, E, E, E],
                 [E, E, E, E]]
        assert_raises(ValueError, TicTacToe, board=board)

    def test_move(self):
        game = TicTacToe()
        game.move(0, 2)
        expected_board = [[E, E, X], [E, E, E], [E, E, E]]
        assert_equal(game.board, expected_board)
        assert_equal(game.turn, O)

    def test_bad_move(self):
        game = TicTacToe()
        # Place X
        game.move(0, 0)
        # Check to make sure that O is the current player
        assert_equal(game.turn, O)
        # Attempt to place a piece in the same space as X
        assert_raises(BadMoveException, game.move, 0, 0)
        # Check that O is still the current player
        assert_equal(game.turn, O)
        # Try to place a piece outside of the game dimensions
        assert_raises(BadMoveException, game.move, 1, 3)
        # Check that O is still the current player
        assert_equal(game.turn, O)
        # Place a good piece
        game.move(1, 1)
        # Check that the final game state is as expected
        assert_equal(game.turn, X)
        expected_board = [[X, E, E], [E, O, E], [E, E, E]]
        assert_equal(game.board, expected_board)

    def test_move_to_end_game(self):
        # Test X win
        board = [[X, E, O],
                 [O, X, X],
                 [O, E, E]]
        game = TicTacToe(board=board)
        assert_equal(game.status, CONTINUE)
        game.move(2, 2)
        assert_equal(game.status, X_WIN)

        # Test O win
        board = [[X, X, O],
                 [O, E, X],
                 [O, E, X]]
        game = TicTacToe(board=board, first_move=O)
        assert_equal(game.status, CONTINUE)
        game.move(1, 1)
        assert_equal(game.status, O_WIN)

        # Test cat's game
        board = [[X, O, O],
                 [O, X, X],
                 [E, X, O]]
        game = TicTacToe(board=board)
        assert_equal(game.status, CONTINUE)
        game.move(2, 0)
        assert_equal(game.status, CATS_GAME)

    def test_is_available(self):
        game = TicTacToe()
        assert_equal(game.is_available(1, 1), True)
        game.move(1, 1)
        assert_equal(game.is_available(1, 1), False)

    def test_available_moves(self):
        board = [[X, O, X],
                 [E, X, O],
                 [O, X, E]]
        game = TicTacToe(board=board)
        expected_available = [(1, 0), (2, 2)]
        assert_equal(game.available_moves(), expected_available)

    def test_find_winner(self):
        for S, win_state in [(X, X_WIN), (O, O_WIN)]:
            # Check horizontal win states
            board = [[S, S, S],
                     [E, E, E],
                     [E, E, E]]
            game = TicTacToe(board=board)
            assert_equal(game._find_winner(), S)
            assert_equal(game.status, win_state)

            board = [[E, E, E],
                     [S, S, S],
                     [E, E, E]]
            game = TicTacToe(board=board)
            assert_equal(game._find_winner(), S)
            assert_equal(game.status, win_state)

            board = [[E, E, E],
                     [E, E, E],
                     [S, S, S]]
            game = TicTacToe(board=board)
            assert_equal(game._find_winner(), S)
            assert_equal(game.status, win_state)

            # Check vertical win states
            board = [[S, E, E],
                     [S, E, E],
                     [S, E, E]]
            game = TicTacToe(board=board)
            assert_equal(game._find_winner(), S)
            assert_equal(game.status, win_state)

            board = [[E, S, E],
                     [E, S, E],
                     [E, S, E]]
            game = TicTacToe(board=board)
            assert_equal(game._find_winner(), S)
            assert_equal(game.status, win_state)

            board = [[E, E, S],
                     [E, E, S],
                     [E, E, S]]
            game = TicTacToe(board=board)
            assert_equal(game._find_winner(), S)
            assert_equal(game.status, win_state)

            # Check diagonal win states
            board = [[S, E, E],
                     [E, S, E],
                     [E, E, S]]
            game = TicTacToe(board=board)
            assert_equal(game._find_winner(), S)
            assert_equal(game.status, win_state)

            board = [[E, E, S],
                     [E, S, E],
                     [S, E, E]]
            game = TicTacToe(board=board)
            assert_equal(game._find_winner(), S)
            assert_equal(game.status, win_state)

    def test_is_winning_line(self):
        game = TicTacToe()
        assert_equal(game._is_winning_line([X, X, X]), True)
        assert_equal(game._is_winning_line([O, O, O]), True)
        assert_equal(game._is_winning_line([O, X, O]), False)
        assert_equal(game._is_winning_line([E, E, E]), False)
        assert_equal(game._is_winning_line([X, E, X]), False)

    def test_full_game(self):
        game = TicTacToe()
        board = [[E, E, E],
                 [E, E, E],
                 [E, E, E]]
        assert_equal(game.board, board)
        assert_equal(game.status, CONTINUE)

        game.move(1, 1)
        board = [[E, E, E],
                 [E, X, E],
                 [E, E, E]]
        assert_equal(game.board, board)
        assert_equal(game.status, CONTINUE)

        game.move(2, 1)
        board = [[E, E, E],
                 [E, X, E],
                 [E, O, E]]
        assert_equal(game.board, board)
        assert_equal(game.status, CONTINUE)

        game.move(0, 0)
        board = [[X, E, E],
                 [E, X, E],
                 [E, O, E]]
        assert_equal(game.board, board)
        assert_equal(game.status, CONTINUE)

        game.move(2, 2)
        board = [[X, E, E],
                 [E, X, E],
                 [E, O, O]]
        assert_equal(game.board, board)
        assert_equal(game.status, CONTINUE)

        game.move(2, 0)
        board = [[X, E, E],
                 [E, X, E],
                 [X, O, O]]
        assert_equal(game.board, board)
        assert_equal(game.status, CONTINUE)

        game.move(1, 0)
        board = [[X, E, E],
                 [O, X, E],
                 [X, O, O]]
        assert_equal(game.board, board)
        assert_equal(game.status, CONTINUE)

        game.move(0, 2)
        board = [[X, E, X],
                 [O, X, E],
                 [X, O, O]]
        assert_equal(game.board, board)
        assert_equal(game.status, X_WIN)
