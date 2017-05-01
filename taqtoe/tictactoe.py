"""
Class file for TicTacToe
"""
import itertools

import numpy as np

from taqtoe.constants import *
from taqtoe.exceptions import BadMoveException


class TicTacToe:
    """
    Represents a game of Tic-Tac-Toe.
    
    - move(row, col)
    - is_available(row, col)
    - available_moves()
    - winner()
    - print_board()
    """
    def __init__(self, board=None, first_move=X, debug=False):
        """
        Creates a Tic-Tac-Toe game

        :param board: An initial state for the game board. If `None`,
            the game is initialized with an empty (starting) board.
        :param first_move: Integer id of the player that should move first.
        """
        # Set the initial state
        if board is None:
            E = EMPTY
            board = [[E, E, E],
                     [E, E, E],
                     [E, E, E]]
        self._validate_board(board)
        self.board = board
        # Set the initial first turn
        self._validate_first_move(first_move)
        self.turn = first_move
        # Set the initial state
        self.status = self._determine_state()
        self.debug = debug

    def move(self, row, col):
        """
        Attempts to place a move in the cell located at (`row`, `col`).
        Automatically alternates turns between the two players.

        :param row: Integer row coordinate to place the move.
        :param col: Integer column coordinate to place the move.
        :raises BadMoveException: If the move is invalid.
        :return: The board state. (one of CONTINUE, X_WIN, O_WIN, or CATS_GAME)
        """
        if self.debug:
            self.print_board()
            print('MOVING {}, {}'.format(row, col))
        if self.status != CONTINUE:
            return self.status
        self._validate_move(row, col)
        self.board[row][col] = self.turn
        self.turn = X if self.turn == O else O
        self.status = self._determine_state()
        return self.status

    def is_available(self, row, col):
        """
        :return: True if the cell at `(row, col)` is empty. False otherwise.
        """
        return self.board[row][col] == EMPTY

    def available_moves(self):
        """
        Creates a list of all available moves on the game board. These are
        the spaces where

        :return: List of (row, col) tuples corresponding to the coordinates of
            available cells.
        """
        return [
            (i, j)
            for i, row in enumerate(self.board)
            for j, _ in enumerate(row)
            if self.is_available(i, j)
        ]

    def winner(self):
        """
        Returns the integer constant associated with the player (X or O) who 
        won this game. If there is no winner, it returns the integer constant 
        associated with an empty space.
        
        :return: Integer. The constant associated with the winning team. See 
            taqtoe.constants for available team constants.
        """
        if self.status == X_WIN:
            return X
        elif self.status == O_WIN:
            return O
        else:
            return EMPTY

    def board_format_string(self):
        """
        Creates a string that can be used to visually represent a 3x3 
        tic-tac-toe board. There are nine string format locations that can be
        filled with numbers, empty space, Xs, Os, etc.
        
        :return: String. Designed to be formatted with nine inputs.
        """
        board_strings = [' _________________ ']
        for i in range(3):
            board_strings.extend([
                '|     |     |     |',
                '|  {}  |  {}  |  {}  |',
                '|_____|_____|_____|'
            ])
        return '\n'.join(board_strings)

    def print_board(self, print_available_cell_indices=False):
        """
        Prints a string representation of the game board to the console. The 
        center of each cell will have an X, O, or be empty. Empty cells can 
        be set to show either a blank space or a numeric index with 
        `print_available_cell_indices`.
        
        :param print_available_cell_indices: Boolean. If True, available 
            cells will display a numeric index instead of a blank space.
        """
        cell_strings = []
        for i, cell in enumerate(np.ravel(self.board)):
            if cell == X:
                cell_strings.append('X')
            elif cell == O:
                cell_strings.append('O')
            else:
                empty = str(i) if print_available_cell_indices else ' '
                cell_strings.append(empty)
        print(self.board_format_string().format(*cell_strings))

    def _determine_state(self):
        """
        Checks the board state to determine if there is a winner, a tied game,
        or if the game should continue.
        :return: Integer. One of CONTINUE, X_WIN, O_WIN, or CATS_GAME
        """
        winner = self._find_winner()
        if winner != EMPTY:
            return X_WIN if winner == X else O_WIN
        if len(self.available_moves()) == 0:
            return CATS_GAME
        return CONTINUE

    def _find_winner(self):
        """
        Checks rows, columns, and diagonals to see if either X or O have won. If
        there is a winner, it returns the integer constant corresponding to
        them. Otherwise, it returns the integer constant corresponding to an
        empty space.

        Note: This implementation does not check if the board has more than
        one winning combination. Thus, if a board is set to have both an X
        win an O win, it will return the first winning combination it sees.
        The order it checks is the first row, then the first column, then the
        second row and so on. After checking the rows and columns, it checks
        both diagonals (top-left to bottom-right first).

        :return: Integer corresponding to the winning team, if any. One of X, O,
            or EMPTY (if there is no winner)
        """
        for i in range(3):
            # check rows
            if self._is_winning_line(self.board[i]):
                return self.board[i][0]
            # Check cols
            if self._is_winning_line([self.board[j][i] for j in range(3)]):
                return self.board[0][i]
        # check diagonals
        if self._is_winning_line([self.board[i][i] for i in range(3)]):
            return self.board[0][0]
        if self._is_winning_line([self.board[2-i][i] for i in range(3)]):
            return self.board[2][0]
        # default: no winning combination. return EMPTY
        return EMPTY

    def _is_winning_line(self, cells):
        """
        Checks to see if all values in a set of cells are of the same team
        (X or O). If so, this method returns True.
        :param cells: List of integers. The cell values to check.
        :return: Boolean. True if all cell values represent the same team (X
        or O). False otherwise.
        """
        return all(c == X for c in cells) or all(c == O for c in cells)

    def _validate_board(self, board):
        """
        Checks that `board` is of proper length and only contains valid cell
        values.

        :param board: List of integers. The board state.
        :raises ValueError: If state is invalid.
        """
        if any(cell not in (EMPTY, X, O)
               for cell in itertools.chain.from_iterable(board)):
            raise ValueError(
                'Initial cells must only contain integers {}, {}, or {}.'
                .format(EMPTY, X, O))
        if len(board) != 3 or len(board[0]) != 3:
            raise ValueError('cells must be a 3x3 matrix')

    def _validate_first_move(self, first_move):
        """
        Checks to make sure the that `first_move` parameter sent to the
        __init__ method is either X or O

        :param first_move:
        """
        if first_move not in (X, O):
            raise ValueError(
                'first_turn must be set to X ({}) or O ({}). Received {}'
                .format(X, O, first_move))

    def _validate_move(self, row, col):
        """
        Checks to make sure that the move indicated by (`row`, `col`) is both a 
        valid set of coordinates and is available.
        
        :param row: The index of the row coordinate for the desired move.
        :param col: The index of the column coordinate for the desired move.
        :raises BadMoveException: If move is invalid. 
        """
        if not 0 <= row < 3 or not 0 <= col < 3:
            raise BadMoveException(
                'Cell ({}, {}) is not valid (row and col must be between 0-2).'
                .format(row, col))
        if not self.is_available(row, col):
            raise BadMoveException(
                'Cell ({}, {}) is already taken.'
                .format(row, col))


if __name__ == '__main__':
    # Simulate a game getting played.
    game = TicTacToe()
    print_indices = False
    game.print_board(print_indices)

    game.move(1, 1)
    game.print_board(print_indices)

    game.move(2, 1)
    game.print_board(print_indices)

    game.move(0, 0)
    game.print_board(print_indices)

    game.move(2, 2)
    game.print_board(print_indices)

    game.move(2, 0)
    game.print_board(print_indices)

    game.move(1, 0)
    game.print_board(print_indices)

    game.move(0, 2)
    game.print_board(print_indices)
