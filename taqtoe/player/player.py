from abc import ABCMeta, abstractmethod

from taqtoe.constants import X, O


class Player:
    """
    Abstract class defining the interface of a tic-tac-toe player.
    """
    __metaclass__ = ABCMeta

    def __init__(self, team, name='Player', verbose=False):
        """
        Creates a player.
        :param team: Integer. Indicates if the player is playing Xs or Os 
        :param name: String. Human-readable name for the player. 
            Defaults to 'Player'.
        :param verbose: Boolean. If true, prints additional move information 
            to the console.
        """
        self.is_first = team
        if team not in (X, O):
            raise ValueError(
                'Player must be created with team of X ({}) or O ({})'
                .format(X, O))
        self.team = team
        self.team_string = 'X' if team == X else 'O'
        self.name = name
        self.verbose = verbose

    @abstractmethod
    def move(self, game):
        """
        Abstract method for selecting a move. Given a TicTacToe object, should 
        run the `TicTacToe.move()` and return the game status. 
        
        For helpers to covert between `(row, col)` tuples and a single 
        integer index, see `row_col_to_idx()` and `idx_to_row_col` in 
        `taqtoe.utils`.
        
        :param game:
        :return:
        """
        raise NotImplementedError(
            'Abstract method move() must be implemented by concrete subclass.')
