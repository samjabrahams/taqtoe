import numpy as np

import taqtoe.utils as utils
from taqtoe.model.config import default_config
from taqtoe.model.model import TicTacToeDQNModel
from taqtoe.constants import CONTINUE
from taqtoe.tictactoe import TicTacToe
from taqtoe.player.player import Player


class DQNPlayer(Player):
    """
    Class for a Deep Q-Network player.
    """
    def __init__(self, team, config, use_pretrained=True, name='MinMaxPlayer',
                 verbose=True):
        """
        Creates a Deep Q-Network Player
        :param team: Integer. One of taqtoe.constants.X or taqtoe.constants.O
        :param config: Configuration dictionary. See taqtoe.model.config for 
            default options.
        :param use_pretrained: Boolean. If true, uses pre-trained weights 
            included in taqtoe.resources.
        :param name: String name for the player.
        :param verbose: Boolean. If true, print out additional text to the 
            console.
        """
        super(DQNPlayer, self).__init__(team, name, verbose)
        self.model = TicTacToeDQNModel(config)
        self.model.load(use_pretrained=use_pretrained)

    def move(self, game):
        """
        Uses a `TicTacToeDQNModel` to predict score values for each action. 
        It then selects the highest scoring valid move (ie. it automatically 
        skips over high scores for cells that already contain a piece).
        
        :param game: `TicTacToe` object. The game with the board state.
        :return: Integer representing the game status after moving.
        """
        board = self.team * np.array(game.board)
        values = self.model.infer(board)
        top_choices = sorted([(val, i) for i, val in enumerate(values)],
                             reverse=True)
        available_moves = utils.row_col_moves_to_idx(game.available_moves())
        for val, i in top_choices:
            if i in available_moves:
                row, col = utils.idx_to_row_col(i)
                break
        if self.verbose:
            print('Predicted choice values:')
            self.print_values(values)
            print('{} selects ({}, {})'.format(self.name, row, col))
        return game.move(row, col)

    def print_values(self, values):
        """
        Prints out `values` as a 3x3 grid of floating points. Used to 
        visualize what the player thinks about the current board state.
        """
        f_string = '\n'.join(['{:+.8f}\t' * 3] * 3) + '\n'
        print(f_string.format(*values))


if __name__ == '__main__':
    # Setup a simple 2-player game
    from taqtoe.constants import X, O
    from taqtoe.player import HumanPlayer
    config = default_config
    game = TicTacToe()
    team1 = np.random.choice([X, O])
    team2 = -team1
    player1 = HumanPlayer(team1, 'Sam')
    player2 = DQNPlayer(team2, config, 'Hal')
    while game.status == CONTINUE:
        player = player1 if game.turn == team1 else player2
        player.move(game)
    print('Game over!')
    game.print_board()
