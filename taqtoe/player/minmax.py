import copy

import numpy as np

from taqtoe.constants import CONTINUE
from taqtoe.tictactoe import TicTacToe
from taqtoe.player.player import Player


class MinMaxPlayer(Player):
    """
    Class for a min-max algorithm player.
    """
    def __init__(self, team, max_depth=4, name='MinMaxPlayer', verbose=True):
        super(MinMaxPlayer, self).__init__(team, name, verbose)
        self.max_depth = max_depth

    def move(self, game):
        _, row, col = self._get_best_move(game, 0)
        game.move(row, col)
        if self.verbose:
            print('{} selects ({}, {})'.format(self.name, row, col))

    def _get_best_move(self, game, depth):
        """
        Min-max algorithm using TicTacToeGame
        :param game: A TicTacToeGame object.
        :param depth: The maximum number of actions to look ahead.
        :return: The score and row/col index of the best move found.
        """
        if game.state != CONTINUE or depth > self.max_depth:
            # Base case: game is over
            score = 0
            if game.winner() == self.team:
                score = 10 - depth
            elif game.winner() == -self.team:
                score = depth - 10
            return score, -1, -1

        scores = []
        moves = game.available_moves()

        for row, col in moves:
            board = copy.deepcopy(game.board)
            new_game = TicTacToe(board, game.turn)
            new_game.move(row, col)
            score, _, _ = self._get_best_move(new_game, depth+1)
            scores.append(score)

        if game.turn == self.team:
            score = max(scores)
        else:
            score = min(scores)
        # Randomly pick from the moves with the highest/lowest score
        indices = np.argwhere(np.array(scores) == score).reshape([-1])
        index = np.random.choice(indices)
        row, col = moves[index]
        return score, row, col


if __name__ == '__main__':
    # Setup a simple 2-player game
    from taqtoe.constants import X, O
    from taqtoe.player import HumanPlayer

    game = TicTacToe()
    team1 = np.random.choice([X, O])
    team2 = -team1
    player1 = HumanPlayer(team1, 'Sam')
    player2 = MinMaxPlayer(team2, 3, 'Hal')
    while game.status == CONTINUE:
        game.print_board()
        player = player1 if game.turn == team1 else player2
        print('{}\'s turn.'.format(player.name))
        player.move(game)
    print('Game over!')
    game.print_board()