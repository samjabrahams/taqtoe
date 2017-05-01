import taqtoe.utils as utils
from taqtoe.exceptions import BadMoveException
from taqtoe.player.player import Player


class HumanPlayer(Player):
    """
    Class for a human tic-tac-toe player.
    """
    def move(self, game):
        """
        Asks for user input from the console to make a move. Will loop until 
        a valid move is provided.
        
        :param game: The TicTacToeGame that the player should move for.
        :return: Integer. The game status after moving the piece.
        """
        available_moves = utils.row_col_moves_to_idx(game.available_moves())
        while True:
            move = self._prompt_player(game, available_moves)
            try:
                int_move = self._validate_user_input(move)
                row, col = utils.idx_to_row_col(int_move)
                return game.move(row, col)
            except BadMoveException as e:
                utils.print_with_border(str(e))

    def _prompt_player(self, game, available_moves):
        """
        Displays current game status and requests a move from the player.
        :param game:
        :param available_moves: List of integers representing available cells.
        :return: String input from the user (stripped of whitespace).
        """
        print('\n{}\'s turn: please select a move (you are {}).'
              .format(self.name, self.team_string))
        game.print_board(print_available_cell_indices=True)
        print('Available moves: {}'.format(str(available_moves)))
        return input('> ').strip()

    def _validate_user_input(self, move):
        """
        Validates
        :param s: 
        :return: 
        """
        if not move.isdigit():
            raise BadMoveException('You must pass in a number as input.')
        move = int(move)
        if not 0 <= move < 9:
            raise BadMoveException('Move must be in range [0-8].')
        return move

if __name__ == '__main__':
    # Setup a simple 2-player game
    from taqtoe.constants import X, O, CONTINUE
    from taqtoe.tictactoe import TicTacToe
    game = TicTacToe()
    player1 = HumanPlayer(X, 'Sam')
    player2 = HumanPlayer(O, 'Zak')
    game.print_board()
    while game.status == CONTINUE:
        player = player1 if game.turn == X else player2
        player.move(game)
        game.print_board()
    print('Game over!')
    game.print_board()
