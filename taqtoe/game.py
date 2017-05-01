import numpy as np

import taqtoe.utils as utils
from taqtoe.constants import *
from taqtoe.player import HumanPlayer, DQNPlayer
from taqtoe.tictactoe import TicTacToe


def print_winner(game, player1, player2):
    """
    Announces the winner of the game, if there is one.
    
    :param game: `TicTacToe` object.
    :param player1: `Player` object.
    :param player2: `Player` object.
    """
    if game.winner() != 0:
        winner = player1.name if game.winner() == player1.team else player2.name
    else:
        winner = 'Nobody'
    print('Game over! {} wins!'.format(winner))


def play_again():
    """
    Prompts the console, asking if the user wants to play again. Returns 
    boolean indicating player decision.
    
    :return: Boolean. True if the player wants to keep playing.
    """
    while True:
        ans = input('Play again? (yes/no) ')
        if ans.strip().lower() in ['yes', 'y']:
            return True
        if ans.strip().lower() in ['no', 'n']:
            return False
        print('Unrecognized answer. Please answer with \'yes\' or \'no\'.')


def play_game(player1, player2):
    """
    Plays a game of TicTacToe between player1 and player2.
    
    :param player1: Player object
    :param player2: Player object
    :return: Integer. The status of the completed game. See taqtoe.constants 
        for possible status codes.
    """
    # Select and announce first player
    game = TicTacToe(first_move=np.random.choice([X, O]))
    first = 'X' if game.turn == X else 'O'
    print('\n{} is up first!'.format(first))
    # Game loop
    while game.status == CONTINUE:
        player = player1 if game.turn == player1.team else player2
        player.move(game)
        game.print_board()
    # Game over! Print out info on winner.
    print_winner(game, player1, player2)
    return game.status


def main(config, use_pretrained=True):
    """
    Main TaqQToe game method. Sets up a human player and a Deep 
    Q-Network player to play tic-tac-toe against each other for as many games as
    the user wants.
    
    :param config: Configuration dictionary. See taqtoe.model.config for 
        default options.
    :param use_pretrained: Boolean. Specifies whether to use included 
        pre-trained weights or to use custom trained weights.
    """
    if not use_pretrained and not utils.custom_weights_exist(config):
        print('Can\'t find weight files in {}.'.format(config['save_dir']))
        return
    # Setup players
    human_name = input('What\'s your name? ')
    team1 = np.random.choice([X, O])
    team2 = -team1
    player1 = HumanPlayer(team1, human_name)
    player2 = DQNPlayer(team2, config, use_pretrained, 'Hal')
    # Announce teams
    print('\n{} is playing as {}\'s, and {} is playing as {}\'s'.format(
        player1.name, player1.team_string, player2.name, player2.team_string))
    # Main loop
    keep_playing = True
    while keep_playing:
        _ = play_game(player1, player2)
        # Let user play again without restarting script.
        keep_playing = play_again()


if __name__ == '__main__':
    from taqtoe.model.config import default_config
    main(default_config)

