"""
Class file for TicTacToeTrainer
"""
import copy

import numpy as np

import taqtoe.utils as utils
from taqtoe.constants import *
from taqtoe.tictactoe import TicTacToe
from taqtoe.model.model import TicTacToeDQNModel


class TicTacToeTrainer:
    """
    Trains a TicTacToeDQNModel against itself.
    
    Creates a "train" TicTacToeDQNModel as well as an "opponent" 
    TicTacToeDQNModel. The two play games against one another (with the first 
    player being randomly assigned each game). After each game, the train 
    model performs a step of gradient descent using a mini-batch of historical
    experiences. The train model weights are saved to disk at a specified 
    interval, at which point the opponent model updates its own parameters to 
    match the train model. This process continues until the specified number 
    of training games/gradient descent updates have been completed.
    
    The main primary public method is `train()`.
    """
    def __init__(self, config):
        """
        Builds a TicTacToeTrainer.
        
        Required configuration parameters:
        
        - 'debug': Boolean. If True, additional debug info is printed to the
            console.
        - 'num_warmup_games': Integer. The number of games that will be 
            played at random to initialize historical examples.
        - 'max_games': Integer. The maximum number of games to be played during 
            training. Does not include warmup games.
        - 'train_epsilon': Float [0-1]. The probability that the train model 
            will make a random choice (instead of the maximum estimated value) 
            for its actions during training. Will be decayed according to 
            'epsilon_decay_steps' and 'min_epsilon' parameters.
        - 'epsilon_decay_steps': Integer. The number of training steps before 
            reaching complete epsilon decay.
        - 'min_epsilon': Float [0-1]. The value of epsilon after completely 
            decaying.
        - 'opponent_epsilon': Float [0-1]. The probability that the opponent 
            model will make a random choice (instead of the maximum estimated 
            value) for its actions during training. Does not decay.
        - 'save_steps': Integer. The interval of steps before the train model's
            weights are saved to disk.
        - 'update_target_steps': Integer. The interval of steps before 
            updating the train model's "target" DQN parameters to match 
            those in its "online" DQN.
        
        This will also pass configuration options to TicTacToeDQNModel (and 
        thus TicTacToeDQNGraph). See model.py and graph.py for information on 
        their required options.
        
        :param config: Dictionary mapping string keys to configuration options.
        """
        self.config = copy.deepcopy(config)
        self.debug = config['debug']
        self.num_warmup_games = config['num_warmup_games']
        self.max_games = config['max_games']
        self.train_epsilon = config['train_epsilon']
        self.epsilon_decay_steps = config['epsilon_decay_steps']
        self.min_epsilon = config['min_epsilon']
        self.opponent_epsilon = config['opponent_epsilon']
        self.save_steps = config['save_steps']
        self.update_target_steps = config['update_target_steps']
        self.summary_steps = config['summary_steps']
        self.train_model, self.opponent_model = self.build_models(config)

    def build_models(self, config):
        """
        Builds separate TicTacToeDQNModel objects for the primary training 
        model and the opponent model.
        
        :param config: Dictionary mapping string keys to configuration options.
        :return: Tuple `(TicTacToeDQNModel, TicTacToeDQNModel)`. The train and 
            opponent models.
        """
        config['is_training'] = True
        train_model = TicTacToeDQNModel(config)
        opponent_config = copy.deepcopy(config)
        opponent_config['is_training'] = False
        opponent_model = TicTacToeDQNModel(opponent_config)
        return train_model, opponent_model

    def train(self):
        """
        Runs training on the model.
        
        Specific steps of the training process:
        
        1. Runs config['num_warmup_games'] number of "warmup" games, 
            where both players act in a completely random fashion. This 
            populates the experience history such that the first several 
            batches of training don't have to be smaller than future batches 
            due to there not being enough data.
        2. The initial weights are saved to disk.
        3. Run config['max_games'] number of games. For each game:
            * Set the epsilon value for the train model based on config[
                'train_epsilon'], config['epsilon_range'] and config[
                'epsilon_decay_steps']
            * Play a game of tic-tac-toe to completion (see run_game())
            * Perform gradient descent on the train model to update its weights.
            * Every config['save_step'] number of games, save the train 
                model's weights to disk and update the opponent model to use the 
                current train model weights.
            * Every config['update_target_steps'] number of games, update the 
                train model's "target" DQN to use the parameters in its 
                "online" DQN.
            * Every config['summary_steps'] number of games, get summary 
                statistics from training.
        """
        # Save step 0 to disk and load initial weights to opponent
        self.warmup(self.num_warmup_games)
        self.train_model.save()
        self.opponent_model.load()
        for i in range(self.max_games):
            train_epsilon = self.current_train_epsilon(i)
            self.run_game(train_epsilon, self.opponent_epsilon)
            loss = self.train_model.train_batch()
            if self.train_model.step % self.save_steps == 0:
                self.train_model.save()
                self.opponent_model.load()

            if self.train_model.step % self.update_target_steps == 0:
                self.train_model.update_target()

            if self.debug and i % self.summary_steps == 0:
                print('Step: {}\tloss: {}'.format(i, loss))

    def run_game(self, train_epsilon, opponent_epsilon):
        """
        Plays a game to completion, saving experience from both players to 
        the train model's experience buffer. Randomly selects which player 
        goes first.
        
        :param train_epsilon: Float [0-1]. The probability that the train model 
        makes a random move.
        :param opponent_epsilon: Float [0-1]. The probability that the opponent 
        model makes a random move.
        """
        # Always set the train model to be team X, but randomly select which
        # model goes first
        train_team = X
        first = np.random.choice([X, O])
        game = TicTacToe(first_move=first)
        # Maintain the previous board state and choices for both models
        last_board = {
            X: None,
            O: None
        }
        last_choice = {
            X: None,
            O: None
        }
        while game.status == CONTINUE:
            # Select train/opponent values based on who's turn it is
            model = self.train_model if game.turn == train_team \
                else self.opponent_model
            epsilon = train_epsilon if game.turn == train_team \
                else opponent_epsilon
            # Add last move to experience buffer, if there is one
            if last_board[game.turn] is not None:
                self.train_model.save_experience(
                    last_board[game.turn], last_choice[game.turn], 0)
            # Pick a move!
            row, col, choice = self.infer_from_model(
                model, game, game.turn, epsilon)
            # Save the board state and selected move
            # Multiply the board state by game.turn (either 1 for X or -1 for O)
            # to normalize experience by team. This allows us to use opponent's
            # experience as if it were the train model playing.
            last_board[game.turn] = np.copy(game.board) * game.turn
            last_choice[game.turn] = choice
            game.move(row, col)
        # Game is over. Flush the game history.
        reward = game.winner()
        # Get rewards for both last X and O moves
        self.train_model.save_experience(last_board[X], last_choice[X], reward)
        self.train_model.save_experience(last_board[O], last_choice[O], -reward)
        self.train_model.flush_game_history()

    def warmup(self, num_games):
        """
        Runs a number of games played by random players. Used to populate the 
        experience history buffer before beginning training.
        
        :param num_games: Number of games to warm up with.
        """
        for i in range(num_games):
            self.run_game(1.0, 1.0)

    def infer_from_model(self, model, game, team, epsilon):
        """
        Use `model` to select a move, given `game` state, with `epsilon` 
        chance of a random move being selected. If the model selects an 
        unavailable move, a random move will be chosen instead.
        
        :param model: `TicTacToeDQNModel`. The model which should select a move.
        :param game: `TicTacToe` object. The game containing the current 
            board state.
        :param team: Integer (-1 for O, 1 for X). Indicates which team this 
            model is playing for.
        :param epsilon: Float [0-1]. The probability that a random move will 
            be selected instead of the model's preferred option.
        :return: Tuple of integers. `(row, col, choice)` The row and column 
            coordinates followed by the index returned from 
            `utils.row_col_to_idx(row, col)`
        """
        board = team * np.array(game.board)
        choice = np.argmax(model.infer(board))
        moves = game.available_moves()
        do_random = np.random.rand() < epsilon
        if choice in utils.row_col_moves_to_idx(moves) and not do_random:
            row, col = utils.idx_to_row_col(choice)
        else:
            row, col = moves[np.random.choice(len(moves))]
            choice = utils.row_col_to_idx(row, col)
        return row, col, choice

    def current_train_epsilon(self, steps):
        """
        Helper to calculate epsilon with after `steps` amount of decay.
        
        :param steps: Integer. The number of steps of decay.
        :return: The current epsilon for the train model (no lower than 
            `self.min_epsilon`)
        """
        epsilon_range = self.train_epsilon - self.min_epsilon
        decay_per_step = epsilon_range / self.epsilon_decay_steps
        current_epsilon = self.train_epsilon - decay_per_step * steps
        return max(self.min_epsilon, current_epsilon)


if __name__ == '__main__':
    from taqtoe.model.config import default_config
    trainer = TicTacToeTrainer(default_config)
    trainer.train()
