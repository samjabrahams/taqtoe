"""
Class file for TicTacToeDQNModel
"""
import os.path
from collections import namedtuple

import numpy as np
import tensorflow as tf

from taqtoe.model.graph import TicTacToeDQNGraph


class TicTacToeDQNModel:
    """
    Represents a stateful Dueling Double DQN network model.
     
    It contains the state for Variables (in a TensorFlow `Session`), 
    the experience buffer, and the current game history. It also contains 
    methods to run inference, training steps, and saving/loading weights to 
    disk.
    
    Main public functions:
    
    - train_batch(): Runs a single training step using a batch of data from the 
        experience buffer.
    - infer(board): Given a board state, returns the predicted values for 
        each move.
    - save_experience(board, choice, reward): Saves `board`, `choice`, 
        and `reward` to the current game history. 
    - flush_game_history(): Flush the current game history to the 
        experience buffer.Automatically removes older experiences as the 
        buffer fills.
    - update_target(): Updates the parameters in the "target" DQN to match 
        those in the "online" DQN.
    - save(path): Saves the model parameters to the specified path. If path 
        is not provided, creates a save path from config['save_dir'] and 
        config['save_name']
    - load(path): Loads the model parameters from the specified path. If path 
        is not provided, loads the most recent weights in config['save_dir'].
    """
    # Helper class for holding (state, action, reward) tuples.
    Experience = namedtuple('Experience', ['state', 'action', 'reward'])

    def __init__(self, config):
        """
        Builds a TicTacToeDQNModel.
        
        Required configuration parameters
        
        - 'batch_size': Integer. The number of examples to use in a training 
            batch.
        - 'debug': Boolean. If true, additional debug information will be 
            printed to the console.
        - 'learning_rate': Float. The learning rate to use for gradient descent.
        - 'max_buffer_size': Integer. The maximum number of (state, action, 
            reward) historical examples to store in the example buffer.
        - 'save_dir': The directory location to store trained weights.
        - 'save_name': The base filename for the weight checkpoint files.
        
        `save_path = os.path.join(save_dir, save_name)`
        `pretrained_path = os.path.join(save_dir, pretrained_name)`
        
        It will also pass configuration options on to TicTacToeDQNGraph. See 
        graph.py for required options.
        
        :param config: Dictionary mapping string keys to configuration options.
        """
        self.batch_size = config['batch_size']
        self.debug = config['debug']
        self.learning_rate = config['learning_rate']
        self.max_buffer_size = config['max_buffer_size']
        self.save_dir = os.path.abspath(config['save_dir'])
        self.save_name = config['save_name']
        self.save_path = os.path.join(self.save_dir, self.save_name)
        self.pretrained_name = config['pretrained_name']
        self.pretrained_path = os.path.join(self.save_dir, self.pretrained_name)

        self.experience_buffer = []
        self.current_game_history = []
        self.step = 0

        self.model = TicTacToeDQNGraph(config)
        self.inputs = self.model.inputs
        self.endpoints = self.model.endpoints
        self.graph = self.model.graph
        self.sess = self.build_session()

    def train_batch(self):
        """
        Runs a single step of gradient descent using randomly sampled data 
        from the experience buffer.
        
        :return: Float. The batch loss.
        """
        batch_boards, batch_actions, batch_rewards = self.get_batch()
        feed_dict = {
            self.inputs['boards']: batch_boards,
            self.inputs['actions']: batch_actions,
            self.inputs['rewards']: batch_rewards,
            self.inputs['learning_rate']: self.learning_rate
        }
        fetches = [self.endpoints['global_step'], self.endpoints['loss'],
                   self.endpoints['train']]
        self.step, loss, _ = self.sess.run(fetches, feed_dict)
        return loss

    def infer(self, board):
        """
        Runs inference on a single board state, returning the vector of 
        predicted values for each action.
        
        :param board: 2D array of integers. The current board state (as 
            provided by a `TicTacToe` object.
        :return: List of floats. The predicted value of each action.
        """
        board = np.reshape(board, [1, 3, 3])
        feed_dict = {self.inputs['boards']: board}
        return self.sess.run(self.endpoints['online_q'], feed_dict).squeeze()

    def save_experience(self, board, choice, reward):
        """
        Save the `board`, `choice`, and `reward` from a single time step to 
        the current game history.
        
        :param board: 2D array of integers. The current board state (as 
            provided by a `TicTacToe` object.
        :param choice: Integer. The selected action on the board in the form 
            of a single index. The action index corresponds to the value 
            returned from taqtoe.utils.row_col_to_idx()
        :param reward: Float. The reward from this action/state pair.
        """
        self.current_game_history.append(self.Experience(board, choice, reward))

    def flush_game_history(self):
        """
        Clears the current game history and pushes it to the experience buffer.
        If the old experiences need to be removed (due to the buffer 
        exceeding self.max_buffer_size), then those values are removed. 
        """
        # Assign rewards to current game history and append to experience buffer
        self.experience_buffer.extend(self.current_game_history)
        # Clear the current game history
        self.current_game_history = []
        # Remove old experiences, if necessary
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer = \
                self.experience_buffer[-self.max_buffer_size:]

    def get_batch_experiences(self):
        """
        Randomly samples self.batch_size number of examples from the 
        experience buffer.
        
        :return: List of Experience objects. The sampled experiences from the 
            buffer.
        """
        indices = np.random.choice(len(self.experience_buffer), self.batch_size)
        return [
            self.experience_buffer[i] for i in indices
        ]

    def get_batch(self):
        """
        Returns batches of game states, actions, and rewards for time steps. 
        Essentially separates the tupled data in `Experience` objects into 
        aligned lists.
        
        :return: Tuple (list<state>, list<actions>, list<rewards>) the batch 
            data required to run training.
        """
        batch = self.get_batch_experiences()
        batch_boards = []
        batch_actions = []
        batch_rewards = []
        for experience in batch:
            board = np.reshape(experience.state, [1, 3, 3])
            batch_boards.append(board)
            batch_actions.append(experience.action)
            batch_rewards.append(experience.reward)
        batch_boards = np.concatenate(batch_boards)
        return batch_boards, batch_actions, batch_rewards

    def update_target(self):
        """
        Updates the parameters in the "target" DQN to match those in the 
        "online" DQN.
        """
        self.sess.run(self.endpoints['update_target'])

    def save(self, path=None):
        """
        Saves current weights to the `save_path` specified in `__init__()`
        
        :param path: Optional string. If provided, save weights to location 
            specified (instead of `self.save_path`)
        """
        if path is None:
            path = self.save_path
        if self.debug:
            print('Saving weights to {}'.format(path))
        return self.endpoints['saver'].save(self.sess, path, self.step)

    def load(self, path=None, use_pretrained=False):
        """
        Restores the most recent weights from disk into the model's session.
        
        :param path: Optional string. If provided, loads weights from the 
            location specified (instead of most recent checkpoint in  
            `self.save_dir`)
        :param use_pretrained: Boolean. If True, automatically loads in 
            pretrained weights specified by config['pretrained_name']
        """
        if path is None:
            if use_pretrained:
                path = self.pretrained_path
            else:
                path = tf.train.latest_checkpoint(self.save_dir)
        if self.debug:
            print('Loading weights from {}'.format(path))
        self.endpoints['saver'].restore(self.sess, path)

    def build_session(self):
        """
        Constructs a TensorFlow `Session` with this model's graph and 
        initializes its Variables.

        :return: A TensorFlow `Session`.
        """
        sess = tf.Session(graph=self.graph)
        sess.run(self.endpoints['init'])
        return sess


if __name__ == '__main__':
    from taqtoe.model.config import default_config
    model = TicTacToeDQNModel(default_config)
    tf.summary.FileWriter('test', graph=model.graph).close()
