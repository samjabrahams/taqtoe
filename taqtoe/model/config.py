"""
Default configuration for taqtoe.model classes
"""
from taqtoe.utils import get_resources_dir


default_config = {
    # The number of game states to process per train step
    'batch_size': 128,
    # Sets the debug state to True. Additional info is printed to console
    'debug': True,
    # The starting rate at which to sample random moves for the training
    # agent
    'train_epsilon': 0.1,
    # The rate at which to sample random moves for the opposing agent
    'opponent_epsilon': 0.3,
    # The number of steps before training epsilon has fully decayed
    'epsilon_decay_steps': 15000,
    # The minimum value of the training epsilon
    'min_epsilon': 0.05,
    # The discount rate for future versus immediate reward
    'gamma': 0.99,
    # When true, constructs loss/training portions of the graph
    'is_training': True,
    # Learning rate for gradient descent
    'learning_rate': 1e-5,
    # The maximum number of recent experiences to sample from when training
    'max_buffer_size': 10000,
    # The number of games to play during training
    'max_games': 100000,
    # The number of games to play randomly before starting training
    'num_warmup_games': 200,
    # The directory to save training weights
    'save_dir': get_resources_dir(),
    # The name of the weight checkpoints
    'save_name': 'dqn',
    # The base name for pretrained weights
    'pretrained_name': 'pretrained',
    # The interval at which to save model weights
    'save_steps': 10000,
    # The interval at which to get summary statistics during training
    'summary_steps': 500,
    # 'tau': the number of steps to wait before updating the "target" DQN
    'update_target_steps': 5000,
}
