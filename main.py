import argparse

import taqtoe.game
from taqtoe.ascii import title
from taqtoe.model.config import default_config as config
from taqtoe.model.train import TicTacToeTrainer
from taqtoe.third_party.fileLock import FileLock
from taqtoe.utils import get_save_path

parser = argparse.ArgumentParser(
    description='TaQToe allows you to train and play against a Deep Q '
                'Learning designed for tic tac toe.')
parser.add_argument('-t', '--train',
                    help='Train a model from scratch.',
                    action='store_true')
parser.add_argument('-c', '--use_custom_weights',
                    help='Use custom trained weights instead of '
                         'pre-trained weights',
                    action='store_true')
args = parser.parse_args()


def print_file_lock_error(config):
    notice = ['File path to weights are currently locked. Are you training to '
              'the same path in another process?',
              'Config save_dir: {}'.format(config['save_dir']),
              'Config save_name: {}'.format(config['save_name'])
              ]
    print('\n'.join(notice))


if __name__ == '__main__':
    print(title)
    if args.train:
        # Try to train custom weights
        save_path = get_save_path(config)
        try:
            # Use lock to make sure we don't have multiple processes training
            # to the same path at once
            with FileLock(save_path, timeout=1):
                # Train weights from scratch
                print('Training DQN network from scratch...')
                trainer = TicTacToeTrainer(config)
                trainer.train()
        except FileLock.FileLockException as e:
            print_file_lock_error(config)
    else:
        # Play a game of tic-tac-toe
        use_pretrained = not args.use_custom_weights
        taqtoe.game.main(config, use_pretrained)
