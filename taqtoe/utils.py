import os.path
import pkg_resources

import tensorflow as tf


def idx_to_row_col(idx):
    """
    Converts a scalar `idx` of a cell into a tuple of `(row, col)`
     -----
    |0|1|2|
    |3|4|5|
    |6|7|8|
     -----

    :param idx: Integer id of a cell on the game board [0-8]
    :raises ValueError: If `idx` is not in the range [0-8]
    :return: Integer tuple `(row, col)`
    """
    if not 0 <= idx < 9:
        raise ValueError('idx must be in range [0-8] (got {})'.format(idx))
    row = idx // 3
    col = idx % 3
    return row, col


def row_col_to_idx(row, col):
    """
    Converts a pair of row and column indices of a cell to a single `idx`
    integer.

     -----
    |0|1|2|
    |3|4|5|
    |6|7|8|
     -----

    :param row: Integer index of a row on a game board
    :param col: Integer index of a column on a game board
    :return: Integer `idx`
    """
    if any(not 0 <= i < 3 for i in (row, col)):
        raise ValueError(
            'row/col must be between [0-2] (got ({}, {})'.format(row, col))
    return (row * 3) + col


def row_col_moves_to_idx(moves):
    """
    Converts a list of moves in the form `(row, col)` to a list of single 
    integer indices.
    :param moves: List of `(row, col)` integer coordinate tuples.
    :return: List of integers.
    """
    return [
        row_col_to_idx(row, col)
        for row, col in moves
    ]


def print_with_border(*msgs):
    """
    Prints string messages with an asterisk border to the console.
   
    ``` 
    *****************************
    * This is an example        *
    * when using multiple lines *
    *****************************
    ```

    :param msgs: One or more strings to print with a border
    :return: None
    """
    lens = [len(msg) for msg in msgs]
    length = max(lens)
    border = '*' * (length + 4)
    print(border)
    for msg in msgs:
        # Add star to beginning
        res = ['* ' + msg]
        # add white space to make sure final * aligns to the end
        res.append(' ' * (length - len(msg)))
        # Add * at the end
        res.append(' *')
        print(''.join(res))
    print(border)


def get_resources_dir():

    resources_init = pkg_resources.resource_filename(
        'taqtoe.resources', '__init__.py')
    return os.path.dirname(resources_init)


def get_save_path(config):
    """
    Helper to create the save path to custom trained weights. Meant to be 
    passed into `TicTacToeDQNModel.save()` and `tf.train.latest_checkpoint`
    """
    return os.path.join(config['save_dir'], config['save_name'])


def get_pretrained_path(config):
    """
    Helper to create the path to custom trained weights. Meant to be 
    passed into `TicTacToeDQNModel.load()`
    """
    return os.path.join(get_resources_dir(), config['pretrained_name'])


def custom_weights_exist(config):
    """
    Helper to check if custom weights exist at a given path. Given a config 
    dictionary.
    """
    return tf.train.latest_checkpoint(config['save_dir']) is not None
