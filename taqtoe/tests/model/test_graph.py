"""
Tests for taqtoe.model.graph
"""
# Prevents TensorFlow logs from polluting Nose tests
# (without setting an environment variable)
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)

from unittest import TestCase

import tensorflow as tf
from nose.tools import assert_raises, assert_equal, assert_not_equal

from taqtoe.model.config import default_config
from taqtoe.model.graph import TicTacToeDQNGraph


class TestTicTacToeDQNGraphTrain(TestCase):
    def setUp(self):
        self.dqn_graph = TicTacToeDQNGraph(default_config)

    def test_target_online_variables(self):
        online_vars = self.dqn_graph.get_online_variables()
        target_vars = self.dqn_graph.get_target_variables()
        # Make sure they are the same length
        assert_equal(len(online_vars), len(target_vars))
        # Make sure that there is no overlap between online and target vars
        for var in online_vars:
            assert var not in target_vars

    def test_build_inputs(self):
        boards, processed_inputs = self.dqn_graph.build_inputs()
        # Check to make sure that types are correct
        assert_equal(boards.op.type, 'Placeholder')
        assert_equal(type(processed_inputs), tf.Tensor)
        # Check to make sure that the shapes of the inputs are correct
        assert_equal(boards.get_shape().as_list(), [None, 3, 3])
        assert_equal(processed_inputs.get_shape().as_list(), [None, 3, 3, 1])


class TestTicTacToeDQNGraphInfer(TestCase):
    def setUp(self):
        self.dqn_graph = TicTacToeDQNGraph(default_config)