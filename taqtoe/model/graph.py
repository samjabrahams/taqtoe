"""
Class file for TicTacToeDQNGraph
"""
import tensorflow as tf

import taqtoe.model.layers as layers


class TicTacToeDQNGraph:
    """
    Represents a Dueling Double DQN network graph. 
    
    Once the graph is built, the most important handles are the `graph`, 
    `endpoints`, and `inputs` attributes.
    
    `TicTacToeDQNGraph.graph` is the TensorFlow `Graph` object built according 
    to the provided configuration.
    
    `TicTacToeDQNGraph.inputs` is a dictionary mapping string names to 
    TensorFlow placeholder `Tensor` objects. Available keys:
    
    - 'boards': The input placeholder for the board(s) state for tic tac toe. It
        expects a Numpy array with shape `[None, 3, 3, 1]`
    
    The following inputs are also available if `config['is_training'] == True`
    
    - 'actions': The input placeholder containing a vector of chosen actions.
        Expects a one-dimensional vector of integers between [0-8]
    - 'rewards': The input placeholder containing a vector of rewards at a 
        given state. Expects a one-dimensional vector of floating point values.
    - 'learning_rate': The input placeholder specifying the learning rate. 
        Expects a scalar float input.
        
    TicTacToeDQNGraph.endpoints is a dictionary mapping string names to 
    TensorFlow `Tensor` and `Operation` objects which represent useful and/or 
    significant intermediate values and actions throughout the graph.
    Available keys:
        
    - 'processed_inputs': The board inputs after going through preprocessing.
    - 'target_v': The value path of the "target" DQN
    - 'target_a': The advantage path of the "target" DQN
    - 'target_q': The combined predicted Q of the "target" DQN (Q = V + A)
    - 'online_v': The value path of the "online" DQN
    - 'online_a': The advantage path of the "online" DQN
    - 'online_q': The combined predicted Q of the "online" DQN (Q = V + A)
    - 'global_step': The global step `Variable`.
    - 'saver': The `Saver` object, used to save and load weights.
    
    The following endpoints are also available if `config['is_training'] == 
    True`
    
    - 'loss': The loss of the Dueling Double DQN model.
    - 'train': An `Operation` which performs a step of gradient descent.
    - 'update_target': An `Operation` which copies the weights from the 
        "online" DQN to the "target" DQN.
    """
    def __init__(self, config):
        """
        Builds a TicTacToeDQNGraph.
        
        Required configuration parameters:
        
        - 'debug': Boolean. Specifies whether debug information should be 
            printed to the console.
        - 'gamma': Float in [0-1]. The discount rate for future versus 
            immediate rewards.
        - 'is_training': Boolean. Specifies if the graph should be 
            constructed in training mode.
        
        :param config: Dictionary mapping string keys to configuration options.
        """
        self.debug = config['debug']
        self.gamma = config['gamma']
        self.is_training = config['is_training']
        self.online_scope = 'online'
        self.target_scope = 'target'
        self.inputs = {}
        self.endpoints = {}
        self.graph = tf.Graph()
        self.build_graph()

    def build_graph(self):
        """
        Builds the DQN network inside of `self.graph`.
        """
        with self.graph.as_default():
            # Build essential graph components
            self.build_core_graph()
            # Build training graph if specified
            if self.is_training:
                self.build_train_graph()
            # Create global initialization function
            init = tf.global_variables_initializer()
            self.endpoints['init'] = init

    def build_core_graph(self):
        """
        Builds the core portion of the DQN graph. This will always be built.
        
        Broken down into `build_inputs()` and `build_dqn()`
        methods.
        """
        # Core inputs to the network
        boards, processed_inputs = self.build_inputs()
        # Build the "target" DQN
        target_v, target_a, target_q = \
            self.build_dqn(processed_inputs, self.target_scope)
        # Build the "online" DQN
        online_v, online_a, online_q = \
            self.build_dqn(processed_inputs, self.online_scope)
        # Create standard TensorFlow global step Variable
        global_step = tf.Variable(0, trainable=False, name='global_step')
        # Create Saver object to save and load Variables
        saver = tf.train.Saver()

        # Update inputs with created placeholder
        self.inputs.update({
            'boards': boards
        })
        # Update endpoints with core model objects
        self.endpoints.update({
            'processed_inputs': processed_inputs,
            'target_v': target_v,
            'target_a': target_a,
            'target_q': target_q,
            'online_v': online_v,
            'online_a': online_a,
            'online_q': online_q,
            'global_step': global_step,
            'saver': saver,
        })

    def build_train_graph(self):
        """
        Builds the training portion of the graph. Will only be run when 
        config['is_training'] is set to True.
        
        Broken down into `build_loss()`, `build_optimizer()`, 
        and 'build_update_target()` methods.
        """
        # Get required endpoints from core graph.
        target_q = self.endpoints['target_q']
        online_q = self.endpoints['online_q']
        global_step = self.endpoints['global_step']
        # Build the loss function
        actions, rewards, loss = self.build_loss(target_q, online_q)
        # Build the gradient descent optimization operations
        learning_rate, train = self.build_optimizer(loss, global_step)
        # Build an operation to copy weights from "online" DQN to "target" DQN
        update_target = self.build_update_target()

        # Update inputs with training placeholders
        self.inputs.update({
            'actions': actions,
            'rewards': rewards,
            'learning_rate': learning_rate
        })
        # Update endpoints with training model objects
        self.endpoints.update({
            'loss': loss,
            'train': train,
            'update_target': update_target,
        })

    def build_inputs(self):
        """
        Creates input placeholder for feeding tic-tac-to board state and 
        performs preprocessing for the model.
        
        :return: Tuple of Tensors. `(boards, preprocessed_inputs)` 
            corresponding to the board input placeholder and the result of 
            those inputs after preprocessing.
        """
        with tf.name_scope('inputs'):
            boards = tf.placeholder(tf.float32, [None, 3, 3], name='board')
            boards_reshaped = tf.reshape(boards, [-1, 3, 3, 1])
        return boards, boards_reshaped

    def build_dqn(self, inputs, scope):
        """
        Builds a deep dueling double deep Q-Network. This is the main method 
        describing the core model architecture.
        
        Here is the current model specification:
        
        ```
        - 2D Convolution; kernel size: 3x3, depth: 128, stride: 1, padding: 1
        - 2D Convolution; kernel size: 3x3, depth: 128, stride: 1, padding: 1
        - 2D Convolution; kernel size: 3x3, depth: 128, stride: 1, padding: 1
        - Dueling output layer (from Wang et al.) outputting a 9-dimensional 
            vector (one entry for each possible action from the network)
        ```
        
        Resources:
        
        "Deep Reinforcement Learning with Double Q-Learning"
        Hasselt et al.
        https://arxiv.org/abs/1509.06461
        
        "Dueling Network Architectures for Deep Reinforcement Learning"
        Wang et al.
        https://arxiv.org/abs/1511.06581

        :param inputs: 4D Tensor with shape `[batch_size, height, width, depth]`
        :param scope: String. A name for this DQN (usually will be either 
            'target' or 'online').
        :return: Tuple of Tensors. `(value, advantage, q)`. The predicted 
            value, advantage, and total Q values.
        """
        with tf.variable_scope(scope):
            # Convolutional layers
            conv = layers.conv2d(inputs, 128, [3, 3], scope='conv1')
            conv = layers.conv2d(conv, 128, [3, 3], scope='conv2')
            conv = layers.conv2d(conv, 128, [3, 3], scope='conv3')
            # Flatten from batch of 3D tensors to 1D vectors
            flat = layers.flatten(conv)
            # Dueling network output
            with tf.name_scope('dueling_streams'):
                value_path, advantage_path = tf.split(flat, 2, axis=1)
                value = layers.fully_connected(value_path, 1, scope='fc_value')
                advantage = layers.fully_connected(
                    advantage_path, 9, scope='fc_advantage')
                with tf.name_scope('Q'):
                    q = value + (advantage - tf.reduce_max(advantage, 1, True))

            return value, advantage, q

    def build_loss(self, target_q, online_q):
        """
        Creates Operations and placeholders for the loss function of the 
        network. Uses the Double Q-Learning algorithm from Hasselt et al. to 
        prevent overestimating action values.

        https://arxiv.org/abs/1509.06461

        :param target_q: The Q output of the "target" DQN
        :param online_q: The Q output of the "output" DQN
        :return: Tuple of Tensors. `(actions, rewards, loss)`. Placeholders 
            for the batch actions and rewards, as well as a handle to the 
            training loss.
        """
        with tf.name_scope('loss'):
            rewards = tf.placeholder(tf.float32, [None], name='rewards')
            rewards_expanded = tf.expand_dims(rewards, 1)
            actions = tf.placeholder(tf.int32, [None], name='actions')
            actions_one_hot = tf.one_hot(actions, 9)

            selected_q = tf.reduce_sum(online_q * actions_one_hot, 1, True)
            max_target_q = tf.gather(target_q, tf.argmax(online_q, 1))

            with tf.name_scope('y'):
                y = rewards_expanded + self.gamma * max_target_q
            with tf.name_scope('loss'):
                loss = tf.reduce_mean(tf.square(y - selected_q))

        return actions, rewards, loss

    def build_optimizer(self, loss, global_step):
        """
        Creates Operations and placeholders for performing gradient descent, 
        given a loss function.

        :param loss: `Tensor`. The value to minimize via gradient descent.
        :param global_step: A handle to the global step `Variable`.
        :return: A `Operation` which performs a gradient descent update.
        """
        with tf.name_scope('train'):
            learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
            opt = tf.train.AdamOptimizer(learning_rate)
            train = opt.minimize(
                loss,
                global_step=global_step,
                var_list=self.get_online_variables()
            )
        return learning_rate, train

    def build_update_target(self):
        """
        Creates Operations to set the Variables in the "target" DQN to the 
        corresponding values in the "online" DQN.

        :return: A `Operation` which runs all of the created `tf.assign` ops.
        """
        # Create dictionary mapping op name to op, minus the variable scope
        # This will allow us to align variables between online and target models
        online_vars = {
            op.name[op.name.index('/'):]: op
            for op in self.get_online_variables()
        }
        updates = []
        with tf.name_scope('target_updates'):
            for target_var in self.get_target_variables():
                key = target_var.name[target_var.name.index('/'):]
                online_var = online_vars[key]
                updates.append(tf.assign(target_var, online_var))
        with tf.control_dependencies(updates):
            # This allows us to group all of our update Ops in one go.
            update_target = tf.no_op('update_target')
        return update_target

    def _get_trainable_variables_from_scope(self, s):
        """
        Helper to get all trainable Variables from a named Graph scope.

        :param s: String. The prefix of the scope to get variables from.
        :return: List of TensorFlow `Variable` objects.
        """
        return self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, s)

    def get_target_variables(self):
        """
        Helper to get a list of all variables from the "target" DQN.

        :return: List of TensorFlow `Variable` objects.
        """
        return self._get_trainable_variables_from_scope(self.target_scope)

    def get_online_variables(self):
        """
        Helper to get a list of all variables from the "online" DQN.

        :return: List of TensorFlow `Variable` objects.
        """
        return self._get_trainable_variables_from_scope(self.online_scope)


if __name__ == '__main__':
    # Create a graph and export it to TensorBoard
    from taqtoe.model.config import default_config
    model = TicTacToeDQNGraph(default_config)
    tf.summary.FileWriter('test', graph=model.graph).close()
