"""
Layer functions for taqtoe.model
"""
import numpy as np
import tensorflow as tf


def get_shape(tensor):
    """
    Helper to get the shape of a `Tensor` in list form
    
    :param tensor: The Tensor to get the shape from.
    :returns: List of integers representing the shape of `tensor`
    """
    return tensor.get_shape().as_list()


def compute_fans(shape):
    """
    Compute fan-in and fan-out of a given shape. Useful for Xavier and He 
    methods of initializing weights.
    
    :param shape: List of integers representing a Tensor shape.
    :returns: Tuple of integers, `(fan_in, fan_out)`, representing the number of 
        total inputs and outputs from this weight, respectively.
    """
    if len(shape) == 2:
        # 2D weight matrix, which maps from fan-in to fan-out
        # Shape: [fan-in, fan-out]
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 4:
        # 4D weight matrix, representing a set of convolutional kernels/filters
        # Shape: [kernel_height, kernel_width, in_channels, out_channels]

        kernel_area = shape[0] * shape[1]
        fan_in = shape[2] * kernel_area
        fan_out = shape[3] * kernel_area
    else:
        # We'll leave out other shaped weights for this implementation
        raise ValueError(
            'Invalid shape {}: shape must have either 2 or 4 dimensions.'.format(
                shape))
    return fan_in, fan_out


def he_initializer(shape):
    """
    Returns a TensorFlow initializer using the recommended function from He 
    et al (https://arxiv.org/abs/1502.01852) for ReLU activation functions.
    
    stddev = sqrt(2 / num_inputs)
    
    :param shape: List of integers. The shape of the weight we're initializing.
    :return: A TensorFlow initializer (for `tf.get_variable()`)
    """
    fan_in, _ = compute_fans(shape)
    stddev = np.sqrt(2.0 / fan_in)
    return tf.truncated_normal_initializer(stddev=stddev)


def flatten(inputs, is_batched=True, scope=None):
    """
    Flattens `inputs` into 1D vectors. If `is_batched` is set to `True`, 
    the first dimension (the `num_batch` dimension) of `inputs` will remain the 
    same.
    
    :param inputs: `Tensor` to be flattened.
    :param is_batched: Boolean. Specifies if the first dimensions of the 
        inputs should remain the same.
    :param scope: Optional string. Provides a name to the scope of the 
        operation.
    :return: 
    """
    with tf.name_scope(scope, 'flatten'):
        shape = get_shape(inputs)
        if is_batched:
            num_units = np.prod(shape[1:])
            return tf.reshape(inputs, [-1, num_units])
        else:
            num_units = np.prod(shape)
            return tf.reshape(inputs, [num_units])


def fully_connected(inputs, num_outputs, weight_initializer=he_initializer,
                    bias_value=0.01, activation=tf.nn.relu, reuse=False,
                    scope=None):
    """
    Creates a fully connected neural network layer. 
   
    `fully_connected` has the following features:
    
    * Automatically creates weight and bias Variables based on `inputs` and 
        `num_outputs`.
    * Weight initializer, initial bias values, and activation function are 
        fully adjustable
    * `reuse` allows user to use the same weights for multiple versions of a 
        model. 
    
    Although scope has a default value of `None`, this method requires the 
    user to input a value for it in order to properly make use of 
    `tf.get_variable()`
    
    :param inputs: Tensor. The input to this layer (e.g. from a previous layer)
    :param num_outputs: Integer. The size of the output vector. The "number of 
        neurons".
    :param weight_initializer: Initializer function which should map from `(
        list(int) -> TensorFlow initializer)`
    :param bias_value: Float. The initial value for the layer's bias parameters.
    :param activation: Function mapping from `(Tensor -> Tensor)`. Defaults 
        to `tf.nn.relu`.
    :param reuse: Boolean. Specifies if the weights in this layer should be 
        reused with `tf.get_variable()`. 
    :param scope: String. The name of this layer. If `reuse` is False, 
        then this must be a new string unique to this layer. If `reuse` is 
        True, then this must be set to a value used by a previous call to 
        `fully_connected()`.
    :return: Tensor. The transformed output from this layer.
    """
    if scope is None:
        raise ValueError(
            'Parameter scope to fully_connected() cannot be None.')
    with tf.variable_scope(scope, reuse=reuse):
        # Create/get weight and bias Variables
        num_inputs = get_shape(inputs)[-1]
        w_shape = [num_inputs, num_outputs]
        w_init = weight_initializer(w_shape)
        b_init = tf.constant_initializer(bias_value)
        weights = tf.get_variable('weights', shape=w_shape, initializer=w_init)
        biases = tf.get_variable('biases', shape=[num_outputs],
                                 initializer=b_init)
        z = tf.matmul(inputs, weights) + biases
        return z if activation is None else activation(z)


def conv2d(inputs, out_channels, kernel_size, strides=[1, 1], padding='SAME',
           weight_initializer=he_initializer, bias_value=0.01,
           activation=tf.nn.relu, reuse=False, scope=None):
    """
    Creates a 2D convolutional neural network layer.
    
    `conv2d` has the following features:
    
    * Automatically creates weight and bias Variables based on `inputs` and 
        `out_channels`.
    * Weight initializer, initial bias values, and activation function are 
        fully adjustable.
    * Provides easier format for `kernel_size` and `strides`
    * `reuse` allows user to use the same weights for multiple versions of a 
        model.
    
    Although scope has a default value of `None`, this method requires the 
    user to input a value for it in order to properly make use of 
    `tf.get_variable()`
    
    :param inputs: Tensor. The input to this layer (e.g. from a previous layer)
    :param out_channels: Integer. The number of output channels this layer 
        should have. This is equivalent to setting the number of kernels.
    :param kernel_size: List of integers specifying `[height, width]`. The 
        dimensions of the kernels in the layer.
    :param strides: List of integers specifying `[vertical_stride, 
        horizontal_stride]`. The amount of spaces to move in either direction
        when performing convolutions.
    :param padding: String. Either 'VALID' or 'SAME'. Specifies the padding 
        strategy to use.
    :param weight_initializer: Initializer function which should map from `(
        list(int) -> TensorFlow initializer)`
    :param bias_value: Float. The initial value for the layer's bias parameters.
    :param activation: Function mapping from `(Tensor -> Tensor)`. Defaults 
        to `tf.nn.relu`.
    :param reuse: Boolean. Specifies if the weights in this layer should be 
        reused with `tf.get_variable()`.
    :param scope: String. The name of this layer. If `reuse` is False, 
        then this must be a new string unique to this layer. If `reuse` is 
        True, then this must be set to a value used by a previous call to 
        `fully_connected()`.
    :return: Tensor. The transformed output from this layer.
    """
    if scope is None:
        raise ValueError('Parameter scope to conv2d() cannot be None.')
    with tf.variable_scope(scope):
        input_shape = inputs.get_shape().as_list()
        strides = [1] + strides + [1]
        in_channels = input_shape[-1]
        filter_shape = kernel_size + [in_channels, out_channels]
        w_init = weight_initializer(filter_shape)
        b_init = tf.constant_initializer(bias_value)
        filters = tf.get_variable(
            'filters', shape=filter_shape, initializer=w_init)
        biases = tf.get_variable(
            'biases', shape=[out_channels], initializer=b_init)
        conv = tf.nn.conv2d(inputs, filters, strides, padding)
        z = tf.nn.bias_add(conv, biases)
        return z if activation is None else activation(z)