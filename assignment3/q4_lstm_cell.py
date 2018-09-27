#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q4: LSTM
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys

import tensorflow as tf
import numpy as np

logger = logging.getLogger("hw3.q4")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class LSTMCell(tf.nn.rnn_cell.RNNCell):
    """Wrapper around our LSTM cell implementation that allows us to play
    nicely with TensorFlow.
    """
    def __init__(self, input_size, state_size):
        self.input_size = input_size
        self._state_size = state_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state, scope=None):
        """Updates the state using the previous @state and @inputs.
        Remember the LSTM equations are:

        z_t = sigmoid(x_t W_z + h_{t-1} U_z + b_z)
        r_t = sigmoid(x_t W_r + h_{t-1} U_r + b_r)
        o_t = tanh(x_t W_o + r_t * h_{t-1} U_o + b_o)
        h_t = z_t * h_{t-1} + (1 - z_t) * o_t

        TODO: In the code below, implement an GRU cell using @inputs
        (x_t above) and the state (h_{t-1} above).
            - Define U_r, W_r, b_r, U_z, W_z, b_z and U_o, W_o, b_o to
              be variables of the apporiate shape using the
              `tf.get_variable' functions.
            - Compute z, r, o and @new_state (h_t) defined above
        Tips:
            - Remember to initialize your matrices using the xavier
              initialization as before.
        Args:
            inputs: is the input vector of size [None, self.input_size]
            state: is the previous state vector of size [None, self.state_size]
            scope: is the name of the scope to be used when defining the variables inside.
        Returns:
            a pair of the output vector and the new state vector.
        """
        scope = scope or type(self).__name__

        # It's always a good idea to scope variables in functions lest they
        # be defined elsewhere!
        with tf.variable_scope(scope):
            ### YOUR CODE HERE (~20-30 lines)
            W_r = tf.get_variable('W_r', shape=(self.input_size, self._state_size), initializer=tf.contrib.layers.xavier_initializer())
            U_r = tf.get_variable('U_r', shape=(self._state_size, self._state_size), initializer=tf.contrib.layers.xavier_initializer())
            b_r = tf.get_variable('b_r', shape=(self._state_size,), initializer=tf.constant_initializer(0))
            r_t = tf.nn.sigmoid(tf.matmul(inputs, W_r) + tf.matmul(state, U_r) + b_r)
            
            W_z = tf.get_variable('W_z', shape=(self.input_size, self._state_size), initializer=tf.contrib.layers.xavier_initializer())
            U_z = tf.get_variable('U_z', shape=(self._state_size, self._state_size), initializer=tf.contrib.layers.xavier_initializer())
            b_z = tf.get_variable('b_z', shape=(self._state_size,), initializer=tf.constant_initializer(0))
            z_t = tf.nn.sigmoid(tf.matmul(inputs, W_z) + tf.matmul(state, U_z) + b_z)
            
            W_o = tf.get_variable('W_o', shape=(self.input_size, self._state_size), initializer=tf.contrib.layers.xavier_initializer())
            U_o = tf.get_variable('U_o', shape=(self._state_size, self._state_size), initializer=tf.contrib.layers.xavier_initializer())
            b_o = tf.get_variable('b_o', shape=(self._state_size,), initializer=tf.constant_initializer(0))
            o_t = tf.nn.tanh(tf.matmul(inputs, W_o) + tf.matmul(state * r_t, U_o) + b_o)
            
            new_state = z_t * state + (1 - z_t) * o_t
            ### END YOUR CODE ###
        # For a GRU, the output and state are the same (N.B. this isn't true
        # for an LSTM, though we aren't using one of those in our
        # assignment)
        return output, new_state
