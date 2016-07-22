from __future__ import division

import tensorflow as tf
from tensorflow.models.rnn import rnn

from input_reader import VOCAB_SIZE


BATCH_SIZE = 100
N_STEP = 100
HIDDEN_SIZE = 200
NUM_LAYER = 2

class ReviewCharModel(object):

    def __init__(
        self,
        batch_size,
        num_steps,
        hidden_size,
        num_layers=1,
        vocab_size=VOCAB_SIZE,
        lr=0.01,
    ):
        self._input = tf.placeholder(tf.float32, [batch_size, num_steps, vocab_size])
        self._target = tf.placeholder(tf.float32, [batch_size, num_steps, vocab_size])
        self._keep_prob = tf.placeholder(tf.float32)

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
            lstm_cell, output_keep_prob=self._keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers)

        self._init_state = cell.zero_state(batch_size, tf.float32)

        inputs = [tf.squeeze(input_, [1])
                  for input_ in tf.split(1, num_steps, self._input)]
        outputs, state = rnn.rnn(cell, inputs, initial_state=self._init_state)

        output = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])
        softmax_w = tf.get_variable(
            'softmax_w',
            shape=[hidden_size, vocab_size],
            initializer=tf.truncated_normal_initializer(stddev=0.1),
        )
        softmax_b = tf.get_variable(
            'softmax_b',
            shape=[vocab_size],
        )
        logits = tf.matmul(output, softmax_w) + softmax_b
        self._char_prob = tf.nn.softmax(logits)

        # Use prediction cross entropy as the loss function
        target_reshaped = tf.reshape(self._target, [-1, vocab_size])
        self._cost = -tf.reduce_mean(
            tf.reduce_sum(target_reshaped * tf.log(self._char_prob), reduction_indices=[1])
        )

        self._final_state = state

        self._optimize = tf.train.AdamOptimizer(learning_rate=lr).minimize(self._cost)

    def get_init_state(self, session):
        state, =  session.run([self._init_state])
        return state

    def train(self, session, input_x, target, state, keep_prob=0.5):
        if state is None:
            state = self._init_state
        cost, state, _ = session.run(
            [self._cost, self._final_state, self._optimize],
            feed_dict={
                self._input: input_x,
                self._target: target,
                self._init_state: state,
                self._keep_prob: keep_prob
            },
        )
        return cost, state

    def eval(self, session, input_x, target, state):
        cost, = session.run(
            [self._cost],
            feed_dict={
                self._input: input_x,
                self._target: target,
                self._init_state: state,
                self._keep_prob: 1,
            },
        )
        return cost

    def generate(self, session, input, state):
        char_prob, state = session.run(
            [self._char_prob, self._final_state],
            feed_dict={
                self._input: input,
                self._init_state: state,
                self._keep_prob: 1,
            },
        )
        return char_prob, state
