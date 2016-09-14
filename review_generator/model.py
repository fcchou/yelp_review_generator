import tensorflow as tf
import json


class CharRNN(object):
    def __init__(
        self,
        hidden_size,
        num_layers,
        vocab_size,
        batch_size=100,
        num_steps=50,
        lr=0.001,
    ):
        """Multi-layer recurrent neural net for characters.

        Args:
            hidden_size (int): Size of RNN's hidden state
            num_layers (int): Number of stacked RNN layers
            vocab_size (int): Size of the vocabulary (i.e. number of valid characters)
            batch_size (int): Size of the mini-batch.
            num_steps (int): Number of unrolled steps for RNN training
            lr (float): Learning rate
        """
        self._input = tf.placeholder(tf.float32, [batch_size, num_steps, vocab_size])
        self._target = tf.placeholder(tf.float32, [batch_size, num_steps, vocab_size])
        self._keep_prob = tf.placeholder(tf.float32)

        rnn_cell = tf.nn.rnn_cell.GRUCell(hidden_size, activation=tf.nn.relu)
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(
            rnn_cell, output_keep_prob=self._keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * num_layers)

        self._init_state = cell.zero_state(batch_size, tf.float32)

        inputs = [tf.squeeze(input_, [1])
                  for input_ in tf.split(1, num_steps, self._input)]
        outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._init_state)

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

        # Clip the predicted probability so the log-loss does not become nan
        epsilon = 1e-5
        self._char_prob = tf.clip_by_value(tf.nn.softmax(logits), epsilon, 1 - epsilon)

        # Use prediction cross entropy as the loss function
        target_reshaped = tf.reshape(self._target, [-1, vocab_size])
        self._cost = -tf.reduce_mean(
            tf.reduce_sum(target_reshaped * tf.log(self._char_prob), reduction_indices=[1])
        )

        self._final_state = state

        self._optimize = tf.train.AdamOptimizer(learning_rate=lr).minimize(self._cost)

    @classmethod
    def from_config_file(
        cls,
        config_file,
        batch_size=100,
        num_steps=50,
        lr=0.001,
    ):
        """Construct CharRNN model from config file.

        The config file is in JSON format as follows:
        {"hidden_size": 100, "num_layers": 2, "vocab_size": 128}

        Args:
            config_file(str): Path to the JSON format config.
            batch_size (int): Size of the mini-batch.
            num_steps (int): Number of unrolled steps for RNN training
            lr (float): Learning rate

        Returns:
            A CharRNN object.
        """
        with open(config_file) as f:
            config = json.load(f)
        return cls(**config, batch_size=batch_size, num_steps=num_steps, lr=lr)

    def get_init_state(self, session):
        """Get the initial hidden state.

        Args:
            session (tf.Session): An initiated tensorflow session.

        Returns:
            Hidden state vector.
        """
        state, = session.run([self._init_state])
        return state

    def train(self, session, input_x, target, state, keep_prob=0.5):
        """Train the model with a mini-batch .

        Args:
            session (tf.Session): An initiated tensorflow session.
            input_x (array-like):
                Input mini-batch. Must match the shape (batch_size, num_steps, vocab_size).
                The input character is encoded as an one-hot vector.
            target (array-like):
                Target characters to be modeled. Has the same shape as input_x.
            state (array-like): Hidden state vector
            keep_prob (float): The probability to keep the output during drop-out.

        Returns:
            cost (float): Value of current cost function
            state (array): Updated hidden state.
        """
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

    def generate(self, session, input_x, state):
        """Generate character prediction

        Args:
            session (tf.Session): An initiated tensorflow session.
            input_x (array-like):
                Input vector. Must match the shape (batch_size, num_steps, vocab_size).
            state (array-like): Hidden state vector

        Returns:
            Predicted probability of character.
        """
        char_prob, state = session.run(
            [self._char_prob, self._final_state],
            feed_dict={
                self._input: input_x,
                self._init_state: state,
                self._keep_prob: 1,
            },
        )
        return char_prob, state
