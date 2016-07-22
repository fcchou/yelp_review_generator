import tensorflow as tf
import numpy as np
from input_reader import VOCAB_SIZE, get_raw_data_from_text
from review_model import ReviewCharModel, HIDDEN_SIZE, NUM_LAYER
from train import MODEL_PATH


with tf.variable_scope("model"):
    model_gen = ReviewCharModel(1, 1, HIDDEN_SIZE, NUM_LAYER)
session = tf.Session()
session.run(tf.initialize_all_variables())
saver = tf.train.Saver()
saver.restore(session, MODEL_PATH)

# Generate stuffs
def get_input_from_char_id(char_id):
    input_vec = np.zeros([1, 1, VOCAB_SIZE])
    input_vec[0, 0, char_id] = 1
    return input_vec


def sample_char(output_prob):
    output_prob = output_prob.reshape(-1)
    output_prob /= np.sum(output_prob)
    return np.random.choice(np.arange(VOCAB_SIZE), p=output_prob)


initial_text = 'Great food!'

final_text = initial_text
init_data = get_raw_data_from_text(initial_text)
state = model_gen.get_init_state(session)
output = None
for c in init_data:
    output, state = model_gen.generate(session, get_input_from_char_id(c), state)

for i in xrange(500):
    c = sample_char(output)
    output, state = model_gen.generate(session, get_input_from_char_id(c), state)
    final_text += chr(c)

print final_text