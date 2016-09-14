import numpy as np
import tensorflow as tf
import argparse
import json

from review_generator.model import CharRNN


parser = argparse.ArgumentParser(description='Generate text from trained CharRNN model')
parser.add_argument(
    '--model_ckpt',
    dest='model_ckpt',
    help='Checkpoint file of trained model',
    default='char_rnn_model.ckpt',
)
parser.add_argument(
    '--model-config',
    dest='model_config',
    help='Config file containing model spec',
    required=True,
)

parser.add_argument(
    '--temperature',
    dest='temperature',
    help='Boltzmann temperature in character generation. Lower temperature means less randomness '
         '(only generate the most probable text sequence)',
    type=float,
    default=1.0,
)
parser.add_argument(
    '--len',
    dest='len',
    help='Length of the generated text (in characters)',
    type=int,
    default=5000,
)
parser.add_argument(
    '--use-cpu',
    dest='use_cpu',
    help='Use CPU for generation',
    action='store_true'
)
args = parser.parse_args()


if args.use_cpu:
    with tf.device('/cpu:0'):
        model = CharRNN.from_config_file(args.model_config, batch_size=1, num_steps=1)
else:
    model = CharRNN.from_config_file(args.model_config, batch_size=1, num_steps=1)

vocab_size = json.load(open(args.model_config))['vocab_size']

session = tf.Session()
session.run(tf.initialize_all_variables())
saver = tf.train.Saver()
saver.restore(session, args.model_ckpt)


# Generate stuffs
def get_input_from_char_id(char_id):
    input_vec = np.zeros([1, 1, vocab_size])
    input_vec[0, 0, char_id] = 1
    return input_vec


def sample_char(output_prob):
    output_prob = output_prob.reshape(-1)
    energy = -np.log(output_prob)
    prob = np.exp(-energy / args.temperature)
    prob /= np.sum(prob)
    return np.random.choice(np.arange(vocab_size), p=prob)


def generate(length):
    state = model.get_init_state(session)
    final_text = b''
    c = 0

    for i in range(length):
        output, state = model.generate(session, get_input_from_char_id(c), state)
        c = sample_char(output)
        char = bytearray((c,))
        final_text += char
    return final_text.replace(b'\0', b'\n------------------\n')


gen_text = generate(args.len)
print(gen_text.decode('utf-8'))
