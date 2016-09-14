import tensorflow as tf
import argparse

from review_generator.input_reader import get_raw_data_from_file, char_iterator
from review_generator.model import CharRNN

parser = argparse.ArgumentParser(description='Train the CharRNN model')
parser.add_argument('--input-file', dest='input_file', help='Input raw text file', required=True)
parser.add_argument(
    '--output-model',
    dest='output_model',
    help='Output location for the model checkpoint',
    default='char_rnn_model.ckpt',
)
parser.add_argument(
    '--model-config',
    dest='model_config',
    help='Config file containing model spec',
    required=True,
)
parser.add_argument(
    '--batch-size',
    dest='batch_size',
    help='Size of the mini-batch',
    type=int,
    default=100,
)
parser.add_argument(
    '--num-steps',
    dest='num_steps',
    help='Number of unrolled steps in truncated back-prop',
    type=int,
    default=100,
)
parser.add_argument(
    '--learning-rate',
    dest='learning_rate',
    help='Learning rate for model training',
    type=float,
    default=0.001,
)
parser.add_argument(
    '--num-checkpoint-steps',
    dest='num_checkpoint_steps',
    help='Number of mini-batches between each checkpointing',
    type=int,
    default=300,
)
parser.add_argument(
    '--num-training-epochs',
    dest='num_training_epochs',
    help='Number of epochs (passes through the whole data) in training',
    type=int,
    default=10,
)
parser.add_argument(
    '--warm-start',
    dest='warm_start',
    help='Restart the training from the given checkpoint file',
)
args = parser.parse_args()


train_data = get_raw_data_from_file(args.input_file)

model_train = CharRNN.from_config_file(
    args.model_config,
    batch_size=args.batch_size,
    num_steps=args.num_steps,
    lr=args.learning_rate,
)

session = tf.Session()
session.run(tf.initialize_all_variables())
saver = tf.train.Saver()

if args.warm_start is not None:
    saver.restore(session, args.warm_start)

state = model_train.get_init_state(session)

cost_sum = 0
n_batch_processed = 0
for i in range(args.num_training_epochs):
    print('Enter epoch {}'.format(i))
    for x, y in char_iterator(train_data, args.batch_size, args.num_steps):
        cost, state = model_train.train(session, x, y, state)
        cost_sum += cost
        if (n_batch_processed + 1) % args.num_checkpoint_steps == 0:
            saver.save(session, args.output_model)
            print('mini-batch #{}. mean_cost:'.format(n_batch_processed), cost_sum / args.num_checkpoint_steps)
            cost_sum = 0
        n_batch_processed += 1
saver.save(session, args.output_model)
