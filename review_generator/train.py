import tensorflow as tf
from input_reader import get_raw_data_from_file, char_iterator
from review_model import ReviewCharModel, HIDDEN_SIZE, NUM_LAYER

BATCH_SIZE = 100
N_STEP = 100
NUM_EPOCH = 5
MODEL_PATH = '/tmp/fchou_model.ckpt'

train_data = get_raw_data_from_file('/Users/fchou/review_training.txt')

with tf.variable_scope("model"):
    model_train = ReviewCharModel(BATCH_SIZE, N_STEP, HIDDEN_SIZE, NUM_LAYER)

session = tf.Session()
session.run(tf.initialize_all_variables())
saver = tf.train.Saver()
saver.restore(session, MODEL_PATH)

state = model_train.get_init_state(session)

i = 0
cost_sum = 0
for _ in xrange(NUM_EPOCH):
    for x, y in char_iterator(train_data, BATCH_SIZE, N_STEP):
        cost, state = model_train.train(session, x, y, state)
        cost_sum += cost
        i += 1
        if i % 100 == 0:
            saver.save(session, MODEL_PATH)
            print 'mean_cost', float(cost_sum) / i
saver.save(session, MODEL_PATH)