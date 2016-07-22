import numpy as np

VOCAB_SIZE = 128

def get_raw_data_from_file(data_path):
    word_ids = np.fromfile(data_path, dtype='S1').view(dtype=np.uint8)
    return word_ids


def get_raw_data_from_text(input_str):
    return np.fromiter(input_str, dtype='S1').view(dtype=np.uint8)


def char_iterator(raw_data, batch_size, num_steps):
    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = raw_data[:(batch_len * batch_size)].reshape(batch_size, batch_len)

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in xrange(epoch_size):
        x_int = data[:, i * num_steps:(i + 1) * num_steps]
        x_onehot =encode_onehot(x_int)
        y_int = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        y_onehot =encode_onehot(y_int)
        yield (x_onehot, y_onehot)


def encode_onehot(x_int):
    x_size, y_size = x_int.shape
    x_temp = np.zeros([x_size * y_size, VOCAB_SIZE], dtype=np.float32)
    x_temp[[np.arange(x_size * y_size), x_int.reshape(-1)]] = 1
    x_onehot = x_temp.reshape([x_size, y_size, VOCAB_SIZE])
    return x_onehot
