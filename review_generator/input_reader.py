import numpy as np

VOCAB_SIZE = 128


def get_raw_data_from_file(data_path):
    """Load the text from file and convert to integer id (ASCII).

    Args:
        data_path (str): Path to the file

    Returns:
        1D numpy array of 8bit integers.
    """
    with open(data_path, 'rb') as f:
        word_ids = np.fromfile(f, dtype='S1').view(dtype=np.uint8)
    return word_ids


def get_raw_data_from_text(input_str):
    """Convert string to integer id (ASCII).

    Args:
        input_str (str): String to be converted

    Returns:
        1D numpy array of 8bit integers.
    """
    input_str = bytearray(input_str)
    return np.fromiter(input_str, dtype='S1').view(dtype=np.uint8)


def char_iterator(raw_data, batch_size, num_steps):
    """Convert raw input data into mini-batches.

    Args:
        raw_data (np.ndarray): 1D numpy array of np.uint8, for the characters encoded in ASCII
        batch_size (int): Size of the mini-batch.
        num_steps (int): Number of characters in the sequence for each mini-batch, for truncated back-prop.

    Returns:
        A generator of mini-batches of (x, y). Here x and y are one-hot encoded character mini-batches for the training
        input (current character) and output (next character).
    """
    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = raw_data[:(batch_len * batch_size)].reshape(batch_size, batch_len)

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x_int = data[:, i * num_steps:(i + 1) * num_steps]
        x_onehot = encode_onehot(x_int)
        y_int = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        y_onehot = encode_onehot(y_int)
        yield (x_onehot, y_onehot)


def encode_onehot(x_int, max_val=VOCAB_SIZE):
    """Encode integer vector into one-hot representation

    Args:
        x_int (np.ndarray): 1D numpy array of integer categorical features.
        max_val (int): The max value of the integer categories

    Returns:
        One hot encoded (binary) feature vector.
    """
    x_size, y_size = x_int.shape
    x_temp = np.zeros([x_size * y_size, max_val], dtype=np.float32)
    x_temp[[np.arange(x_size * y_size), x_int.reshape(-1)]] = 1
    x_onehot = x_temp.reshape([x_size, y_size, max_val])
    return x_onehot
