import torch
from cannon.utils import cuda_move


def ints_to_chars(seq):
    return [chr(s) for s in seq]


def chars_to_tensor(chars, max_char=255):
    chars = [ord(c) for c in chars]
    data = torch.tensor(chars)

    if max_char is not None:
        data[data > max_char] = max_char + 1
    return data


def get_enwik8_splits(data_file, batch_size, debug = False):
    with open(data_file, 'r') as f:
        if debug:
            chars = f.read(10 ** 4)
        else:
            chars = f.read()
    print("loaded {} chars".format(len(chars)))

    data = chars_to_tensor(chars)
    train_split_idx = int(len(chars) * 0.6)
    train_data = data[:train_split_idx]
    dim_batch = train_data.shape[0] // batch_size
    train_data = train_data[:dim_batch * batch_size].reshape(batch_size, -1)
    train_data = train_data.transpose(0, 1)
    train_data = cuda_move(train_data)

    val_split_idx = int(len(chars) * 0.8)
    val_data = data[train_split_idx:val_split_idx]
    dim_batch = val_data.shape[0] // batch_size
    val_data = val_data[:dim_batch * batch_size].reshape(batch_size, -1)
    val_data = val_data.transpose(0, 1)
    val_data = cuda_move(val_data)

    test_data = data[val_split_idx:]
    dim_batch = test_data.shape[0] // batch_size
    test_data = test_data[:dim_batch * batch_size].reshape(batch_size, -1)
    test_data = test_data.transpose(0, 1)
    test_data = cuda_move(test_data)
    return train_data, val_data, test_data