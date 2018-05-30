import sys
sys.path.append('./src')
from tasks import PianoRollData
import torch


def test_piano_roll():
    train_dataset = PianoRollData('./data/MIDI/piano-roll/JSB Chorales.pickle', 'train', timesteps=50)
    # test one_hot_list generator
    print("test data loader")
    for song_x, song_y in train_dataset.get_one_hot_list():
        assert torch.equal(song_x[1:], song_y[:-1])

    # test get_data
    X_train, y_train, masks = train_dataset.get_data()
    print("test data loader")
    for i in range(0, X_train.size(1)):
        X_i, y_i = X_train[:, i, :], y_train[:, i, :]
        mask_i = masks[:, i]
        max_idx = int(mask_i.sum().data.numpy()[0])  # need to keep track of padding
        assert torch.equal(X_i[1:max_idx], y_i[:max_idx-1])


def test_masking():
    train_dataset = PianoRollData('./data/MIDI/piano-roll/JSB Chorales.pickle', 'train', timesteps=50)
    batch_size = 32
    X_train, y_train, masks = train_dataset.get_data()
    print("test masking")
    for i in range(0, X_train.size(1), batch_size):
        y_i = y_train[:, i:i + batch_size, :]
        mask_i = masks[:, i:i + batch_size]
        y_mask = y_i * mask_i.expand_as(y_i)
        assert torch.equal(y_i, y_mask)


if __name__ == '__main__':
    test_piano_roll()
    test_masking()
    print("test DONE.")
