


if __name__ == '__main__':
    data_dir = "data/MIDI/piano-roll/JSB Chorales.pickle"
    n_epochs = 12
    batch_size = 128
    n_notes = 88

    log_dir = './logs/debug/tcb'

    print("Loading dataset...")
    train_dataset = PianoRollData(data_dir, 'train', skip=500)
    val_dataset = PianoRollData(data_dir, 'valid', skip=500)

    par = {'in_size': 88, 'hidden_size': 10, 'k': 5, 'out_size': 88}
    cbs = [TBCallback(log_dir, input_dim=(5, 1, 88))]
    ulm = UnrolledLMTrainer(par, batch_size, n_epochs, log_dir, callbacks=cbs)
    ulm.fit(train_dataset, val_dataset)
