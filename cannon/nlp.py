import pickle
import numpy as np


class GloveEmbedding:
    def __init__(self):
        # download at: https://github.com/PrincetonML/SIF/blob/master/auxiliary_data/enwiki_vocab_min200.txt
        freq_file = 'data/enwiki_vocab_min200.txt'
        self.word_to_row, self.W_emb = try_load_glove()

        for col_id in range(self.W_emb.shape[1]):
            col_norm = np.linalg.norm(self.W_emb[:, col_id])
            self.W_emb[:, col_id] = self.W_emb[:, col_id] / col_norm

        self.unknown = np.mean(self.W_emb, axis=0)
        self.unknown = self.unknown / np.linalg.norm(self.unknown)

        self.word_to_freq_dict = get_word_to_freq(freq_file)

        self.id_to_freq_dict = {}
        for k, v in self.word_to_freq_dict.items():
            if k in self.word_to_row:
                word_id = self.word_to_row[k]
                self.id_to_freq_dict[word_id] = v

    def word_to_id(self, word):
        return get_word_idx(word, self.word_to_row)

    def __getitem__(self, idx):
        if idx == -1:
            return self.unknown
        return self.W_emb[idx]

    def id_to_freq(self, idx):
        if idx in self.id_to_freq_dict:
            return self.id_to_freq_dict[idx]
        else:
            # print("NO weight: {}".format(idx))
            return 1.0

    def word_to_freq(self, word):
        if word in self.word_to_freq_dict:
            return self.word_to_freq_dict[word]
        else:
            return 0


def get_word_to_freq(weight_file):
    word_to_weight = {}
    tot = 0
    with open(weight_file) as f:
        for line in f:
            line = line.strip().split()
            if len(line) == 2:
                word, weight = line[0], float(line[1])
                word_to_weight[word] = weight
                tot += weight

    for k, v in word_to_weight.items():
        word_to_weight[k] = v / tot
    return word_to_weight


def try_load_glove():
    glove_file = '/home/carta/data/glove.840B.300d.txt'
    matrix_file = '/home/carta/data/glove_matrix.npy'
    dict_file = '/home/carta/data/glove_word_to_row.pickle'
    try:
        print("loading from preprocessed files.")
        with open(dict_file, 'rb') as f:
            word_to_row = pickle.load(f)
        W_emb = np.load(matrix_file)
    except FileNotFoundError:
        print("GloVe preprocessed files not found. Loading from {}".format(glove_file))
        word_to_row, W_emb = get_word_emb(glove_file)
        print("word vectors loaded.")

        # save preloaded word embeddings
        with open(dict_file, 'wb') as f:
            pickle.dump(word_to_row, f)
        np.save(matrix_file, W_emb)
    print("Loading completed.")
    return word_to_row, W_emb


def get_word_emb(emb_file):
    word_to_rowid ={}
    emb_matrix = []
    with open(emb_file, 'r') as f:
        n = 0
        for line in f:
            try:
                line = line.split()
                word = line[0]
                emb = [float(el) for el in line[1:]]
                if len(emb) != 300:
                    p = ' '.join(line[:10] + ['...'])
                    print("ERROR: wrong vector length at line {} ({})".format(n+1, p))
                    continue
                # add emb to dictionary
                word_to_rowid[word] = n
                emb_matrix.append(emb)
                n += 1
            except ValueError:
                p = ' '.join(line[:10] + ['...'])
                print("error on line {}: {}".format(n+1, p))
    return word_to_rowid, np.array(emb_matrix)


def get_word_idx(word, word_to_row):
    if word in word_to_row:
        return word_to_row[word]
    elif "unk" in word_to_row:
        print("Unknown word: {}".format(word))
        return -1 #word_to_row["unk"]
    else:
        print("Unknown word (no 'unk' token available): {}".format(word))
        return -1


if __name__ == '__main__':
    # load GloVe embeddings
    glove = GloveEmbedding()
