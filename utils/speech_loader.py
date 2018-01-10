#-*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import csv
import codecs
import string
import librosa
from collections import Counter
import random
import glob
import pandas as pd
from six.moves import cPickle

train_file = "data/train.csv"
_data_path = "data/"

# index to byte mapping
index2byte = ['<EMP>', ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
              'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
              'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# byte to index mapping
byte2index = {}
for i, ch in enumerate(index2byte):
    byte2index[ch] = i

# vocabulary size
vocab_size = len(index2byte)

class SpeechLoader():

    def __init__(self, wav_path=None, label_file=None, batch_size=1, n_mfcc=20, encoding='utf-8'):
        self.batch_size = batch_size
        self.encoding = encoding
        self.n_mfcc = n_mfcc
        self.vocab_size = vocab_size

        # path setting
        data_dir = os.path.join(os.getcwd(), 'data', 'mfcc'+str(n_mfcc))
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        # data cache
        wavs_file = os.path.join(data_dir, "wavs.file")
        vocab_file = os.path.join(data_dir,"vocab.file")
        mfcc_tensor = os.path.join(data_dir, "mfcc.tensor")
        label_tensor = os.path.join(data_dir, "label.tensor")

        # data process
        if not(os.path.exists(mfcc_tensor) and os.path.exists(label_tensor)):
            print("reading wav files")
            self.preprocess(wav_path, label_file, wavs_file, vocab_file, mfcc_tensor, label_tensor)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, mfcc_tensor, label_tensor)

        # minibatch
        # self.create_batches()
        # pointer reset
        # self.reset_batch_pointer()

    # convert sentence to index list
    def str2index(self, str_):
        # clean white space
        str_ = ' '.join(str_.split())
        # remove punctuation and make lower case
        str_ = str_.translate(str.maketrans('','',string.punctuation)).lower()

        res = []
        for ch in str_:
            try:
                res.append(byte2index[ch])
            except KeyError:
                # drop OOV
                pass
        return res

    def preprocess(self, wav_path, label_file, wavs_file, vocab_file, mfcc_tensor, label_tensor):

        self.mfcc_tensor = []
        self.label_tensor = []
        # create csv writer
        csv_f = open(train_file, 'w')
        writer = csv.writer(csv_f, delimiter=',')

        # read label-info
        df = pd.read_table(_data_path + 'VCTK-Corpus/speaker-info.txt', usecols=['ID'],
                           index_col=False, delim_whitespace=True)

        # read file IDs
        file_ids = []
        for d in [_data_path + 'VCTK-Corpus/txt/p%d/' % uid for uid in df.ID.values]:
            file_ids.extend([f[-12:-4] for f in sorted(glob.glob(d + '*.txt'))])

        for i, f in enumerate(file_ids):
            # wave file name
            wave_file = _data_path + 'VCTK-Corpus/wav48/%s/' % f[:4] + f + '.wav'
            fn = wave_file.split('/')[-1]
            target_filename = 'data/mfcc/' + fn + '.npy'
            if os.path.exists(target_filename):
                continue
            # print info
            print("VCTK corpus preprocessing (%d / %d) - '%s']" % (i, len(file_ids), wave_file))

            # load wave file
            wave, sr = librosa.load(wave_file, mono=True, sr=None)

            # re-sample ( 48K -> 16K )
            wave = wave[::3]

            # get mfcc feature
            mfcc = librosa.feature.mfcc(wave, sr=16000, n_mfcc = self.n_mfcc)
            mfcc_ts = np.transpose(mfcc, [1, 0])
            self.mfcc_tensor.append(mfcc_ts.tolist())

            # get label index
            label = self.str2index(open(_data_path + 'VCTK-Corpus/txt/%s/' % f[:4] + f + '.txt').read())
            self.label_tensor.append(label)

            # save result ( exclude small mfcc data to prevent ctc loss )
            if len(label) < mfcc.shape[1]:
                # save meta info
                writer.writerow([fn] + label)
                # save mfcc
                np.save(target_filename, mfcc, allow_pickle=False)

        self.wav_max_len = max(len(mfcc) for mfcc in self.mfcc_tensor)
        with open(mfcc_tensor, 'wb') as f:
            cPickle.dump(self.mfcc_tensor, f)

        self.label_max_len = max(len(label) for label in self.label_tensor)
        with open(label_tensor, 'wb') as f:
            cPickle.dump(self.label_tensor, f)

    def load_preprocessed(self, vocab_file, mfcc_tensor, label_tensor):

        with open(mfcc_tensor, 'rb') as f:
            self.mfcc_tensor = cPickle.load(f)
        self.wav_max_len = max(len(mfcc) for mfcc in self.mfcc_tensor)
        print("longest audio wave ", self.wav_max_len)

        with open(label_tensor, 'rb') as f:
            self.label_tensor = cPickle.load(f)
        self.label_max_len = max(len(label) for label in self.label_tensor)
        print("longest sentence: ", self.label_max_len)

    def create_batches(self):
        self.n_batches = len(self.mfcc_tensor) // self.batch_size
        if self.n_batches==0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.mfcc_tensor = self.mfcc_tensor[:self.n_batches*self.batch_size]
        self.label_tensor = self.label_tensor[:self.n_batches*self.batch_size]

        # random shuffle the data
        if len(self.mfcc_tensor) != len(self.label_tensor):
            assert False, "Data length does not match the label length!"

        data_tensor = []
        for i in range(len(self.mfcc_tensor)):
            data_tensor.append([self.mfcc_tensor[i], self.label_tensor[i]])

        random.shuffle(data_tensor)
        self.mfcc_tensor = []
        self.label_tensor = []
        for i in range(len(data_tensor)):
            self.mfcc_tensor.append(data_tensor[i][0])
            self.label_tensor.append(data_tensor[i][1])

        # create batches
        self.x_batches = []
        self.y_batches = []

        for i in range(self.n_batches):
            from_index = i*self.batch_size
            to_index = from_index + self.batch_size
            mfcc_batches = self.mfcc_tensor[from_index:to_index]
            label_batches = self.label_tensor[from_index:to_index]
            # padding with 0
            for mfcc in mfcc_batches:
                while len(mfcc) < self.wav_max_len:
                    mfcc.append([0]*self.n_mfcc)
            for label in label_batches:
                while len(label) < self.label_max_len:
                    label.append(0)

            self.x_batches.append(mfcc_batches)
            self.y_batches.append(label_batches)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0

    def load_one_file(self, file_path):
        # load wave file
        wave, sr = librosa.load(file_path, mono=True, sr=None)
        # re-sample ( 48K -> 16K )
        wave = wave[::3]
        # get mfcc feature
        mfcc = librosa.feature.mfcc(wave, sr=16000, n_mfcc = self.n_mfcc)
        mfcc_ts = np.transpose(mfcc, [1, 0])
        mfcc_ts = np.expand_dims(mfcc_ts, axis = 0)
        mfcc_ts = mfcc_ts.tolist()

        while len(mfcc_ts[0]) < self.wav_max_len:
            mfcc_ts[0].append([0]*self.n_mfcc)

        return mfcc_ts

    def index2str(self, index_list):
        # transform label index to character
        str_ = ''
        for ch in index_list:
            if ch > 0:
                str_ += index2byte[ch]
            elif ch == 0:  # <EOS>
                break
        return str_

if __name__ == '__main__':
    data = SpeechLoader()
