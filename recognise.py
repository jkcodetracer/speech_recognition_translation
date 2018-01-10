# -*- coding:utf-8 -*-

from __future__ import print_function
from model import WaveNet
from utils import SpeechLoader

import tensorflow as tf  # 1.0.0
import numpy as np
import librosa
import os

TRAIN_DIR = "./train/asr/"
TEXT_DIR = "./data/VCTK-Corpus/wav48/p231/"

def speech_to_text():
    n_mfcc = 60
    batch_size = 1
    n_epoch = 100

    speech_loader = SpeechLoader(batch_size = batch_size, n_mfcc = n_mfcc)
    n_out = speech_loader.vocab_size

    model = WaveNet(n_out, batch_size = batch_size, n_mfcc = n_mfcc)
    model.build_graph()

    chpt = tf.train.get_checkpoint_state(TRAIN_DIR)
    if chpt:
        print ("restore model paramters from %s" % chpt.model_checkpoint_path)
        model.restore(chpt.model_checkpoint_path)
    else:
        print ("init a new model.")
        model.init_sess()


    file_names = os.listdir(TEXT_DIR)
    file_list = [os.path.join(TEXT_DIR, file_name) for file_name in file_names]

    step = 0
    for file in file_list:
        step += 1
        mfcc_features = speech_loader.load_one_file(file)

        output = model.predict(mfcc_features)
        # transfer to word
        words = speech_loader.index2str(output[0])
        print("Input(%d): %s" % (step, file))
        print("Output(%d): %s" % (step, words))

if __name__ == '__main__':
    speech_to_text()

