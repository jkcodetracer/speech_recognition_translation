from utils import TextLoader
from model import Seq2SeqNMT
import time
import os
import tensorflow as tf
import numpy as np

SOURCE_LANG = "en"
TARGET_LANG = "fr"

SOURCE_VOCAB_SIZE = 30000
TARGET_VOCAB_SIZE = 30000

ORIGINAL_DATA_DIR = "./data/nmt/"
TRAIN_DIR = "./train/nmt/"

ITERATION = 1000
BATCH_SIZE = 8
HIDDEN_UNITS = 10
N_LAYERS = 2
LR = 0.5

STEP_PER_CKP = 10

# To group data into similar length. make processing more efficient
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

def translate():
    textloader = TextLoader(ORIGINAL_DATA_DIR, SOURCE_VOCAB_SIZE, TARGET_VOCAB_SIZE,
                            SOURCE_LANG, TARGET_LANG, _buckets, batch_size = 1)
    s_vocab_path = os.path.join(ORIGINAL_DATA_DIR,
                                "vocab%d.%s" % (SOURCE_VOCAB_SIZE, SOURCE_LANG))
    t_vocab_path = os.path.join(ORIGINAL_DATA_DIR,
                                "vocab%d.%s" % (TARGET_VOCAB_SIZE, TARGET_LANG))
    s_vocab, _ = textloader.init_vocabulary(s_vocab_path)
    _, t_id2vocab = textloader.init_vocabulary(t_vocab_path)

    # create seq2seq model
    model = Seq2SeqNMT(SOURCE_VOCAB_SIZE, TARGET_VOCAB_SIZE, _buckets, HIDDEN_UNITS, N_LAYERS, batch_size = 1, learning_rate = LR)
    model.build_graph(train = False)

    chpt = tf.train.get_checkpoint_state(TRAIN_DIR)
    if chpt:
        print ("restore model paramters from %s" % chpt.model_checkpoint_path)
        model.restore(chpt.model_checkpoint_path)
    else:
        print ("init a new model.")
        model.init_sess()

    TEST_SENTENCE_PATH = os.path.join(ORIGINAL_DATA_DIR, "test.%s" % SOURCE_LANG)
    f_s = open(TEST_SENTENCE_PATH, 'r')
    step = 0

    for sentence in f_s:
        step += 1
        word_ids = textloader.sentence_to_token_id(tf.compat.as_bytes(sentence), s_vocab)
        # find out the buckets
        bid = len(_buckets) - 1
        for i, bucket in enumerate(_buckets):
            if bucket[0] >= len(word_ids):
                bid = i
                break

        encoder_inputs, decoder_inputs, target_weights = textloader.get_batch({bid: [(word_ids, [])]}, bid)
        _, output_logits = model.predict(encoder_inputs,decoder_inputs, target_weights, bid)
        # greedy decoder
        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        if TextLoader.EOS_ID in outputs:
            outputs = outputs[:outputs.index(TextLoader.EOS_ID)]
        result = [tf.compat.as_str(t_id2vocab[output]) for output in outputs]
        print "source(%d): %s" % (step, sentence)
        print "inference(%d): %s" % (step, result)

if __name__ == "__main__":
    translate()
