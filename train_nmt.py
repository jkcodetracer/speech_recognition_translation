from utils import TextLoader
from model import Seq2SeqNMT
import time
import os
import tensorflow as tf

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

def train():
    textloader = TextLoader(ORIGINAL_DATA_DIR, SOURCE_VOCAB_SIZE, TARGET_VOCAB_SIZE,
                            SOURCE_LANG, TARGET_LANG, _buckets, BATCH_SIZE)
    source_train, target_train, source_dev, target_dev, _, _ = textloader.prepare_data()
    train_data = textloader.read_data(source_train, target_train)
    dev_data = textloader.read_data(source_dev, target_dev)

    textloader.init_train_bucket(train_data)

    # create seq2seq model
    model = Seq2SeqNMT(SOURCE_VOCAB_SIZE, TARGET_VOCAB_SIZE, _buckets, HIDDEN_UNITS, N_LAYERS, BATCH_SIZE, LR)
    model.build_graph()

    chpt = tf.train.get_checkpoint_state(TRAIN_DIR)
    if chpt:
        print ("restore model paramters from %s" % chpt.model_checkpoint_path)
        model.restore(chpt.model_checkpoint_path)
    else:
        print ("init a new model.")
        model.init_sess()

    current_step = 0
    avg_time = 0.0
    avg_loss = 0.0
    for _ in range(ITERATION):
        bid = textloader.pick_bucket()
        encoder_inputs, decoder_inputs, target_weights = textloader.get_batch(train_data, bid)
        begin_time = time.time()
        gradient_norm, loss = model.train_batch(encoder_inputs, decoder_inputs, target_weights, bid)
        avg_time += (time.time() - begin_time) / STEP_PER_CKP
        avg_loss += loss / STEP_PER_CKP

        current_step += 1

        if current_step % STEP_PER_CKP == 0:
            print ("total step %d learning rate %.4f avg-time %.2f avg-loss: %6f" %
                   (model.global_step.eval(session = model.sess), model.learning_rate.eval(session = model.sess), avg_time, avg_loss))

            chk_path = os.path.join(TRAIN_DIR, "nmt.ckpt." + str(current_step))
            model.save(chk_path)
            avg_time = 0.0
            avg_loss = 0.0

if __name__ == "__main__":
    train()
