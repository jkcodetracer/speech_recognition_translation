
from utils import SpeechLoader
from model import WaveNet
import tensorflow as tf

TRAIN_DIR = "./train/asr/"

def train():
    batch_size = 32
    n_epoch = 100
    n_mfcc = 60

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

    speech_loader.create_batches()
    model.train_val(speech_loader.mfcc_tensor, speech_loader.label_tensor, ckpt_dir = TRAIN_DIR, n_epoch = n_epoch,
                    val_rate = 0.15)

if __name__ == '__main__':
    train()
