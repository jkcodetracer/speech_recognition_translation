import os
import re
import sys
import random
import numpy as np

from tensorflow.python.platform import gfile
import tensorflow as tf

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")


class TextLoader():
    EOS_ID = 2

    def __init__(self, data_dir, source_vocab_size, target_vocab_size, source_lang, target_lang, buckets, batch_size):
        self.data_dir = data_dir
        self.source_vocabulary_size = source_vocab_size
        self.target_vocabulary_size = target_vocab_size
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.buckets = buckets
        self.batch_size = batch_size

    def init_train_bucket(self, data_set):
        train_bucket_sizes = [len(data_set[b]) for b in xrange(len(self.buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        self.train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

    def pick_bucket(self):
        # Choose a bucket according to data distribution. We pick a random number
        # in [0, 1] and use the corresponding interval in train_buckets_scale.
        random_number_01 = np.random.random_sample()
        bucket_id = min([i for i in xrange(len(self.train_buckets_scale))
                         if self.train_buckets_scale[i] > random_number_01])

        return bucket_id

    def basic_tokenizer(self, sentence):
        words = []
        for space_separated_fragment in sentence.strip().split():
            words.extend(_WORD_SPLIT.split(space_separated_fragment))
        return [w for w in words if w]

    def create_vocabulary(self, vocabulary_path, data_path, max_vocabulary_size,
                          tokenizer = None, normalize_digits = True):
        '''
            convert data in (data_path) to a vocabulary and write it into vocabulary_path
        :param vocabulary_path:
        :param data_path:
        :param max_vocabulary_size:
        :param tokenizer:
        :param normalize_digits:
        :return:
        '''
        if not gfile.Exists(vocabulary_path):
            print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
            vocab = {}
            with gfile.GFile(data_path, mode="rb") as f:
                counter = 0
                for line in f:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  processing line %d" % counter)
                    line = tf.compat.as_bytes(line)
                    tokens = tokenizer(line) if tokenizer else self.basic_tokenizer(line)
                    for w in tokens:
                        word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
                        if word in vocab:
                            vocab[word] += 1
                        else:
                            vocab[word] = 1
                vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
                if len(vocab_list) > max_vocabulary_size:
                    vocab_list = vocab_list[:max_vocabulary_size]
                with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
                    for w in vocab_list:
                        vocab_file.write(w + b"\n")

    def init_vocabulary(self, vocabulary_path):
        '''
            Load vocabulary from (vocabulary_path)
        :param vocabulary_path:
        :return:
        '''
        if gfile.Exists(vocabulary_path):
            rev_vocab = []
            with gfile.GFile(vocabulary_path, mode="rb") as f:
                rev_vocab.extend(f.readlines())
            rev_vocab = [line.strip() for line in rev_vocab]
            vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
            return vocab, rev_vocab
        else:
            raise ValueError("Vocabulary file %s not found.", vocabulary_path)

    def sentence_to_token_id(self, sentence, vocabulary, tokenizer = None, normalize_digits = True):
        '''
            convert sentence to token/word id
        :param sentence:
        :param vocabulary:
        :param tokenizer:
        :param normalize_digits:
        :return:
        '''
        if tokenizer:
            words = tokenizer(sentence)
        else:
            words = self.basic_tokenizer(sentence)

        if not normalize_digits:
            return [vocabulary.get(w, UNK_ID) for w in words]
            # Normalize digits by 0 before looking words up in the vocabulary.
        return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]

    def data_to_token_id(self, data_path, target_path, vocabulary_path, tokenizer = None, normalize_digits = True):
        '''
            convert data to token/word id
        :param data_path:
        :param target_path:
        :param vocabulary_path:
        :param tokenizer:
        :param normalize_digits:
        :return:
        '''
        if not gfile.Exists(target_path):
            print("Tokenizing data in %s" % data_path)
            vocab, _ = self.initialize_vocabulary(vocabulary_path)
            with gfile.GFile(data_path, mode="rb") as data_file:
                with gfile.GFile(target_path, mode="w") as tokens_file:
                    counter = 0
                    for line in data_file:
                        counter += 1
                        if counter % 100000 == 0:
                            print("  tokenizing line %d" % counter)
                        token_ids = self.sentence_to_token_ids(tf.compat.as_bytes(line), vocab,
                                                          tokenizer, normalize_digits)
                        tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

    def parse_files_to_lists(self, data_path, lang, xml):
        if not xml:
            with gfile.GFile(data_path + lang, mode="r") as f:
                texts = f.readlines()
                texts = (t for t in texts if "</" not in t)
        else:
            import xml.etree.ElementTree as ET
            filename = data_path + lang + '.xml'
            tree = ET.parse(filename)
            texts = (seg.text for seg in tree.iter('seg'))
        return texts

    def prepare_data(self):
        """Get TED talk data from data_dir, create vocabularies and tokenize data.
        Args:
          data_dir: directory in which the data sets will be stored.
          ja_vocabulary_size: size of the Japanese vocabulary to create and use.
          en_vocabulary_size: size of the English vocabulary to create and use.
          tokenizer: a function to use to tokenize each data sentence;
            if None, basic_tokenizer will be used.
        Returns:
          A tuple of 6 elements:
            (1) path to the token-ids for Japanese training data-set,
            (2) path to the token-ids for English training data-set,
            (3) path to the token-ids for Japanese development data-set,
            (4) path to the token-ids for English development data-set,
            (5) path to the Japanese vocabulary file,
            (6) path to the English vocabulary file.
        """

        print ("Convert original data in %s " % self.data_dir)
        print ("source language: %s " % self.source_lang)
        print ("target language: %s " % self.target_lang)

        _data_dir = self.data_dir
        _train_path = os.path.join(_data_dir, 'train.')
        _dev_path = os.path.join(_data_dir, 'dev.')

        if not os.path.isfile(os.path.join(self.data_dir, "train." + self.source_lang)):
            data_dir = os.path.join(self.data_dir, "%s-%s/" % (self.source_lang, self.target_lang))

            # Get nmt data to the specified directory.
            train_path = os.path.join(data_dir, "train.tags.%s-%s." % (self.source_lang, self.target_lang))
            dev_path = os.path.join(data_dir, "IWSLT15.TED.dev2010.%s-%s." % (self.source_lang, self.target_lang))

            # Parse xml files into lists of texts.
            s_texts_train = self.parse_files_to_lists(train_path, self.source_lang, False)
            t_texts_train = self.parse_files_to_lists(train_path, self.target_lang, False)
            s_texts_dev = self.parse_files_to_lists(dev_path, self.source_lang, True)
            t_texts_dev = self.parse_files_to_lists(dev_path, self.target_lang, True)

            # Write out training set and dev sets.
            with gfile.GFile(_train_path + self.source_lang, mode="w") as f:
                for line in s_texts_train:
                    f.write(line)
            with gfile.GFile(_train_path + self.target_lang, mode="w") as f:
                for line in t_texts_train:
                    f.write(line)
            with gfile.GFile(_dev_path + self.source_lang, mode="w") as f:
                for line in s_texts_dev:
                    f.write(line + "\n")
            with gfile.GFile(_dev_path + self.target_lang, mode="w") as f:
                for line in t_texts_dev:
                    f.write(line + "\n")

        # Create vocabularies of the appropriate sizes.
        s_vocab_path = os.path.join(_data_dir, "vocab%d.%s" % (self.source_vocabulary_size, self.source_lang))
        t_vocab_path = os.path.join(_data_dir, "vocab%d.%s" % (self.target_vocabulary_size, self.target_lang))
        self.create_vocabulary(s_vocab_path, _train_path + self.source_lang, self.source_vocabulary_size)
        self.create_vocabulary(t_vocab_path, _train_path + self.target_lang, self.target_vocabulary_size)

        # Create token ids for the training data.
        s_train_ids_path = _train_path + ("ids%d.%s" % (self.source_vocabulary_size, self.source_lang))
        t_train_ids_path = _train_path + ("ids%d.%s" % (self.target_vocabulary_size, self.target_lang))
        self.data_to_token_id(_train_path + self.source_lang, s_train_ids_path, s_vocab_path)
        self.data_to_token_id(_train_path + self.target_lang, t_train_ids_path, t_vocab_path)

        # Create token ids for the development data.
        s_dev_ids_path = _dev_path + ("ids%d.%s" % (self.source_vocabulary_size, self.source_lang))
        t_dev_ids_path = _dev_path + ("ids%d.%s" % (self.target_vocabulary_size, self.target_lang))
        self.data_to_token_id(_dev_path + self.source_lang, s_dev_ids_path, s_vocab_path)
        self.data_to_token_id(_dev_path + self.target_lang, t_dev_ids_path, t_vocab_path)

        return (s_train_ids_path, t_train_ids_path,
                s_dev_ids_path, t_dev_ids_path,
                s_vocab_path, t_vocab_path)

    def read_data(self, source_path, target_path, max_size=None):
        """Read data from source and target files and put into buckets.
        Args:
          source_path: path to the files with token-ids for the source language.
          target_path: path to the file with token-ids for the target language;
            it must be aligned with the source file: n-th line contains the desired
            output for n-th line from the source_path.
          max_size: maximum number of lines to read, all other will be ignored;
            if 0 or None, data files will be read completely (no limit).
        Returns:
          data_set: a list of length len(_buckets); data_set[n] contains a list of
            (source, target) pairs read from the provided data files that fit
            into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
            len(target) < _buckets[n][1]; source and target are lists of token-ids.
        """
        data_set = [[] for _ in self.buckets]
        with tf.gfile.GFile(source_path, mode="r") as source_file:
            with tf.gfile.GFile(target_path, mode="r") as target_file:
                source, target = source_file.readline(), target_file.readline()
                counter = 0
                while source and target and (not max_size or counter < max_size):
                    counter += 1
                    if counter % 100000 == 0:
                        print("  reading data line %d" % counter)
                        sys.stdout.flush()
                    source_ids = [int(x) for x in source.split()]
                    target_ids = [int(x) for x in target.split()]
                    target_ids.append(EOS_ID)
                    # put one pair into a bucket
                    for bucket_id, (source_size, target_size) in enumerate(self.buckets):
                        if len(source_ids) < source_size and len(target_ids) < target_size:
                            data_set[bucket_id].append([source_ids, target_ids])
                            break
                    source, target = source_file.readline(), target_file.readline()
        return data_set

    def get_batch(self, data, bucket_id):
        """Get a random batch of data from the specified bucket, prepare for step.
        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.
        Args:
          data: a tuple of size len(self.buckets) in which each element contains
            lists of pairs of input and output data that we use to create a batch.
          bucket_id: integer, which bucket to get the batch for.
        Returns:
          The triple (encoder_inputs, decoder_inputs, target_weights) for
          the constructed batch that has the proper format to call step(...) later.
        """
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in xrange(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])

            # Encoder inputs are padded and then reversed.
            # ?? why?
            encoder_pad = [PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([GO_ID] + decoder_input +
                                  [PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(self.batch_size)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(self.batch_size)], dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in xrange(self.batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights
