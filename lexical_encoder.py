# -*- coding: utf-8 -*-
#
#
# This is a Lexical encoder based on subword entropies
#
# Created by Alex Molina
# July 2020

from math import log
import tensorflow as tf
import tensorflow_datasets as tfds
import dill as pickle
from utils import text_generator


class LexicalEncoder:
    def __init__(self):
        pass

    @classmethod
    def load_from_file(cls, filename):
        #print('Loading LexicalEncoder instance...')
        with open(filename, 'rb') as f:
            lexcoder = pickle.load(f)
            return lexcoder

    @property
    def bpencoder(self):
        #print("Getting subword encoder...")
        return self._bpencoder

    @property
    def probs_subwords(self):
        #print("Getting subword probability dist...")
        return self._probs_subwords

    @property
    def ents_subwords(self):
        #print("Getting subword entropy dist...")
        return self._subwords_entropies

    @bpencoder.setter
    def bpencoder(self, bpencoder_instance):
        #print("Setting subword encoder...")
        if not isinstance(bpencoder_instance, tfds.features.text.SubwordTextEncoder):
            raise ValueError("bpencoder must be an instance of SubwordTextEncoder")
        self._bpencoder = bpencoder_instance

    def encode(self, text_string, padding=False):
        encoded_str = self._bpencoder.encode(text_string)
        ents = []
        if not padding:
            for s in encoded_str:
                try:
                    H = self._subwords_entropies[s]
                except:
                    H = 0.0
                ents.append(H)
            return ents
        else:
            # fill in with zeros the not used subwords
            for i in range(self.bpencoder.vocab_size-1):
                if i in encoded_str:
                    try:
                        H = self._subwords_entropies[i]
                    except:
                        H = 0.0
                else:
                    H = 0.0
                ents.append(H)
            return ents

    def save_to_file(self, filename):
        #print('Saving LexicalEncoder instance...')
        with open(filename+'.lexenc', 'wb') as f:
            pickle.dump(self, f)

    def set_subwords_distributions(self, corpus_generator):
        """Computes the subword prob dist"""
        #print("Setting subword probability and entropy distributions...")
        if not self._bpencoder:
            raise ValueError("bpencoder must be set before")
        # Frequencies dist
        encoded_subword_counts = self._subword_counts(corpus_generator)
        # subword probability distribution estimation
        V = float(sum(encoded_subword_counts.values()))
        self._probs_subwords = {s: (float(f_s)/V) for s, f_s in encoded_subword_counts.items()}
        """Buils a entropy dictionary for each subword"""
        self._subwords_entropies = {s: -(p * log(p, 2)) for s, p in self._probs_subwords.items()}

    def _subword_counts(self, data):
        """Build a dictionary with subword counts from a SubwordTextEncoder"""
        subword_counts = dict()
        for d in data:
            for i in self._bpencoder.encode(d):
                if i in subword_counts.keys():
                    subword_counts[i] += 1
                else:
                    subword_counts[i] = 1
        return subword_counts


if __name__ == '__main__':
    ################ Lexical encoding with subword entropies ################
    lang = 'es'
    vocab_size = 528
    subword_len = 5
    data_root_dir = 'data/'
    filename = '{}_vocab_{}_len_{}'.format(lang, vocab_size, subword_len)
    # create a subword encoder
    print('Building SubwordTextEncoder...')
    subword_encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
             corpus_generator=text_generator(data_root_dir),
             target_vocab_size=vocab_size,
             max_subword_length=subword_len)
    # store a vocab file
    subword_encoder.save_to_file(filename)
    # create instance
    lexcoder = LexicalEncoder()
    lexcoder.bpencoder = subword_encoder
    print(lexcoder.bpencoder.subwords[:10])
    lexcoder.set_subwords_distributions(text_generator(data_root_dir))
    # store a LexicalEncoder object instance including its distributions
    lexcoder.save_to_file(filename)
    other_lexcoder = LexicalEncoder.load_from_file(filename+'.lexenc')
    print('Loaded lexcoder', type(other_lexcoder))
    # Encoding a text
    text = 'este es el texto de prueba'
    print('text:\n', text)
    lexencoded_text = other_lexcoder.encode(text, padding=True)
    print('LexEncoded text with padding:\n', lexencoded_text)
    lexencoded_text = other_lexcoder.encode(text, padding=False)
    print('LexEncoded text without padding:\n', lexencoded_text)
