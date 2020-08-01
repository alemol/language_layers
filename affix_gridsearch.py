# -*- coding: utf-8 -*-
#
#
# A grid search to optimize parameter using linguistic knowledge
#
# Created by Alex Molina
# July 2020

import pandas as pd
import tensorflow_datasets as tfds
from lexical_encoder import LexicalEncoder
from utils import (DATA_DIR, text_generator, get_afixes, common_affix_coverage)

if __name__ == '__main__':

    lang = 'es'
    affixes = get_afixes(lang)
    vocab_sizes = [i for i in range(300,500,100)]
    subword_lengths = [3,5,7,9,11]
    for subword_len in subword_lengths:
        for vocab_size in vocab_sizes:
            # Subword encoder build
            encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
             corpus_generator=text_generator(DATA_DIR),
             target_vocab_size=vocab_size,
             max_subword_length=subword_len)
            # asses the common affix_coverage
            #print(encoder.subwords)
            affix_coverage = common_affix_coverage(affixes, encoder.subwords)
            # subword probability distribution
                # create instance
            lexcoder = LexicalEncoder()
            lexcoder.bpencoder = encoder
            del(encoder)
            lexcoder.set_subwords_distributions(text_generator(DATA_DIR))
            entdist = pd.Series(lexcoder.ents_subwords)
            del(lexcoder)
            # create grid entry
            outcome = '{},{},{},{},{},{},{},{},{},{}'.format(lang,vocab_size,subword_len,affix_coverage,entdist.kurtosis(),entdist.skew(),entdist.min(),entdist.mean(),entdist.std(),entdist.max())
            print(outcome)
            del(entdist)
