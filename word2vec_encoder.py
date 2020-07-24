# -*- coding: utf-8 -*-
#
# Creates a Word2vec model for mexican spanish
#
# Created by Alex Molina
# July 2020

import gensim
from utils import text_generator


if __name__ == '__main__':
    data_root_dir = 'data/'
    train_corpus = list(s for s in text_generator(data_root_dir, tok_and_tag=True))
    model = gensim.models.doc2vec.Doc2Vec(vector_size=40, min_count=4, epochs=4)
    print('Building vocabulary...')
    model.build_vocab(train_corpus)
    print('model.wv.vocab size', len(model.wv.vocab))
    print('Training model doc2vec model...')
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    # Using the model to generate a doc embedding
    text = 'este es el texto de prueba'
    print('text:\n', text)
    unseen_tokens = gensim.utils.simple_preprocess(text)
    vector = model.infer_vector(unseen_tokens)
    print('Semantic encoded text:\n', vector)
