# -*- coding: utf-8 -*-
#
# utilities
#
# Created by Alex Molina
# July 2020

import os
import re
import pandas as pd
import gensim
import nltk

DATA_DIR='./data'
ES_AFFIXES_FILE='./resources/es_affixes_utf8.txt'

def text_generator(dir_name, tok_and_tag=False):
    """text generator reads from csv files yielding str"""
    for root, directory, files in os.walk(dir_name, topdown=True):
        for file in files:
            if file.endswith('.csv'):
                full_path = os.path.join(root, file)
                data_table = pd.read_csv(full_path,
                    encoding='UTF-8',
                    engine='python')
                data = pd.DataFrame(data_table, columns = ['text'])
                for  i, t in enumerate(data.text):
                    try:
                        if tok_and_tag:
                            tokens = preprocess(t, as_tokens=True)
                            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
                        else:
                            yield preprocess(t, as_tokens=False)
                    except:
                        continue

def get_afixes(lang):
    affixes_f = ES_AFFIXES_FILE if lang == 'es' else 'en'
    with open(affixes_f) as f:
        affixes = [l.replace('\n', '') for l in f.readlines()]
        return affixes

def common_affix_coverage(ref_set, eval_set):
    overlap = set(ref_set) & set(eval_set)
    #print('Overlap', overlap)
    return float(len(overlap)) / len(ref_set)

def preprocess(text, as_tokens=False):
    # Removing the @
    text = re.sub(r"@[A-Za-z0-9]+", ' ', text)
    # Removing the URL links
    text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    # Remove numbers
    text = re.sub(r'\b(\d+)\b', '', text)
    toks = nltk.word_tokenize(text.lower())
    if as_tokens:
        return toks
    else:
        return ' '.join(toks)

if __name__ == '__main__':
    s = 'ça va? ¿No lo sé,   amigos @mex 2 o 3000 en http://hola.com '
    print(preprocess(s))
