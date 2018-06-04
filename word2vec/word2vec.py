import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tflearn
import collections
import json


class Word2Vec(object):

        def __init__(self, data):
                self.data = data

        @staticmethod
        def vocab_to_num(words, vocab_size):
                count = [['UNK', -1]]
                count.extend(collections.Counter(words).most_common(vocab_size - 1))
                word_to_int = {}
                for word, _ in count:
                        word_to_int[word] = len(word_to_int)
                with open("vocab.json", "w") as f:
                        json.dump(word_to_int, f, indent=2)
                return word_to_int



