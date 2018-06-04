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


        @staticmethod
        def create_dataset(sentences, window):
                neighbor_words = []
                context_words = []
                for sentence in range(sentences.shape[0]):
                        contexts = sentences[sentence][window:-window]
                        for index in range(len(contexts)):
                                context = contexts[index]
                                neighbors = np.array([])
                                prev_words = sentences[sentence][index: window + index]
                                next_words = sentences[sentence][index + 1:2 * window + index + 1]
                                neighbors = np.append(neighbors, [prev_words, next_words]).flatten().tolist()
                                for i in range(window * 2):
                                        context_words.append(context)
                                        neighbor_words.append(neighbors[i])
                return context_words, neighbor_words

