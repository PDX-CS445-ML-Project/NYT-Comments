import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tflearn
import collections
import json
import os


class Word2Vec(object):

    def __init__(self, x, y, vocab_size, embedd_size, learning_rate, nce_sample_size, skipgram=True):
        self.input = x,
        self.target = y,
        self.optimizer, self.loss, self.x, self.y, self.sess = Word2Vec.create_nn(vocab_size, embedd_size,
                                                                                  nce_sample_size, skipgram)

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

    @staticmethod
    def create_nn(vocab_size, embedding_size, learning_rate, nce_sample_size, skipgram=True):
        if skipgram:
            x = tf.placeholder(tf.int32, shape=[None, ], name="contexts")
            y = tf.placeholder(tf.int32, shape=[None, ], name="neighbors")
        else:
            x = tf.placeholder(tf.int32, shape=[None, ], name="neighbors")
            y = tf.placeholder(tf.int32, shape=[None, ], name="contexts")
        Embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="word_embeddings")
        nce_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=tf.sqrt(1 / embedding_size)),
                                  name="nce_weights")
        nce_biases = tf.Variable(tf.zeros([vocab_size]), name="nce_biases")
        word_embed = tf.nn.embedding_lookup(Embedding, x, name="word_embed_lookup")
        train_labels = tf.reshape(y, [tf.shape(y)[0], 1])
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                             biases=nce_biases,
                                             labels=train_labels,
                                             inputs=word_embed,
                                             num_sampled=nce_sample_size,
                                             num_classes=vocab_size,
                                             num_true=1))
        optimizer = tf.contrib.layers.optimize_loss(loss,
                                                    tf.train.get_global_step(),
                                                    learning_rate,
                                                    "Adam",
                                                    clip_gradients=5.0,
                                                    name="optimizer")
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        return optimizer, loss, x, y, sess

    def train(self, batch_size, epochs):
        x_train, x_test, y_train, y_test = train_test_split(self.input, self.target)
        num_batches = len(x_train) // batch_size
        saver = tf.train.Saver()

        for epoch in range(epochs):
            for i in range(num_batches):
                if i != range(num_batches - 1):
                    x_batch = x_train[i * batch_size:i * batch_size + batch_size]
                    y_batch = y_train[i * batch_size:i * batch_size + batch_size]
                else:
                    x_batch = x_train[i * batch_size:]
                    y_batch = y_train[i * batch_size:]

                _, l = self.sess.run([self.optimizer, self.loss], feed_dict={self.x: x_batch, self.y: y_batch})
                if i > 0 and i % 1000 == 0:
                    print("STEP", i, "of", num_batches, "LOSS:", l)
        save_path = saver.save(self.sess, os.path.join("tf_log", "word2vec_model.ckpt"))
