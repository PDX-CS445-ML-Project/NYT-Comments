from word2vec import Word2Vec
import itertools
import json
import numpy as np


vocab = None


def export_vocab(comments, categories, vocab_size, export=True):
    words = []
    for key in comments:
        words.extend(list(itertools.chain.from_iterable(comments[key])))
    for key in categories:
        words.extend(list(itertools.chain.from_iterable(categories[key])))
    vocab = Word2Vec.vocab_to_num(words, vocab_size)
    if export:
        with open("vocab.json", "w") as f:
            json.dump(vocab, f, indent=2)
    return comments, categories


def map_func(x):
    if x not in vocab.values():
        return 0
    else:
        return vocab[x]


def map_sentences(comments, categories, export=True):
    global vocab  # global for map function
    with open("vocab.json") as f:
        vocab = json.load(f)
    for key in comments:
        comments[key] = [list(map(map_func, comment) for comment in comments[key])]
    for key in categories:
        categories[key] = [list(map(map_func, category) for category in comments[key])]
    if export:
        with open("mapped_comments.json", "w") as f:
            json.dump(comments, f, indent=2)
        with open("mapped_categories.json", "w") as f:
            json.dump(categories, f, indent=2)
    return comments, categories


def create_trainset(window, export=True, skipgram=True):
    with open("mapped_comments.json") as f:
        comments = json.load(f)
    with open("mapped_categories.json") as f:
        categories = json.load(f)
    sentences = []
    for key in comments:
        sentences.extend(comments[key])
    for key in categories:
        sentences.extend((categories[key]))
    sentences = list(filter(lambda x: x, sentences))
    sentences = np.array(sentences)
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



if __name__ == "__main__":
    with open("../dataset/comments.json") as fp:
        comments = json.load(fp)
    with open("../dataset/categories.json") as fp:
        categories = json.load(fp)
    export_vocab(comments, categories, 35000)

