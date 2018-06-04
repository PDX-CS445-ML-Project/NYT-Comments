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


def create_trainset(window, export=True):
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
    contexts, neighbors = Word2Vec.create_dataset(sentences, window)
    if export:
        with open("contexts.json", "w") as fp:
            json.dump(contexts, fp, indent=2)
        with open("neighbors.json", "w") as fp:
            json.dump(neighbors, fp, indent=2)


if __name__ == "__main__":
    with open("../dataset/comments.json") as fp:
        comments = json.load(fp)
    with open("../dataset/categories.json") as fp:
        categories = json.load(fp)
    export_vocab(comments, categories, 35000)

