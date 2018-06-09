from word2vec import Word2Vec
import itertools
import json
import numpy as np
import sys

vocab = None


def export_vocab(comments, categories, vocab_size, export=True):
    words = []
    for key in comments:
        words.extend(list(itertools.chain.from_iterable(comments[key])))
    for key in categories:
        words.extend(list(itertools.chain.from_iterable(categories[key])))
    vocab = Word2Vec.vocab_to_num(words, vocab_size)
    if export:
        with open("../resources/vocab.json", "w") as f:
            json.dump(vocab, f, indent=2)
    return vocab


def map_func(x):
    if x not in vocab.keys():
        return 0
    else:
        return vocab[x]

    # prints the progress of a process
    # Vladimir Ignatyev  https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console


def progress(count, total, suffix=''):
    bar_len = 60

    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('\r[%s] %s%s ...%s' % (bar, percents, '%', suffix))
    sys.stdout.flush()
    if count == total:
        print("")


def map_sentences(comments, categories, export=True):
    global vocab  # global for map function
    with open("../resources/vocab.json") as f:
        vocab = json.load(f)
    print("mapping words to ints")
    '''
    clen = len(comments.keys())
    for key, index in zip(comments, range(clen)):
        progress(index, clen, suffix="converting comments")
        comments[key] = [list(map(map_func, comment)) for comment in comments[key]]
    '''
    ctlen = len(categories.keys())
    for key, index in zip(categories, range(ctlen)):
        progress(index, ctlen, suffix="converting categories")
        categories[key] = [list(map(map_func, category)) for category in categories[key]]
    if export:
        with open("mapped_comments.json", "w") as f:
            json.dump(comments, f, indent=2)
        with open("mapped_categories.json", "w") as f:
            json.dump(categories, f, indent=2)
    return comments, categories


def create_trainset(window, export=True):
    with open("mapped_comments.json") as f:
        comments = json.load(f)
    sentences = []
    for key, index in zip(comments, range(len(comments))):
        progress(index, len(comments), "combining sentences")
        sentences.extend(comments[key])

    sentences = list(filter(lambda x: x, sentences))
    print("finished")
    sentences = np.array(sentences)
    contexts, neighbors = Word2Vec.create_dataset(sentences, window)
    if export:
        npc = np.array(contexts)
        npn = np.array(neighbors)
        npc.tofile('npcontexts.dat')
        npn.tofile('npneighbors.dat')


if __name__ == "__main__":
    
    #with open("../dataset/comments.json") as fp:
        #comments = json.load(fp)
    
    #with open("../dataset/categories.json") as fp:
        #categories = json.load(fp)

    print("Json loaded")
    # vocab = export_vocab(comments, categories, 35000)
    #comments, categories = map_sentences(None, categories)
    create_trainset(4)
