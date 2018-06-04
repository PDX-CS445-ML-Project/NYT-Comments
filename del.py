import json
import os
import sys

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
    with open("word2vec/vocab.json") as f:
        vocab = json.load(f)
    print("mapping words to ints")
    clen = len(comments.keys())
    for key, index in zip(comments, range(clen)):
        if index == 1:
            break
        progress(index, clen, suffix="converting comments")
        comments[key] = [list(map(map_func, comment)) for comment in comments[key]]
    if export:
        with open("mapped_comments.json", "w") as f:
            json.dump(comments, f, indent=2)
    return comments, categories


if __name__ == "__main__":
    with open("NYT/commentsMarch2018.json") as fp:
        comments = json.load(fp)
    print("Json loaded")
    # vocab = export_vocab(comments, categories, 35000)
    comments, categories = map_sentences(comments, None)