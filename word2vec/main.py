'''
Author: John Karasev
runs the word2vec models.
'''

from word2vec import Word2Vec
import numpy as np


def main():
    # with open("/Users/johnkarasev/PycharmProjects/TweetGrouper/word2vec/contexts.json") as fp:
    #     contexts = json.load(fp)
    # with open("/Users/johnkarasev/PycharmProjects/TweetGrouper/word2vec/neighbors.json") as fp:
    #     neighbors = json.load(fp)
    print("Reading dat files")
    npn = np.fromfile("npneighbors.dat", dtype=int)
    print(str(npn.shape[0]))
    npc = np.fromfile("npcontexts.dat", dtype=int)
    print(str(npc.shape[0]))
    print("finished read")
    # train skipgram model
    skipgram = Word2Vec(npn, npc, 35000, 10, 0.001, 64, "sg.ckpt", batch_size=500)
    skipgram.train(5)
    # train cbow model
    cbow = Word2Vec(npc, npn, 35000, 10, 0.001, 64, "sg.ckpt", batch_size=500)
    cbow.train(5)


if __name__ == "__main__":
    main()
