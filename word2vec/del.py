'''
Author: John Karasev
'''

import json
import numpy as np

# save json files into binary format for faster reading
if __name__ == "__main__":
    with open("mapped_categories.json") as fp:
        categories = json.load(fp)
    with open("mapped_comments.json") as fp:
        comments = json.load(fp)
    np.save('mapped_comments', comments)
    np.save('mapped_categories', categories)
    # use np.load('my_file.npy').item()
