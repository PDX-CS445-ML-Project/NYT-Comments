import sys
import numpy as np
import json


def progress(count, total, suffix=''):
    bar_len = 60

    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('\r[%s] %s%s ...%s' % (bar, percents, '%', suffix))
    sys.stdout.flush()
    if count == total:
        print("")


def clean_dict(d: dict):
    for k in d:
        d[k] = clean_list(d[k])
    return {k: v for k, v in d.items() if v is not None}


def clean_list(l):
    if not isinstance(l, list):
        return l
    for i in range(len(l)):
        l[i] = clean_list(l[i])
    new_list = [v for v in l if v is not None]
    return new_list if len(new_list) > 0 else None


def intersect_dicts(*dicts: dict):
    # TODO
    pass


def average_mapped_dict(mapped_dict: dict, weight_matrix: np.ndarray):
    overflow = dict()
    # np.seterr(all='raise')
    mapped_dict_len = len(mapped_dict)
    prog = 0
    # Loop through each article
    for article_id, sentence_vector_list in mapped_dict.items():
        progress(prog, mapped_dict_len)
        # Loop through each sentence vector for the article
        for sentence_vector_index in range(len(sentence_vector_list)):
            sentence_vector = sentence_vector_list[sentence_vector_index]
            # Verify the sentence vector length
            sentence_vector_len = len(sentence_vector)
            if sentence_vector_len <= 0:
                raise RuntimeError("Sentence vector must have at least one element")
            # Get initial weight vector from sentence vector
            averaged_vector = weight_matrix[sentence_vector[0]]  # type: np.ndarray
            try:
                # Sum the remaining weight vectors using the sentence vector for indexing
                for i in range(1, sentence_vector_len):
                    averaged_vector += weight_matrix[sentence_vector[i]]
            except FloatingPointError:
                print("Floating point error occured during vector summation: " + str(article_id), file=sys.stderr)
                for i in range(1, sentence_vector_len):
                    print(str(sentence_vector[i]) + " => " + str(weight_matrix[sentence_vector[i]]), file=sys.stderr)
                    overflow[sentence_vector[i]] = True
            # Divide by the vector length to compute the average
            averaged_vector /= sentence_vector_len
            # Replace the original sentence vector with the averaged weight vector
            sentence_vector_list[sentence_vector_index] = averaged_vector
            # Debugging: Print overflow row indices
            if len(overflow) > 0:
                print(sorted(overflow.keys()))
        prog += 1
    progress(mapped_dict_len, mapped_dict_len)


def main():
    # Load skipgram weight matrix (35000 rows, 10 columns)
    # Each row corresponds to a word in vector form
    print("Loading skipgram matrix...", end="")
    skipgram_matrix = np.load("../resources/skipgram35000x10.npy")
    for k in [0, 3, 5, 6, 7, 32, 35, 39, 53, 55, 60, 61, 68, 75, 96, 128, 130, 138, 152, 186, 195, 215, 218, 224, 231, 233, 244, 257, 269, 276, 302, 307, 317, 339, 371, 385, 397, 455, 509, 572, 575, 585, 631, 678, 692, 751, 760, 826, 866, 940, 960, 990, 1019, 1053, 1114, 1122, 1144, 1182, 1203, 1380, 1418, 1569, 1628, 1639, 1973, 2024, 2043, 2099, 2176, 2222, 2249, 2491, 2546, 2552, 2584, 2700, 2709, 2825, 2837, 2954, 3004, 3208, 3349, 3541, 4000, 4536, 4638, 5663, 5757, 6045, 6494, 6499, 7659, 7868, 8208, 8354, 8561, 8778, 8863, 9008, 13237, 26916, 32592]:
        print(skipgram_matrix[k])
    print(" Done.")

    # Load vocabulary map (word -> row index into skipgram matrix)
    print("Loading vocabulary map...", end="")
    with open("../resources/vocab.json") as fp:
        vocab = json.load(fp)
    print(" Done.")

    # Load article-category map
    print("Loading category map...", end="")
    category_map = np.load("../resources/mapped_categories.npy").item()  # type: dict
    print(" Done.")
    print("Initial number of articles with categories: " + str(len(category_map)))
    category_map = clean_dict(category_map)
    print("Final number of articles with categories: " + str(len(category_map)))

    # Load article-comment map
    print("Loading comment map...", end="")
    comment_map = np.load("../resources/mapped_comments.npy").item()  # type: dict
    print(" Done.")
    print("Initial number of articles with comments: " + str(len(comment_map)))
    comment_map = clean_dict(comment_map)
    print("Final number of articles with comments: " + str(len(comment_map)))

    # Intersect dictionaries
    intersect_dicts(category_map, comment_map)

    # Average article-category map
    print("Averaging category map...")
    average_mapped_dict(category_map, skipgram_matrix)

    # Average article-comment map
    print("Averaging comment map...")
    average_mapped_dict(comment_map, skipgram_matrix)

    # Save averaged maps to .npy files
    np.save("../resources/averaged_categories.npy", category_map)
    np.save("../resources/averaged_comments.npy", comment_map)

    print("Whoo hoo!")
    pass


if __name__ == "__main__":
    main()
