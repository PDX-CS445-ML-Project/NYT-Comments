import sys
import os
import threading
import numpy as np
from sklearn.mixture import GaussianMixture


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
    if len(dicts) <= 0:
        return
    # Compute key intersection
    keys = set(dicts[0].keys())
    for i in range(1, len(dicts)):
        keys &= set(dicts[i].keys())
    # Remove any keys from both dicts that don't exist in the key intersection set
    for d in dicts:
        for k in list(d.keys()):
            if k not in keys:
                del d[k]


def average_mapped_dict(mapped_dict: dict, weight_matrix: np.ndarray):
    min_vector_len = 1000000
    max_vector_len = 0
    avg_vector_len = 0
    mapped_dict_len = len(mapped_dict)
    prog = 0
    # Loop through each article
    for article_id, sentence_vector_list in mapped_dict.items():
        progress(prog, mapped_dict_len)
        sentence_vector_list_len = len(sentence_vector_list)
        min_vector_len = min(min_vector_len, sentence_vector_list_len)
        max_vector_len = max(max_vector_len, sentence_vector_list_len)
        avg_vector_len += sentence_vector_list_len
        # Loop through each sentence vector for the article
        for sentence_vector_index in range(sentence_vector_list_len):
            sentence_vector = sentence_vector_list[sentence_vector_index]
            # Verify the sentence vector length
            sentence_vector_len = len(sentence_vector)
            if sentence_vector_len <= 0:
                raise RuntimeError("Sentence vector must have at least one element")
            # Get initial weight vector from sentence vector
            averaged_vector = weight_matrix[sentence_vector[0]].copy()  # type: np.ndarray
            # Sum the remaining weight vectors using the sentence vector for indexing
            for i in range(1, sentence_vector_len):
                averaged_vector += weight_matrix[sentence_vector[i]]
            # Divide by the vector length to compute the average
            averaged_vector /= sentence_vector_len
            # Replace the original sentence vector with the averaged weight vector
            sentence_vector_list[sentence_vector_index] = averaged_vector
        prog += 1
    progress(mapped_dict_len, mapped_dict_len)
    print("Minimum number of weight vectors: " + str(min_vector_len))
    print("Maximum number of weight vectors: " + str(max_vector_len))
    print("Average number of weight vectors: " + str(avg_vector_len / mapped_dict_len))


def average(word2vec_model: str):
    # Load word2vec weight matrix (35000 rows, 10 columns)
    # Each row corresponds to a word in vector form
    print("Loading " + word2vec_model + " matrix...", end="")
    word2vec_matrix = np.load("../resources/" + word2vec_model + "35000x10.npy")
    print(" Done.")

    # Load article-category map
    print("Loading category map...", end="")
    category_map = np.load("../resources/mapped_categories.npy").item()  # type: dict
    print(" Done.")
    print("Initial number of articles with categories: " + str(len(category_map)))
    category_map = clean_dict(category_map)
    print("Cleaned number of articles with categories: " + str(len(category_map)))

    # Load article-comment map
    print("Loading comment map...", end="")
    comment_map = np.load("../resources/mapped_comments.npy").item()  # type: dict
    print(" Done.")
    print("Initial number of articles with comments: " + str(len(comment_map)))
    comment_map = clean_dict(comment_map)
    print("Cleaned number of articles with comments: " + str(len(comment_map)))

    # Intersect dictionaries
    print("Computing intersection between maps...", end="")
    intersect_dicts(category_map, comment_map)
    print(" Done.")
    print("Final number of articles with categories: " + str(len(category_map)))
    print("Final number of articles with comments: " + str(len(comment_map)))

    print()

    # Average article-category map
    print("Averaging category map...")
    average_mapped_dict(category_map, word2vec_matrix)
    print()

    # Average article-comment map
    print("Averaging comment map...")
    average_mapped_dict(comment_map, word2vec_matrix)
    print()

    # Return averaged maps
    return category_map, comment_map


def get_averaged_maps(word2vec_model: str):
    averaged_category_map_file_name = "../resources/averaged_" + word2vec_model + "_categories.npy"
    averaged_comment_map_file_name = "../resources/averaged_" + word2vec_model + "_comments.npy"
    if os.path.exists(averaged_category_map_file_name) and os.path.exists(averaged_comment_map_file_name):
        print("Loading averaged " + word2vec_model + " category map...", end="")
        averaged_category_map = np.load(averaged_category_map_file_name).item()
        print(" Done.")
        print("Loading averaged " + word2vec_model + " comment map...", end="")
        averaged_comment_map = np.load(averaged_comment_map_file_name).item()
        print(" Done.")
    else:
        averaged_category_map, averaged_comment_map = average(word2vec_model=word2vec_model)
        # Save averaged maps to .npy files
        print("Saving averaged " + word2vec_model + " category map...", end="")
        np.save(averaged_category_map_file_name, averaged_category_map)
        print(" Done.")
        print("Saving averaged " + word2vec_model + " comment map...", end="")
        np.save(averaged_comment_map_file_name, averaged_comment_map)
        print(" Done.")
    print()
    # Return averaged maps
    return averaged_category_map, averaged_comment_map


def convert_to_matrix(mapped_dict: dict):
    # Calculate final number of entries from dictionary
    entries = 0
    for vector_list in mapped_dict.values():
        entries += len(vector_list)
    print("Total weight vector entries: " + str(entries))
    # Create new numpy matrix
    matrix = np.empty(shape=(entries, 10))
    # Populate matrices from dictionary
    i = 0
    for vector_list in mapped_dict.values():
        for vector in vector_list:
            matrix[i] = vector
            i += 1
    # Return matrix
    return matrix


def run_sample(word2vec_model: str, sample_size, averaged_category_map, averaged_comment_map):
    # Convert article-comment map into a matrix of averaged comment weight vectors
    print("Converting comment map to matrix...")
    comment_matrix = convert_to_matrix(averaged_comment_map)
    print()

    if sample_size is not None:
        print("Selecting " + str(sample_size) + " samples...", end="")
        sample_indices = np.random.choice(comment_matrix.shape[0], size=sample_size, replace=False)
        samples = comment_matrix[sample_indices, :]
    else:
        sample_size = len(comment_matrix)
        print("Selecting " + str(sample_size) + " samples...", end="")
        samples = comment_matrix
    print(" Done.")
    print()

    # Set up GMM and fit to data
    print("Fitting using GMM...")
    gmm = GaussianMixture(n_components=50, verbose=2, verbose_interval=1)
    gmm.fit(samples)
    print()

    gmm_data = {
        "params": gmm.get_params(),
        "weights": gmm.weights_,
        "means": gmm.means_,
        "covariances": gmm.covariances_,
        "precisions": gmm.precisions_,
        "precisions_cholesky": gmm.precisions_cholesky_
    }

    np.save("../resources/gmm_" + str(sample_size) + "_" + word2vec_model + ".npy", gmm_data)


def run(word2vec_model: str, sample_sizes: list):
    averaged_category_map, averaged_comment_map = get_averaged_maps(word2vec_model)
    threads = []
    for sample_size in sample_sizes:
        t = threading.Thread(target=run_sample, kwargs={
            "word2vec_model": word2vec_model,
            "sample_size": sample_size,
            "averaged_category_map": averaged_category_map,
            "averaged_comment_map": averaged_comment_map
        }, daemon=True)
        threads.append(t)
        t.start()
    # Wait for all threads to finish
    for thread in threads:
        thread.join()


def main():
    # Create hyper-parameters
    word2vec_models = ["skipgram", "cbow"]
    sample_sizes = [100000, 1000000, None]

    # Process each variant in parallel using threads
    threads = []
    for word2vec_model in word2vec_models:
        t = threading.Thread(target=run, kwargs={
            "word2vec_model": word2vec_model,
            "sample_sizes": sample_sizes
        }, daemon=True)
        t.start()
        threads.append(t)
    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    print("Whoo hoo!")


if __name__ == "__main__":
    main()
