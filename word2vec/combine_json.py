import os
import json
import ast
import re


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9 ]", "", string)
    return string.strip().lower()


def combine_files(d):
    comments = {}
    categories = {}
    files = os.listdir(d)
    for file in files:
        if "comments" in file:
            with open(os.path.join(d, file)) as fp:
                comments.update(json.load(fp))
        else:
            with open(os.path.join(d, file)) as fp:
                categories.update(json.load(fp))
    for key in categories:
        categories[key] = ast.literal_eval(categories[key])
    for key in comments:
        new = []
        for comment in comments[key]:
            new.append([clean_str(word) for word in comment])
        comments[key] = new
    for key in categories:
        categories[key] = [list(filter(lambda x: x != "and", clean_str(word).split())) for word in categories[key]]
    with open("categories.json", 'w') as of:
        json.dump(categories, of, indent=2)
    with open("comments.json", 'w') as of:
        json.dump(comments, of, indent=2)


def combinesmall():
    with open(os.path.join("..", "NYT", "commentsMarch2018.json")) as fp:
        comments = json.loads(fp)
    with open(os.path.join("..", "NYT", "categoriesMarch2018.json")) as fp:
        categories = json.loads(fp)
    for key in categories:
        categories[key] = ast.literal_eval(categories[key])
    for key in comments:
        new = []
        for comment in comments[key]:
            new.append([clean_str(word) for word in comment])
        comments[key] = new
    for key in categories:
        categories[key] = [list(filter(lambda x: x != "and", clean_str(word).split())) for word in categories[key]]
    with open("../small_dataset/categories.json", 'w') as of:
        json.dump(categories, of, indent=2)
    with open("../small_dataset/comments.json", 'w') as of:
        json.dump(comments, of, indent=2)


if __name__ == "__main__":
    #combine_files("NYT")
    pass
