import os
import json
import ast


def combine_files(dir):
    comments = {}
    categories = {}
    files = os.listdir(dir)
    for file in files:
        if "comments" in file:
            continue
        else:
            with open(dir+file) as fp:
                categories.update(json.load(fp))
    for key in categories:
        print(key)
        categories[key] = ast.literal_eval(categories[key])
    with open("categories.json", 'w') as of:
        json.dump(categories, of, indent=2)
    with open("categories.json", 'w') as of:
        json.dump(categories, of, indent=2)


def main():
    combine_files("./NYT/")






if __name__ == "__main__":
    main()
