import json
import os
import re
import uuid

import numpy as np
import pandas as pd


def pull_rating(file_name):
    reg = r"([0-9]+)_([0-9]+).txt"
    subst = "\\g<2>"
    return re.sub(reg, subst, file_name, 0)


def load_reviews(data_dir, subdirs):
    """Pull in all review information from .txt files"""
    reviews = []
    ratings = []
    ids = []
    for subdir in subdirs:
        for fname in os.listdir(data_dir + subdir):
            # print(fname)
            r = pull_rating(fname)
            ratings.append(r)
            ids.append(str(uuid.uuid4()))

            with open(data_dir + subdir + fname) as f:
                data = f.read()
                reviews.append(data)

    return pd.DataFrame({"id": ids, "rating": ratings, "review": reviews})


def randomly_select(df: pd.DataFrame, n: int, rng):
    """Randomly select from the reviews, ratings, and ids lists"""
    ind = [int(i) for i in rng.random(n) * len(df)]

    return df.loc[
        ind,
    ]


def cleanhtml(raw_html):
    # see https://stackoverflow.com/questions/9662346/python-code-to-remove-html-tags-from-a-string
    html_tags = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    html_comments = "(<!--.*?-->)"

    no_tags = re.sub(html_tags, "", raw_html)
    return re.sub(html_comments, "", no_tags, flags=re.DOTALL)


def remove_annoying_str(text: str) -> str:
    reg = re.compile(r"[\"\n\\]")
    return re.sub(reg, "", text)


proj_dir = os.getcwd()
data_dir = "datasets/imdb"
os.chdir(data_dir)

n_rep = 10
n_train = 1000
n_valid = 200
n_test = 2000

rng = np.random.default_rng(8)

data_dir = "reviews/"
review_df_train = load_reviews(data_dir, ["train/pos/", "train/neg/"])
review_df_test = load_reviews(data_dir, ["test/pos/", "test/neg/"])

# clean up html, quotes
review_df_train["review"] = review_df_train["review"].apply(cleanhtml)
review_df_train["review"] = review_df_train["review"].apply(remove_annoying_str)
review_df_test["review"] = review_df_test["review"].apply(cleanhtml)
review_df_test["review"] = review_df_test["review"].apply(remove_annoying_str)

# turn to hard strings for ast literal eval
review_df_train = review_df_train.applymap(lambda x: f'"{x}"')
review_df_test = review_df_test.applymap(lambda x: f'"{x}"')

for i in range(n_rep):
    print(f"{n_train=}, {i=}")
    review_df_train_i = randomly_select(review_df_train, n=n_train + n_valid, rng=rng)
    train, valid = np.split(review_df_train_i, [n_train])

    test = randomly_select(review_df_test, n=n_test, rng=rng)

    # check correct size
    assert train.shape[0] == n_train
    assert valid.shape[0] == n_valid
    assert test.shape[0] == n_test

    # reset index
    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)
    test = test.reset_index(drop=True)

    # save to csv file
    train.to_csv(f"splits_bag/imdb_{i=}_train.csv")
    valid.to_csv(f"splits_bag/imdb_{i=}_valid.csv")
    test.to_csv(f"splits_bag/imdb_{i=}_test.csv")

