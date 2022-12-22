import json
import os
import re

import numpy as np
import pandas as pd

# TODO: Think about pulling reviews in a balanced manner
# This would require separate random selections and might be tricky.
# Also, it's not as fair with previous comparison

# TODO: There's some annoying javascript code in a small number of reviews...
# see e.g. R3D8BMCJ2Z9KZ4 in amrev_TVs_i=0_test.csv

# Pull in all review information from .json files
def load_reviews(data_dir) -> pd.DataFrame:
    reviews = []
    ratings = []
    ids = []
    for fname in os.listdir(data_dir):
        # print(fname)
        with open(data_dir + fname) as f:
            data = json.load(f)
            for rev in data["Reviews"]:
                if rev["Content"] is not None:
                    reviews.append(rev["Content"])
                    ratings.append(rev["Overall"])
                    ids.append(rev["ReviewID"])

    return pd.DataFrame({"id": ids, "rating": ratings, "review": reviews})


# Randomly select from the reviews, ratings, and ids lists
def randomly_select(df: pd.DataFrame, n: int, rng):
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
data_dir = "datasets/amrev"
os.chdir(data_dir)

# datasets = ['cameras', 'laptops', 'mobilephone', 'tablets', 'TVs', 'video_surveillance']
datasets = ["TVs"]
n_rep = 10
n_train = 1000
n_valid = 200
n_test = 2000

rng = np.random.default_rng(8)

for dataset in datasets:
    print(f"{dataset=}")
    data_dir = f"reviews/{dataset}/"

    review_df = load_reviews(data_dir)
    review_df = review_df.drop_duplicates(subset=["id"]).reset_index(drop=True)

    # clean up html, quotes
    review_df["review"] = review_df["review"].apply(cleanhtml)
    review_df["review"] = review_df["review"].apply(remove_annoying_str)

    # turn to hard strings for ast literal eval
    review_df = review_df.applymap(lambda x: f'"{x}"')

    for i in range(n_rep):
        print(f"{n_train=}, {i=}")
        review_df_i = randomly_select(review_df, n=n_train + n_valid + n_test, rng=rng)

        train, valid, test = np.split(review_df_i, [n_train, n_train + n_valid])

        # check correct size
        assert train.shape[0] == n_train
        assert valid.shape[0] == n_valid
        assert test.shape[0] == n_test

        # reset index
        train = train.reset_index(drop=True)
        valid = valid.reset_index(drop=True)
        test = test.reset_index(drop=True)

        # save to csv file
        train.to_csv(f"splits_bag/amrev_{dataset}_{i=}_train.csv")
        valid.to_csv(f"splits_bag/amrev_{dataset}_{i=}_valid.csv")
        test.to_csv(f"splits_bag/amrev_{dataset}_{i=}_test.csv")

