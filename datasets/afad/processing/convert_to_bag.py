import os
import sys

import numpy as np
import pandas as pd

[sys.path.append(i) for i in [".", "..", "../.."]]  # need to access modules

from datasets.miltools import MILDataSetConverter

proj_dir = os.getcwd()
data_dir = "datasets/afad"
os.chdir(data_dir)


def random_under_sampler(df, y_col: str, limit: int = float("inf")):
    """
    Down-sample a data set to be split evenly by `y_col`.

    NOTE: this varies slightly from the `random_under_sampler` in 
    datasets/bcnb/processing/split_and_convert.py
    In particular, we use regular sampling instead of bootstrap sampling
    """

    df = df.copy()
    classes = np.unique(df[y_col])
    class_indices = dict(zip(classes, np.arange(len(classes))))

    y = df[y_col].apply(lambda x: class_indices[x]).reset_index(drop=True)

    low_n = min(y.value_counts())
    low_n = min(low_n, limit)

    def regular_sample(x: pd.Series, n: int):
        return x.sample(n=n, replace=False).index

    samples = [regular_sample(y[y == i], low_n) for i in range(len(classes))]
    samples_flat = [i for l in samples for i in l]

    return df.iloc[samples_flat].sample(frac=1)


# Just one for now, will need to make multiple in the future
def convert_one(path_in, path_out, bag_size, wr, seed, balance: bool = False):
    df = pd.read_csv(path_in, dtype=str, index_col=0)

    converter = MILDataSetConverter(df, y_col="age", convert_type="wr", shuffle=True)

    df_bag = converter.convert(bag_size, wr, seed)

    df_bag["age_bag"] = df_bag["age"].apply(lambda x: f'"{max(x)}"')

    if balance:
        df_bag = random_under_sampler(df_bag, "age_bag")
    df_bag.to_csv(path_out)
    return None


N_DATASETS_SPLIT = 2
N_DATASETS_BAG = 5

for i in range(N_DATASETS_SPLIT):
    for j in range(N_DATASETS_BAG):
        WITNESS_RATE = 0.5
        BAG_SIZE = 4

        convert_one(
            f"splits/afad_{i=}_train.csv",
            f"splits_bag/afad_bag_wr={WITNESS_RATE}_size={BAG_SIZE}_{i=}_{j=}_train.csv",
            bag_size=BAG_SIZE,
            wr=WITNESS_RATE,
            seed=8 + (N_DATASETS_BAG * i) + j,
            balance=True,
        )
        convert_one(
            f"splits/afad_{i=}_valid.csv",
            f"splits_bag/afad_bag_wr={WITNESS_RATE}_size={BAG_SIZE}_{i=}_{j=}_valid.csv",
            bag_size=BAG_SIZE,
            wr=WITNESS_RATE,
            seed=8 + (N_DATASETS_BAG * i) + j,
        )
        convert_one(
            f"splits/afad_{i=}_test.csv",
            f"splits_bag/afad_bag_wr={WITNESS_RATE}_size={BAG_SIZE}_{i=}_{j=}_test.csv",
            bag_size=BAG_SIZE,
            wr=WITNESS_RATE,
            seed=8 + (N_DATASETS_BAG * i) + j,
        )

os.chdir(proj_dir)

# To read in:
# import ast
# tmp = pd.read_csv(out_fname, index_col=0).applymap(ast.literal_eval)
