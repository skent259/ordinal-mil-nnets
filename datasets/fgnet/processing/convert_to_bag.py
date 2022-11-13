import os
import sys

import numpy as np
import pandas as pd

[sys.path.append(i) for i in [".", "..", "../.."]]  # need to access modules

from datasets.miltools import MILDataSetConverter

proj_dir = os.getcwd()
data_dir = "datasets/fgnet"
os.chdir(data_dir)


def random_over_sampler(df, y_col: str):
    df = df.copy()
    classes = np.unique(df[y_col])
    class_indices = dict(zip(classes, np.arange(len(classes))))

    y = df[y_col].apply(lambda x: class_indices[x]).reset_index(drop=True)

    top_n = max(y.value_counts())

    def boostrap_sample(x: pd.Series, n: int):
        return x.sample(n=n, replace=True).index

    samples = [boostrap_sample(y[y == i], top_n) for i in range(len(classes))]
    samples_flat = [i for l in samples for i in l]

    return df.iloc[samples_flat].sample(frac=1)


# Just one for now, will need to make multiple in the future
def convert_one(path_in, path_out, bag_size, wr, seed, balance: bool = False):
    df = pd.read_csv(path_in, dtype=str, index_col=0)
    # converter = MILDataSetConverter(df, convert_type="random", shuffle=True)
    converter = MILDataSetConverter(
        df, y_col="age_group", convert_type="wr", shuffle=True
    )

    df_bag = converter.convert(bag_size, wr, seed)

    df_bag["age_group_bag"] = df_bag["age_group"].apply(lambda x: f'"{max(x)}"')

    if balance:
        df_bag = random_over_sampler(df_bag, "age_group_bag")
    df_bag.to_csv(path_out)
    return None


N_DATASETS = 5

for i in range(N_DATASETS):
    WITNESS_RATE = 0.5
    BAG_SIZE = 4

    convert_one(
        "fgnet_train.csv",
        f"fgnet_bag_wr={WITNESS_RATE}_size={BAG_SIZE}_i={i}_train.csv",
        bag_size=BAG_SIZE,
        wr=WITNESS_RATE,
        seed=8 + i,
        balance=True,
    )
    convert_one(
        "fgnet_valid.csv",
        f"fgnet_bag_wr={WITNESS_RATE}_size={BAG_SIZE}_i={i}_valid.csv",
        bag_size=BAG_SIZE,
        wr=WITNESS_RATE,
        seed=8 + i,
    )
    convert_one(
        "fgnet_test.csv",
        f"fgnet_bag_wr={WITNESS_RATE}_size={BAG_SIZE}_i={i}_test.csv",
        bag_size=BAG_SIZE,
        wr=WITNESS_RATE,
        seed=8 + i,
    )

os.chdir(proj_dir)

# To read in:
# import ast
# tmp = pd.read_csv(out_fname, index_col=0).applymap(ast.literal_eval)
