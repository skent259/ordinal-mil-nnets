import os
import sys

import numpy as np
import pandas as pd

[sys.path.append(i) for i in [".", "..", "../.."]]  # need to access modules

from datasets.miltools import MILDataSetConverter

proj_dir = os.getcwd()
data_dir = "datasets/aes"
os.chdir(data_dir)

# Just one for now, will need to make multiple in the future
def convert_one(path_in, path_out, bag_size, wr, seed):
    df = pd.read_csv(path_in, dtype=str, index_col=0)
    # converter = MILDataSetConverter(df, convert_type="random", shuffle=True)
    converter = MILDataSetConverter(df, y_col="score", convert_type="wr", shuffle=True)

    df_bag = converter.convert(bag_size, wr, seed)

    df_bag["score_bag"] = df_bag["score"].apply(lambda x: f'"{max(x)}"')

    df_bag.to_csv(path_out)
    return None


N_DATASETS_SPLIT = 1
N_DATASETS_BAG = 5

for i in range(N_DATASETS_SPLIT):
    for j in range(N_DATASETS_BAG):
        WITNESS_RATE = 0.5
        BAG_SIZE = 4

        convert_one(
            "splits/aes_train.csv",
            f"splits_bag/aes_bag_wr={WITNESS_RATE}_size={BAG_SIZE}_{i=}_{j=}_train.csv",
            bag_size=BAG_SIZE,
            wr=WITNESS_RATE,
            seed=8 + (N_DATASETS_BAG * i) + j,
        )
        convert_one(
            "splits/aes_valid.csv",
            f"splits_bag/aes_bag_wr={WITNESS_RATE}_size={BAG_SIZE}_{i=}_{j=}_valid.csv",
            bag_size=BAG_SIZE,
            wr=WITNESS_RATE,
            seed=8 + (N_DATASETS_BAG * i) + j,
        )
        convert_one(
            "splits/aes_test.csv",
            f"splits_bag/aes_bag_wr={WITNESS_RATE}_size={BAG_SIZE}_{i=}_{j=}_test.csv",
            bag_size=BAG_SIZE,
            wr=WITNESS_RATE,
            seed=8 + (N_DATASETS_BAG * i) + j,
        )

os.chdir(proj_dir)

# To read in:
# import ast
# tmp = pd.read_csv(out_fname, index_col=0).applymap(ast.literal_eval)
