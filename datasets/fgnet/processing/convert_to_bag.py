import ast
import os
import re
import sys

import numpy as np
import pandas as pd

[sys.path.append(i) for i in [".", ".."]]  # need to access datasets and models module

from models.dataset import convert_dataset_to_bag_level

proj_dir = os.getcwd()
data_dir = "datasets/fgnet"
os.chdir(data_dir)

# Just one for now, will need to make multiple in the future
def convert_one(path_in, path_out, bag_size, seed):
    df = pd.read_csv(path_in, dtype=str, index_col=0)
    df_bag = convert_dataset_to_bag_level(df, bag_size, shuffle=True, seed=seed)

    df_bag["age_group_bag"] = df_bag["age_group"].apply(lambda x: f'"{max(x)}"')

    df_bag.to_csv(path_out)
    return None


convert_one("fgnet_train.csv", "fgnet_bag_train.csv", bag_size=3, seed=8)
convert_one("fgnet_valid.csv", "fgnet_bag_valid.csv", bag_size=3, seed=8)
convert_one("fgnet_test.csv", "fgnet_bag_test.csv", bag_size=3, seed=8)

os.chdir(proj_dir)


# To read in:
# tmp = pd.read_csv(out_fname, index_col=0).applymap(ast.literal_eval)
