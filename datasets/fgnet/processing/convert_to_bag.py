import os
import sys

import pandas as pd

[sys.path.append(i) for i in [".", "..", "../.."]]  # need to access modules

from datasets.miltools import MILDataSetConverter

proj_dir = os.getcwd()
data_dir = "datasets/fgnet"
os.chdir(data_dir)

# Just one for now, will need to make multiple in the future
def convert_one(path_in, path_out, bag_size, wr, seed):
    df = pd.read_csv(path_in, dtype=str, index_col=0)
    # converter = MILDataSetConverter(df, convert_type="random", shuffle=True)
    converter = MILDataSetConverter(
        df, y_col="age_group", convert_type="wr", shuffle=True
    )

    df_bag = converter.convert(bag_size, wr, seed)

    df_bag["age_group_bag"] = df_bag["age_group"].apply(lambda x: f'"{max(x)}"')

    df_bag.to_csv(path_out)
    return None


convert_one("fgnet_train.csv", "fgnet_bag_wr_train.csv", bag_size=4, wr=0.5, seed=8)
convert_one("fgnet_valid.csv", "fgnet_bag_wr_valid.csv", bag_size=4, wr=0.5, seed=8)
convert_one("fgnet_test.csv", "fgnet_bag_wr_test.csv", bag_size=4, wr=0.5, seed=8)

os.chdir(proj_dir)


# To read in:
# import ast
# tmp = pd.read_csv(out_fname, index_col=0).applymap(ast.literal_eval)
