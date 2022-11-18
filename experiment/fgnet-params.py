import sys
from itertools import product

[sys.path.append(i) for i in [".", ".."]]  # need to access datasets and models module

import pandas as pd

from models.architecture import MILType, OrdinalType
from models.dataset import DataSetType
from models.experiment import ExperimentConfig


def expand_grid(dictionary):
    data = [row for row in product(*dictionary.values())]
    return pd.DataFrame(data, columns=dictionary.keys())


def experiment_df_to_csv(exp: pd.DataFrame, file: str) -> None:
    exp = exp.copy()

    # Add a column for file name
    exp["config"] = [ExperimentConfig(**exp_args) for _, exp_args in exp.iterrows()]
    exp["file"] = [x.file["test_result"] for x in exp["config"]]

    # Convert formally typed columns to their "name"
    for col in ["ordinal_method", "mil_method", "data_set_type"]:
        exp[col] = exp[col].apply(lambda x: x.name)

    # Save to .csv file
    del exp["config"]
    exp.to_csv(file, index=0)


# fgnet-1.0.0
experiment_100 = expand_grid(
    {
        "ordinal_method": [OrdinalType.CORAL, OrdinalType.CORN],
        "mil_method": [MILType.CAP_MI_NET, MILType.MI_NET],
        "data_set_type": [DataSetType.FGNET],
        "data_set_name": [
            "fgnet_bag_wr=0.5_size=4_i=0",
            "fgnet_bag_wr=0.5_size=4_i=1",
            "fgnet_bag_wr=0.5_size=4_i=2",
            "fgnet_bag_wr=0.5_size=4_i=3",
            "fgnet_bag_wr=0.5_size=4_i=4",
        ],
        "batch_size": [1],
        "learning_rate": [0.01, 0.001, 0.0001],
        "epochs": [75],
        "pooling_mode": ["max", "mean"],
    }
)

experiment_df_to_csv(experiment_100, "experiment/params/experiment-fgnet-1.0.0.csv")

experiment_101 = expand_grid(
    {
        "ordinal_method": [OrdinalType.CORAL, OrdinalType.CORN],
        "mil_method": [MILType.CAP_MI_NET, MILType.MI_NET],
        "data_set_type": [DataSetType.FGNET],
        "data_set_name": [
            "fgnet_bag_wr=0.5_size=4_i=0_j=0",
            "fgnet_bag_wr=0.5_size=4_i=0_j=1",
            "fgnet_bag_wr=0.5_size=4_i=0_j=2",
            "fgnet_bag_wr=0.5_size=4_i=0_j=3",
            "fgnet_bag_wr=0.5_size=4_i=0_j=4",
            "fgnet_bag_wr=0.5_size=4_i=1_j=0",
            "fgnet_bag_wr=0.5_size=4_i=1_j=1",
            "fgnet_bag_wr=0.5_size=4_i=1_j=2",
            "fgnet_bag_wr=0.5_size=4_i=1_j=3",
            "fgnet_bag_wr=0.5_size=4_i=1_j=4",
            "fgnet_bag_wr=0.5_size=4_i=2_j=0",
            "fgnet_bag_wr=0.5_size=4_i=2_j=1",
            "fgnet_bag_wr=0.5_size=4_i=2_j=2",
            "fgnet_bag_wr=0.5_size=4_i=2_j=3",
            "fgnet_bag_wr=0.5_size=4_i=2_j=4",
            "fgnet_bag_wr=0.5_size=4_i=3_j=0",
            "fgnet_bag_wr=0.5_size=4_i=3_j=1",
            "fgnet_bag_wr=0.5_size=4_i=3_j=2",
            "fgnet_bag_wr=0.5_size=4_i=3_j=3",
            "fgnet_bag_wr=0.5_size=4_i=3_j=4",
            "fgnet_bag_wr=0.5_size=4_i=4_j=0",
            "fgnet_bag_wr=0.5_size=4_i=4_j=1",
            "fgnet_bag_wr=0.5_size=4_i=4_j=2",
            "fgnet_bag_wr=0.5_size=4_i=4_j=3",
            "fgnet_bag_wr=0.5_size=4_i=4_j=4",
        ],
        "batch_size": [1],
        "learning_rate": [0.01, 0.001, 0.0001],
        "epochs": [150],
        "pooling_mode": ["max", "mean"],
        "early_stopping": [False],
    }
)

experiment_df_to_csv(experiment_101, "experiment/params/experiment-fgnet-1.0.1.csv")

# # Test read in
# test = pd.read_csv("experiment-fgnet-1.0.0.csv")
# test["ordinal_method"] = [OrdinalType[x] for x in test["ordinal_method"]]
# test["mil_method"] = [MILType[x] for x in test["mil_method"]]
# test["data_set_type"] = [DataSetType[x] for x in test["data_set_type"]]
