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


# aes-2.0.1
experiment_201 = expand_grid(
    {
        "ordinal_method": [OrdinalType.CORAL, OrdinalType.CORN],
        "mil_method": [MILType.CAP_MI_NET, MILType.MI_NET],
        "data_set_type": [DataSetType.AES],
        "data_set_name": [
            "aes_bag_wr=0.5_size=4_i=0_j=0",
            "aes_bag_wr=0.5_size=4_i=0_j=1",
            "aes_bag_wr=0.5_size=4_i=0_j=2",
            "aes_bag_wr=0.5_size=4_i=0_j=3",
            "aes_bag_wr=0.5_size=4_i=0_j=4",
        ],
        "batch_size": [1],
        "learning_rate": [0.01, 0.001, 0.0001],
        "epochs": [75],
        "pooling_mode": ["max", "mean"],
        "early_stopping": [True],
    }
)

experiment_df_to_csv(experiment_201, "experiment/params/experiment-aes-2.0.1.csv")


# aes-2.0.2
experiment_202 = expand_grid(
    {
        "ordinal_method": [OrdinalType.CORAL, OrdinalType.CORN],
        "mil_method": [MILType.CAP_MI_NET_DS],
        "data_set_type": [DataSetType.AES],
        "data_set_name": [
            "aes_bag_wr=0.5_size=4_i=0_j=0",
            "aes_bag_wr=0.5_size=4_i=0_j=1",
            "aes_bag_wr=0.5_size=4_i=0_j=2",
            "aes_bag_wr=0.5_size=4_i=0_j=3",
            "aes_bag_wr=0.5_size=4_i=0_j=4",
        ],
        "batch_size": [1],
        "learning_rate": [0.01, 0.001, 0.0001],
        "epochs": [75],
        "pooling_mode": ["max", "mean"],
        "early_stopping": [True],
    }
)

experiment_df_to_csv(experiment_202, "experiment/params/experiment-aes-2.0.2.csv")

# # Test read in
# test = pd.read_csv("experiment/params/experiment-aes-2.0.1.csv")
# test["ordinal_method"] = [OrdinalType[x] for x in test["ordinal_method"]]
# test["mil_method"] = [MILType[x] for x in test["mil_method"]]
# test["data_set_type"] = [DataSetType[x] for x in test["data_set_type"]]
