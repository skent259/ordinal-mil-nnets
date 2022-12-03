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


# bcnb-3.0.1
experiment_301 = [
    {
        "ordinal_method": [OrdinalType.CORAL, OrdinalType.CORN],
        "data_set_type": [DataSetType.BCNB_ALN],
        "mil_method": [MILType.CAP_MI_NET, MILType.MI_NET, MILType.CAP_MI_NET_DS],
        "pooling_mode": ["max", "mean"],
        "data_set_name": [
            "bcnb_aln_i=0",
            "bcnb_aln_i=1",
            "bcnb_aln_i=2",
            "bcnb_aln_i=3",
            "bcnb_aln_i=4",
        ],
        "batch_size": [1],
        "learning_rate": [0.001, 0.0001, 0.00001],
        "epochs": [25],
        "early_stopping": [False],
    },
    {
        "ordinal_method": [OrdinalType.CORAL, OrdinalType.CORN],
        "data_set_type": [DataSetType.BCNB_ALN],
        "mil_method": [MILType.MI_ATTENTION, MILType.MI_GATED_ATTENTION],
        "pooling_mode": [None],
        "data_set_name": [
            "bcnb_aln_i=0",
            "bcnb_aln_i=1",
            "bcnb_aln_i=2",
            "bcnb_aln_i=3",
            "bcnb_aln_i=4",
        ],
        "batch_size": [1],
        "learning_rate": [0.001, 0.0001, 0.00001],
        "epochs": [25],
        "early_stopping": [False],
    },
]

experiment_301_df = pd.concat([expand_grid(x) for x in experiment_301])
experiment_df_to_csv(experiment_301_df, "experiment/params/experiment-bcnb-3.0.1.csv")


# # Test read in
# test = pd.read_csv("experiment/params/experiment-bcnb-3.0.1.csv")
# test["ordinal_method"] = [OrdinalType[x] for x in test["ordinal_method"]]
# test["mil_method"] = [MILType[x] for x in test["mil_method"]]
# test["data_set_type"] = [DataSetType[x] for x in test["data_set_type"]]
