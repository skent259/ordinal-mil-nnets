import sys

[sys.path.append(i) for i in [".", ".."]]  # need to access datasets and models module

import pandas as pd

from application.config import TMAConfig
from experiment.paramtools import expand_grid, hoist
from models.architecture import MILType


def example_df_to_csv(exp: pd.DataFrame, file: str) -> None:
    exp = exp.copy()

    # Add a column for file name
    exp["config"] = [TMAConfig(**exp_args) for _, exp_args in exp.iterrows()]
    exp["file"] = [x.file["metrics"] for x in exp["config"]]

    # Convert formally typed columns to their "name"
    for col in ["mil_method"]:
        exp[col] = exp[col].apply(lambda x: x.name)

    # Save to .csv file
    del exp["config"]
    exp.to_csv(file, index=0)


col_order = [
    "mil_method",
    "pooling_mode",
    "metric",
    "rep",
    "epochs",
]

# tma-7.0.1
ex_701 = expand_grid(
    {
        "mil_pool_combo": [[MILType.CAP_MI_NET, "max"], [MILType.MI_ATTENTION, None]],
        "metric": ["mae", "accuracy"],
        "rep": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "epochs": [150],
    }
)

ex_701 = hoist(ex_701, col="mil_pool_combo", names=["mil_method", "pooling_mode"])
ex_701 = ex_701[col_order]

example_df_to_csv(ex_701, "application/params/example-tma-7.0.1.csv")


# # Test read in
# test = pd.read_csv("application/params/example-tma-7.0.1.csv")
# test["ordinal_method"] = [OrdinalType[x] for x in test["ordinal_method"]]
# test["mil_method"] = [MILType[x] for x in test["mil_method"]]
# test["data_set_type"] = [DataSetType[x] for x in test["data_set_type"]]
