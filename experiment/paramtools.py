import sys
from itertools import product

[sys.path.append(i) for i in [".", ".."]]  # need to access datasets and models module

import pandas as pd

from models.experiment import ExperimentConfig


def expand_grid(dictionary):
    data = [row for row in product(*dictionary.values())]
    return pd.DataFrame(data, columns=dictionary.keys())


def hoist(df, col, names):
    """
    Pull out list column into individual columns
    """
    df = df.copy()
    df[names] = pd.DataFrame(df[col].tolist(), index=df.index)
    del df[col]
    return df


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
