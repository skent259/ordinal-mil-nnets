import sys

[sys.path.append(i) for i in [".", ".."]]  # need to access datasets and models module

from experiment.paramtools import expand_grid, experiment_df_to_csv, hoist
from models.architecture import MILType, OrdinalType
from models.dataset import DataSetType

col_order = [
    "ordinal_method",
    "mil_method",
    "pooling_mode",
    "data_set_type",
    "data_set_name",
    "batch_size",
    "learning_rate",
    "epochs",
    "early_stopping",
]

# imdb-5.0.1
exp_501 = expand_grid(
    {
        "ordinal_method": [
            OrdinalType.CORAL,
            OrdinalType.CORN,
            OrdinalType.CLM_QWK_LOGIT,
            OrdinalType.CLM_QWK_PROBIT,
            OrdinalType.CLM_QWK_CLOGLOG,
        ],
        "mil_pool_combo": [
            [MILType.CAP_MI_NET, "max"],
            [MILType.CAP_MI_NET, "mean"],
            [MILType.MI_NET, "max"],
            [MILType.MI_NET, "mean"],
            [MILType.CAP_MI_NET_DS, "max"],
            [MILType.CAP_MI_NET_DS, "mean"],
            [MILType.MI_ATTENTION, None],
            [MILType.MI_GATED_ATTENTION, None],
        ],
        "data_set_type": [DataSetType.IMDB],
        "data_set_name": [
            "imdb_i=0",
            "imdb_i=1",
            "imdb_i=2",
            "imdb_i=3",
            "imdb_i=4",
            "imdb_i=5",
            "imdb_i=6",
            "imdb_i=7",
            "imdb_i=8",
            "imdb_i=9",
        ],
        "batch_size": [1],
        "learning_rate": [0.001, 0.0001, 0.00001],
        "epochs": [50],
        "early_stopping": [False],
    }
)

exp_501 = hoist(exp_501, col="mil_pool_combo", names=["mil_method", "pooling_mode"])
exp_501 = exp_501[col_order]

experiment_df_to_csv(exp_501, "experiment/params/experiment-imdb-5.0.1.csv")


# # Test read in
# test = pd.read_csv("experiment/params/experiment-imdb-5.0.1.csv")
# test["ordinal_method"] = [OrdinalType[x] for x in test["ordinal_method"]]
# test["mil_method"] = [MILType[x] for x in test["mil_method"]]
# test["data_set_type"] = [DataSetType[x] for x in test["data_set_type"]]
