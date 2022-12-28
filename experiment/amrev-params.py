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

# amrev-4.0.1
exp_401 = expand_grid(
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
        "data_set_type": [DataSetType.AMREV_TV],
        "data_set_name": [
            "amrev_TVs_i=0",
            "amrev_TVs_i=1",
            "amrev_TVs_i=2",
            "amrev_TVs_i=3",
            "amrev_TVs_i=4",
            "amrev_TVs_i=5",
            "amrev_TVs_i=6",
            "amrev_TVs_i=7",
            "amrev_TVs_i=8",
            "amrev_TVs_i=9",
        ],
        "batch_size": [1],
        "learning_rate": [0.001, 0.0001, 0.00001],
        "epochs": [50],
        "early_stopping": [False],
    }
)

exp_401 = hoist(exp_401, col="mil_pool_combo", names=["mil_method", "pooling_mode"])
exp_401 = exp_401[col_order]

experiment_df_to_csv(exp_401, "experiment/params/experiment-amrev-4.0.1.csv")

# NOTE: In future, if add AMREV_CAMERAS, AMREV_LAPTOPS, etc., use 4.1.1, 4.2.1, etc.

# # Test read in
# test = pd.read_csv("experiment/params/experiment-amrev-4.0.1.csv")
# test["ordinal_method"] = [OrdinalType[x] for x in test["ordinal_method"]]
# test["mil_method"] = [MILType[x] for x in test["mil_method"]]
# test["data_set_type"] = [DataSetType[x] for x in test["data_set_type"]]
