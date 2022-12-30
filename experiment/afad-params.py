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

# afad-6.0.1
exp_601 = expand_grid(
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
        "data_set_type": [DataSetType.AFAD],
        "data_set_name": [
            "afad_bag_wr=0.5_size=4_i=0_j=0",
            "afad_bag_wr=0.5_size=4_i=0_j=1",
            "afad_bag_wr=0.5_size=4_i=0_j=2",
            "afad_bag_wr=0.5_size=4_i=0_j=3",
            "afad_bag_wr=0.5_size=4_i=0_j=4",
            "afad_bag_wr=0.5_size=4_i=1_j=0",
            "afad_bag_wr=0.5_size=4_i=1_j=1",
            "afad_bag_wr=0.5_size=4_i=1_j=2",
            "afad_bag_wr=0.5_size=4_i=1_j=3",
            "afad_bag_wr=0.5_size=4_i=1_j=4",
        ],
        "batch_size": [1],
        "learning_rate": [0.001, 0.0001, 0.00001],
        "epochs": [50],
        "early_stopping": [True],
    }
)

exp_601 = hoist(exp_601, col="mil_pool_combo", names=["mil_method", "pooling_mode"])
exp_601 = exp_601[col_order]

experiment_df_to_csv(exp_601, "experiment/params/experiment-afad-6.0.1.csv")


# # Test read in
# test = pd.read_csv("experiment/params/experiment-afad-6.0.1.csv")
# test["ordinal_method"] = [OrdinalType[x] for x in test["ordinal_method"]]
# test["mil_method"] = [MILType[x] for x in test["mil_method"]]
# test["data_set_type"] = [DataSetType[x] for x in test["data_set_type"]]
