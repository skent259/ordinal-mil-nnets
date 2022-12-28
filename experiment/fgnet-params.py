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
        "epochs": [100],
        "pooling_mode": ["max", "mean"],
        "early_stopping": [False],
    }
)

experiment_df_to_csv(experiment_101, "experiment/params/experiment-fgnet-1.0.1.csv")

# fgnet-1.0.2
experiment_102 = expand_grid(
    {
        "ordinal_method": [OrdinalType.CORAL, OrdinalType.CORN],
        "mil_method": [MILType.CAP_MI_NET_DS],
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
        "epochs": [100],
        "pooling_mode": ["max", "mean"],
        "early_stopping": [False],
    }
)

experiment_df_to_csv(experiment_102, "experiment/params/experiment-fgnet-1.0.2.csv")


# fgnet-1.0.3
experiment_103 = expand_grid(
    {
        "ordinal_method": [OrdinalType.CORAL, OrdinalType.CORN],
        "mil_method": [MILType.MI_ATTENTION, MILType.MI_GATED_ATTENTION],
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
        "epochs": [100],
        "pooling_mode": [None],
        "early_stopping": [False],
    }
)

experiment_df_to_csv(experiment_103, "experiment/params/experiment-fgnet-1.0.3.csv")


# fgnet-1.0.4
exp_104 = expand_grid(
    {
        "ordinal_method": [
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
        "epochs": [100],
        "early_stopping": [False],
    }
)

exp_104 = hoist(exp_104, col="mil_pool_combo", names=["mil_method", "pooling_mode"])
exp_104 = exp_104[col_order]

experiment_df_to_csv(exp_104, "experiment/params/experiment-fgnet-1.0.4.csv")


# # Test read in
# test = pd.read_csv("experiment-fgnet-1.0.0.csv")
# test["ordinal_method"] = [OrdinalType[x] for x in test["ordinal_method"]]
# test["mil_method"] = [MILType[x] for x in test["mil_method"]]
# test["data_set_type"] = [DataSetType[x] for x in test["data_set_type"]]
