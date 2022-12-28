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

# bcnb-3.0.1
exp_301 = expand_grid(
    {
        "ordinal_method": [OrdinalType.CORAL, OrdinalType.CORN],
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
        "data_set_type": [DataSetType.BCNB_ALN],
        "data_set_name": [
            "bcnb_aln_i=0",
            "bcnb_aln_i=1",
            "bcnb_aln_i=2",
            "bcnb_aln_i=3",
            "bcnb_aln_i=4",
        ],
        "batch_size": [1],
        "learning_rate": [0.001, 0.0001, 0.00001],
        "epochs": [50],
        "early_stopping": [False],
    }
)

exp_301 = hoist(exp_301, col="mil_pool_combo", names=["mil_method", "pooling_mode"])
exp_301 = exp_301[col_order]

experiment_df_to_csv(exp_301, "experiment/params/experiment-bcnb-3.0.1.csv")


# bcnb-3.0.2
exp_302 = expand_grid(
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
        "data_set_type": [DataSetType.BCNB_ALN],
        "data_set_name": [
            "bcnb_aln_i=0",
            "bcnb_aln_i=1",
            "bcnb_aln_i=2",
            "bcnb_aln_i=3",
            "bcnb_aln_i=4",
        ],
        "batch_size": [1],
        "learning_rate": [0.001, 0.0001, 0.00001],
        "epochs": [50],
        "early_stopping": [False],
    }
)

exp_302 = hoist(exp_302, col="mil_pool_combo", names=["mil_method", "pooling_mode"])
exp_302 = exp_302[col_order]

experiment_df_to_csv(exp_302, "experiment/params/experiment-bcnb-3.0.2.csv")


# # Test read in
# test = pd.read_csv("experiment/params/experiment-bcnb-3.0.1.csv")
# test["ordinal_method"] = [OrdinalType[x] for x in test["ordinal_method"]]
# test["mil_method"] = [MILType[x] for x in test["mil_method"]]
# test["data_set_type"] = [DataSetType[x] for x in test["data_set_type"]]
