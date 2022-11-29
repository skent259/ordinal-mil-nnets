import sys

[sys.path.append(i) for i in [".", ".."]]  # need to access datasets and models module

from itertools import product

import pandas as pd

from models.architecture import MILType, OrdinalType
from models.dataset import DataSetType
from models.experiment import ExperimentConfig, ExperimentRunner


def expand_grid(dictionary):
    data = [row for row in product(*dictionary.values())]
    return pd.DataFrame(data, columns=dictionary.keys())


experiment_1 = expand_grid(
    {
        "ordinal_method": [OrdinalType.CORAL, OrdinalType.CORN],
        "mil_method": [MILType.MI_ATTENTION, MILType.MI_GATED_ATTENTION],
        "data_set_type": [DataSetType.FGNET],
        "data_set_name": ["fgnet_bag_wr=0.5_size=4_i=0",],
        "batch_size": [1],
        "learning_rate": [0.0001],
        "epochs": [2],
        "pooling_mode": [None],
    }
)

print(experiment_1)

for _, exp_args in experiment_1.iterrows():
    exp_config = ExperimentConfig(**exp_args)
    print(exp_config)
    # print(exp_config.model_architecture.build().summary())

    exp = ExperimentRunner(exp_config)
    # exp.run()
    print("\n\n\n")
