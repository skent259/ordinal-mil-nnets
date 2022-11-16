import argparse
import sys
from itertools import product

[sys.path.append(i) for i in [".", ".."]]  # need to access datasets and models module

import pandas as pd

from models.architecture import MILType, OrdinalType
from models.dataset import DataSetType
from models.experiment import ExperimentConfig, ExperimentRunner

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--i", default=1, required=False, type=int)
parser.add_argument("--output_dir", default="", required=False)
args = parser.parse_args()


def expand_grid(dictionary):
    data = [row for row in product(*dictionary.values())]
    return pd.DataFrame(data, columns=dictionary.keys())


experiment_1 = expand_grid(
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

# Run experiment
exp_args = experiment_1.iloc[args.i]
exp_config = ExperimentConfig(**exp_args)
exp = ExperimentRunner(exp_config, output_dir=args.output_dir)

print(exp_args)
exp.run()

