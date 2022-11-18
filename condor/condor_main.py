import argparse
import sys

[sys.path.append(i) for i in [".", ".."]]  # need to access datasets and models module

import pandas as pd

from models.architecture import MILType, OrdinalType
from models.dataset import DataSetType
from models.experiment import ExperimentConfig, ExperimentRunner

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exp_file", required=True)
parser.add_argument("-i", "--i", default=1, required=False, type=int)
parser.add_argument("--output_dir", default="", required=False)
args = parser.parse_args()

# Read in experiment file
exp = pd.read_csv(args.exp_file)
exp["ordinal_method"] = [OrdinalType[x] for x in exp["ordinal_method"]]
exp["mil_method"] = [MILType[x] for x in exp["mil_method"]]
exp["data_set_type"] = [DataSetType[x] for x in exp["data_set_type"]]
del exp["file"]

# Run experiment
exp_args = exp.iloc[args.i]
exp_config = ExperimentConfig(**exp_args)
exp = ExperimentRunner(exp_config, output_dir=args.output_dir)

print(exp_args)
exp.run()
