import sys

[sys.path.append(i) for i in [".", ".."]]  # need to access datasets and models module

from models.architecture import MILType, OrdinalType
from models.dataset import DataSetType
from models.experiment import ExperimentConfig, ExperimentRunner

experiment_1 = [
    {
        "ordinal_method": OrdinalType.CORAL,
        "mil_method": MILType.CAP_MI_NET,
        "data_set_type": DataSetType.FGNET,
        "data_set_name": "fgnet_bag_wr",
        "batch_size": 1,
        "learning_rate": 0.05,
        "epochs": 1,
    }
]

for exp_args in experiment_1:
    exp_config = ExperimentConfig(**exp_args)
    print(exp_config.model_architecture.build().summary())

    exp = ExperimentRunner(exp_config)
    exp.run()
