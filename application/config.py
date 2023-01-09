import sys
from dataclasses import dataclass
from typing import Dict, List

[sys.path.append(i) for i in [".", ".."]]  # need to access datasets and models module

from models.architecture import MILType, OrdinalType


@dataclass
class TMAConfig:

    mil_method: MILType
    pooling_mode: str
    metric: str
    rep: int
    ordinal_method: OrdinalType = None
    learning_rate: float = None
    n_fc_layers: int = None
    fc_layer_size: int = None
    epochs: int = None

    @property
    def file(self) -> Dict[str, str]:
        base_name = (
            f"tma__mil={self.mil_method.value}"
            + f"_pool={self.pooling_mode}"
            + f"_metric={self.metric}"
            + f"_rep={self.rep}"
        )

        return {
            "csv_log": base_name + "_training.log",
            "model": base_name + "_model-{epoch:02d}.hdf5",
            "metrics": base_name + "_metrics.csv",
            "gridsearch": base_name + "_gridsearch_summary.csv",
        }
