from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import coral_ordinal as coral
import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential

from models.dataset import DataSetType
from models.mil_nets.layer import BagWise, MILPool


class OrdinalType(Enum):
    CORN = "corn"
    CORAL = "coral"


class MILType(Enum):
    MI_NET = "mi-net"
    CAP_MI_NET = "cap-mi-net"
    CAP_MI_NET_DS = "cap-mi-net-ds"


@dataclass
class ModelArchitecture:
    """
    Model architecture to be used by experiment methods
    """

    ordinal_type: OrdinalType
    mil_type: MILType
    data_set_type: DataSetType
    data_set_img_size: Tuple[int]
    n_classes: int
    pooling_mode: str = "max"

    def build(self):
        arch = self.base_layers + self.last_layers
        model = Sequential(arch)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.05),
            loss=self.ordinal_loss,
            metrics=[self.ordinal_metrics],
        )

        return model

    @property
    def base_layers(self) -> Tuple:
        if self.data_set_type == DataSetType.FGNET:
            return [
                Input(shape=(None,) + self.data_set_img_size),
                BagWise(Conv2D(32, (5, 5), padding="same", activation="relu")),
                BagWise(Conv2D(32, (5, 5), activation="relu")),
                BagWise(MaxPooling2D(pool_size=(3, 3))),
                Dropout(0.25),
                BagWise(Conv2D(64, (3, 3), padding="same", activation="relu")),
                BagWise(Conv2D(64, (3, 3), activation="relu")),
                BagWise(MaxPooling2D(pool_size=(3, 3))),
                Dropout(0.25),
                BagWise(Conv2D(64, (3, 3), padding="same", activation="relu")),
                BagWise(Conv2D(64, (3, 3), activation="relu")),
                BagWise(MaxPooling2D(pool_size=(3, 3))),
                Dropout(0.25),
                BagWise(Flatten()),
                BagWise(Dense(256, activation="relu")),
                Dropout(0.5),
            ]

    @property
    def last_layers(self) -> Tuple:
        if self.mil_type is MILType.MI_NET:
            return [
                self.ordinal_layer,
                MILPool(pooling_mode=self.pooling_mode)(),
            ]

        if self.mil_type is MILType.CAP_MI_NET:
            return [
                MILPool(pooling_mode=self.pooling_mode)(),
                self.ordinal_layer,
            ]

    @property
    def ordinal_layer(self):
        if self.ordinal_type is OrdinalType.CORAL:
            return coral.CoralOrdinal(self.n_classes)
        if self.ordinal_type is OrdinalType.CORN:
            return coral.CornOrdinal(self.n_classes)

    @property
    def ordinal_loss(self):
        if self.ordinal_type is OrdinalType.CORAL:
            return coral.OrdinalCrossEntropy(num_classes=self.n_classes)
        if self.ordinal_type is OrdinalType.CORN:
            return coral.CornOrdinalCrossEntropy(num_classes=self.n_classes)

    @property
    def ordinal_metrics(self):
        coral.MeanAbsoluteErrorLabels()
        # TODO: need more metrics here...

    @property
    def mil_architecture(self) -> Tuple:
        return None
