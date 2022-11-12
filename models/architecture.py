from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import coral_ordinal as coral
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential

from models.clm_qwk.resnet import bagwise_residual_block
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
        inputs = self.input_layer()
        base = self.base_layers(inputs)
        last = self.last_layers(base)

        model = Model(inputs=inputs, outputs=last)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.05),
            loss=self.ordinal_loss,
            metrics=[self.ordinal_metrics],
        )

        return model

    def input_layer(self) -> tf.keras.layers.Layer:
        return Input(shape=(None,) + self.data_set_img_size)

    def base_layers(self, layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
        if self.data_set_type == DataSetType.FGNET:

            # Small-ish Residual Network from Vargas, Gutierrez, Hervas-Matrinez (2020) Neurocomputing
            x = BagWise(
                Conv2D(32, (7, 7), strides=2, padding="same", activation="relu")
            )(layer)
            x = BagWise(MaxPooling2D(pool_size=(3, 3), strides=2))(x)

            x = bagwise_residual_block(x, 64, (3, 3), stride=1, nonlinearity="relu")
            x = bagwise_residual_block(x, 64, (3, 3), stride=1, nonlinearity="relu")

            x = bagwise_residual_block(x, 128, (3, 3), stride=2, nonlinearity="relu")
            x = bagwise_residual_block(x, 128, (3, 3), stride=1, nonlinearity="relu")
            x = bagwise_residual_block(x, 128, (3, 3), stride=1, nonlinearity="relu")

            x = bagwise_residual_block(x, 256, (3, 3), stride=2, nonlinearity="relu")
            x = bagwise_residual_block(x, 256, (3, 3), stride=1, nonlinearity="relu")
            x = bagwise_residual_block(x, 256, (3, 3), stride=1, nonlinearity="relu")

            x = bagwise_residual_block(x, 512, (3, 3), stride=2, nonlinearity="relu")
            x = bagwise_residual_block(x, 512, (3, 3), stride=1, nonlinearity="relu")
            x = bagwise_residual_block(x, 512, (3, 3), stride=1, nonlinearity="relu")

            x = BagWise(GlobalAveragePooling2D(data_format="channels_last"))(x)
            return x

    def last_layers(self, layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
        if self.mil_type is MILType.MI_NET:
            x = self.ordinal_layer(layer)
            x = MILPool(pooling_mode=self.pooling_mode)()(x)
            return x

        if self.mil_type is MILType.CAP_MI_NET:
            x = MILPool(pooling_mode=self.pooling_mode)()(layer)
            x = self.ordinal_layer(x)
            return x

    @property
    def ordinal_layer(self) -> tf.keras.layers.Layer:
        if self.ordinal_type is OrdinalType.CORAL:
            return coral.CoralOrdinal(self.n_classes)
        if self.ordinal_type is OrdinalType.CORN:
            return coral.CornOrdinal(self.n_classes)

    @property
    def ordinal_loss(self) -> tf.keras.layers.Layer:
        if self.ordinal_type is OrdinalType.CORAL:
            return coral.OrdinalCrossEntropy(num_classes=self.n_classes)
        if self.ordinal_type is OrdinalType.CORN:
            return coral.CornOrdinalCrossEntropy()

    @property
    def ordinal_metrics(self) -> tf.keras.layers.Layer:
        if self.ordinal_type is OrdinalType.CORAL:
            return coral.MeanAbsoluteErrorLabels()
        if self.ordinal_type is OrdinalType.CORN:
            return coral.MeanAbsoluteErrorLabels(corn_logits=True)
        # TODO: need more metrics here...

    @property
    def mil_architecture(self) -> Tuple:
        return None
