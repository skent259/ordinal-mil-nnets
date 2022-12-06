from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import coral_ordinal as coral
import tensorflow as tf
from tensorflow.keras import Model, applications
from tensorflow.keras.layers import (
    Activation,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    Lambda,
    MaxPooling2D,
    average,
)
from tensorflow.keras.models import Sequential

from models.clm_qwk.resnet import bagwise_residual_block
from models.dataset import DataSetType
from models.mil_attention.layer import mil_attention_layer
from models.mil_nets.layer import BagWise, MILPool


class OrdinalType(Enum):
    CORN = "corn"
    CORAL = "coral"


class MILType(Enum):
    MI_NET = "mi-net"
    CAP_MI_NET = "cap-mi-net"
    CAP_MI_NET_DS = "cap-mi-net-ds"
    MI_ATTENTION = "mi-attention"
    MI_GATED_ATTENTION = "mi-gated-attention"


@dataclass
class ModelArchitecture:
    """
    Model architecture to be used by experiment methods

    NOTE: If MILType is MI_ATTENTION or MI_GATED_ATTENTION, pooling_mode will be ignored and the implicit structure is
    close to that or CAP_MI_NET. 
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

        if self.mil_type is not MILType.CAP_MI_NET_DS:
            model = Model(inputs=inputs, outputs=last)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
                loss=self.ordinal_loss,
                metrics=self.ordinal_metrics,
            )
        else:
            out_names = [x.name.split("/", 1)[0] for x in last]  # hack-y way
            out_weights = [1.0 for _ in range(len(last) - 1)] + [0.0]

            model = tf.keras.Model(
                inputs=inputs, outputs=last, name="MI-net_corn_resnet"
            )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
                loss={i: self.ordinal_loss for i in out_names},
                loss_weights={i: j for (i, j) in zip(out_names, out_weights)},
                metrics=self.ordinal_metrics,
            )

        return model

    def input_layer(self) -> tf.keras.layers.Layer:
        return Input(shape=(None,) + self.data_set_img_size)

    def base_layers(self, layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
        if self.data_set_type in [DataSetType.FGNET, DataSetType.BCNB_ALN]:

            # Small-ish Residual Network from Vargas, Gutierrez, Hervas-Matrinez (2020) Neurocomputing
            x1 = BagWise(
                Conv2D(32, (7, 7), strides=2, padding="same", activation="relu")
            )(layer)
            x1 = BagWise(MaxPooling2D(pool_size=(3, 3), strides=2))(x1)

            x1 = bagwise_residual_block(x1, 64, (3, 3), stride=1, nonlinearity="relu")
            x1 = bagwise_residual_block(x1, 64, (3, 3), stride=1, nonlinearity="relu")

            x2 = bagwise_residual_block(x1, 128, (3, 3), stride=2, nonlinearity="relu")
            x2 = bagwise_residual_block(x2, 128, (3, 3), stride=1, nonlinearity="relu")
            x2 = bagwise_residual_block(x2, 128, (3, 3), stride=1, nonlinearity="relu")

            x3 = bagwise_residual_block(x2, 256, (3, 3), stride=2, nonlinearity="relu")
            x3 = bagwise_residual_block(x3, 256, (3, 3), stride=1, nonlinearity="relu")
            x3 = bagwise_residual_block(x3, 256, (3, 3), stride=1, nonlinearity="relu")

            x4 = bagwise_residual_block(x3, 512, (3, 3), stride=2, nonlinearity="relu")
            x4 = bagwise_residual_block(x4, 512, (3, 3), stride=1, nonlinearity="relu")
            x4 = bagwise_residual_block(x4, 512, (3, 3), stride=1, nonlinearity="relu")

            if self.mil_type is MILType.CAP_MI_NET_DS:
                x_out = [
                    BagWise(GlobalAveragePooling2D(data_format="channels_last"))(x)
                    for x in [x1, x2, x3, x4]
                ]
            else:
                x_out = BagWise(GlobalAveragePooling2D(data_format="channels_last"))(x4)

            return x_out

        if self.data_set_type is DataSetType.AES:

            # ResNet 34, as in Shi, Cao, and Raschka (2022)
            # https://www.kaggle.com/datasets/pytorch/resnet34
            x1 = BagWise(
                Conv2D(64, (7, 7), strides=2, padding="same", activation="relu")
            )(layer)
            x1 = BagWise(MaxPooling2D(pool_size=(3, 3), strides=2))(x1)

            x1 = bagwise_residual_block(x1, 64, (3, 3), stride=1, nonlinearity="relu")
            x1 = bagwise_residual_block(x1, 64, (3, 3), stride=1, nonlinearity="relu")
            x1 = bagwise_residual_block(x1, 64, (3, 3), stride=1, nonlinearity="relu")

            x2 = bagwise_residual_block(x1, 128, (3, 3), stride=2, nonlinearity="relu")
            x2 = bagwise_residual_block(x2, 128, (3, 3), stride=1, nonlinearity="relu")
            x2 = bagwise_residual_block(x2, 128, (3, 3), stride=1, nonlinearity="relu")
            x2 = bagwise_residual_block(x2, 128, (3, 3), stride=1, nonlinearity="relu")

            x3 = bagwise_residual_block(x2, 256, (3, 3), stride=2, nonlinearity="relu")
            x3 = bagwise_residual_block(x3, 256, (3, 3), stride=1, nonlinearity="relu")
            x3 = bagwise_residual_block(x3, 256, (3, 3), stride=1, nonlinearity="relu")
            x3 = bagwise_residual_block(x3, 256, (3, 3), stride=1, nonlinearity="relu")
            x3 = bagwise_residual_block(x3, 256, (3, 3), stride=1, nonlinearity="relu")
            x3 = bagwise_residual_block(x3, 256, (3, 3), stride=1, nonlinearity="relu")

            x4 = bagwise_residual_block(x3, 512, (3, 3), stride=2, nonlinearity="relu")
            x4 = bagwise_residual_block(x4, 512, (3, 3), stride=1, nonlinearity="relu")
            x4 = bagwise_residual_block(x4, 512, (3, 3), stride=1, nonlinearity="relu")

            if self.mil_type is MILType.CAP_MI_NET_DS:
                x_out = [
                    BagWise(GlobalAveragePooling2D(data_format="channels_last"))(x)
                    for x in [x1, x2, x3, x4]
                ]
                x_out = [Dense(1000, activation="relu")(x) for x in x_out]
            else:
                x_out = BagWise(GlobalAveragePooling2D(data_format="channels_last"))(x4)
                x_out = BagWise(Dense(1000, activation="relu"))(x_out)

            return x_out

        if self.data_set_type is None:

            # VGG16, pretrained, as in Xu, Zhu, Tang, et al. (2021)
            # see https://www.tensorflow.org/versions/r2.4/api_docs/python/tf/keras/applications/VGG16
            # NOTE: no adaptive average pooling to (7, 7, 512) size as in Xu, Zhu, Tang, et al. (2021)
            # Instead, we re-scale images to (224, 224, 3) from (256, 256, 3) to achieve (7, 7, 512) output

            # `preprocess_input()` expects non-scaled image
            pre = Lambda(lambda x: x * 255.0)(layer)
            pre = applications.vgg16.preprocess_input(pre)

            x1 = BagWise(applications.VGG16(include_top=False, weights="imagenet"))(pre)
            x1 = BagWise(GlobalAveragePooling2D())(x1)
            x2 = BagWise(Dense(2048))(x1)
            x3 = BagWise(Dense(1028))(x2)

            if self.mil_type is MILType.CAP_MI_NET_DS:
                x_out = [x1, x2, x3]
            else:
                x_out = x3

            return x_out

    def last_layers(self, layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
        if self.mil_type is MILType.MI_NET:
            x = self.ordinal_layer(layer)
            x = MILPool(pooling_mode=self.pooling_mode)()(x)
            return x

        if self.mil_type is MILType.CAP_MI_NET:
            x = MILPool(pooling_mode=self.pooling_mode)()(layer)
            x = self.ordinal_layer(x)
            return x

        if self.mil_type is MILType.CAP_MI_NET_DS:
            x_list = [MILPool(pooling_mode=self.pooling_mode)()(x) for x in layer]
            x_list = [self.ordinal_layer(x) for i, x in enumerate(x_list)]

            x_avg = average(x_list, name="out_avg")
            x_all = x_list + [x_avg]

            return x_all

        if self.mil_type is MILType.MI_ATTENTION:
            n_att_weights = 128
            x = mil_attention_layer(layer, n_att_weights, use_gated=False)
            x = self.ordinal_layer(x)
            return x

        if self.mil_type is MILType.MI_GATED_ATTENTION:
            n_att_weights = 128
            x = mil_attention_layer(layer, n_att_weights, use_gated=True)
            x = self.ordinal_layer(x)
            return x

    @property
    def ordinal_layer(self) -> tf.keras.layers.Layer:
        layer = {
            OrdinalType.CORAL: coral.CoralOrdinal(self.n_classes),
            OrdinalType.CORN: coral.CornOrdinal(self.n_classes),
        }
        return layer[self.ordinal_type]

    @property
    def ordinal_loss(self) -> tf.keras.layers.Layer:
        loss = {
            OrdinalType.CORAL: coral.OrdinalCrossEntropy(num_classes=self.n_classes),
            OrdinalType.CORN: coral.CornOrdinalCrossEntropy(),
        }
        return loss[self.ordinal_type]

    @property
    def ordinal_metrics(self) -> tf.keras.layers.Layer:
        # NOTE: will compute RMSE, Accuracy based on saved predictions later, because
        # coral/corn output returns 5 outputs, not 1
        mae_metric = {
            OrdinalType.CORAL: coral.MeanAbsoluteErrorLabels(),
            OrdinalType.CORN: coral.MeanAbsoluteErrorLabels(corn_logits=True),
        }
        return [mae_metric[self.ordinal_type]]

