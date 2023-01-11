import os
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import coral_ordinal as coral
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras import Model, applications
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
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

from models.clm_qwk.activations import CLM
from models.clm_qwk.losses import make_cost_matrix, qwk_loss
from models.clm_qwk.resnet import bagwise_residual_block
from models.dataset import DataSetType
from models.mil_attention.layer import mil_attention_layer
from models.mil_nets.layer import BagWise, MILPool


class OrdinalType(Enum):
    CORN = "corn"
    CORAL = "coral"
    CLM_QWK_LOGIT = "clm_qwk_logit"
    CLM_QWK_PROBIT = "clm_qwk_probit"
    CLM_QWK_CLOGLOG = "clm_qwk_cloglog"


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
    learning_rate: float = 0.05

    def build(self):
        inputs = self.input_layer()
        base = self.base_layers(inputs)
        last = self.last_layers(base)

        if self.mil_type is not MILType.CAP_MI_NET_DS:
            model = Model(inputs=inputs, outputs=last)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss=self.ordinal_loss,
                metrics=self.ordinal_metrics,
            )
        else:
            out_names = [x.name.split("/", 1)[0] for x in last]  # hack-y way
            out_weights = [1.0 for _ in range(len(last) - 1)] + [0.0]

            model = tf.keras.Model(inputs=inputs, outputs=last, name="keras_model")
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss={i: self.ordinal_loss for i in out_names},
                loss_weights={i: j for (i, j) in zip(out_names, out_weights)},
                metrics=self.ordinal_metrics,
            )

        return model

    def input_layer(self) -> tf.keras.layers.Layer:
        if self.data_set_type in [DataSetType.AMREV_TV, DataSetType.IMDB]:
            return Input(shape=(None,), dtype=tf.string, batch_size=1)
        else:
            return Input(shape=(None,) + self.data_set_img_size, batch_size=1)

    def base_layers(self, layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
        if self.data_set_type in [
            DataSetType.FGNET,
            DataSetType.BCNB_ALN,
            DataSetType.AFAD,
        ]:
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

        if self.data_set_type in [DataSetType.AMREV_TV, DataSetType.IMDB]:

            # BERT (Delvin, Chang, Lee, & Toutanova, 2018)
            # https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4

            def keras_hub_loader(folder, url, name):
                """Load KerasLayer from folder if available, with backup url"""
                if os.path.exists(folder):
                    return hub.KerasLayer(folder, name=name)
                else:
                    return hub.KerasLayer(url, name=name)

            # Load BERT from local download or web
            bert_preprocess = keras_hub_loader(
                folder="tfhub/bert_en_uncased_preprocess_3",
                url="https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
                name="bert_preprocess",
            )

            bert_encoder = keras_hub_loader(
                folder="tfhub/bert_en_uncased_L-12_H-768_A-12_4",
                url="https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",
                name="bert_encoder",
            )

            bert = tf.squeeze(layer, axis=0)  # collapse bag_size dimension
            bert = bert_preprocess(bert)
            bert = bert_encoder(bert)
            bert = tf.expand_dims(bert["pooled_output"], axis=0)  # uncollapse

            x1 = BagWise(Dense(300, activation="relu"))(bert)
            x1 = BagWise(Dropout(0.2))(x1)
            x1 = BagWise(BatchNormalization())(x1)

            x2 = BagWise(Dense(300, activation="relu"))(x1)
            x2 = BagWise(Dropout(0.2))(x2)
            x2 = BagWise(BatchNormalization())(x2)

            if self.mil_type is MILType.CAP_MI_NET_DS:
                x_out = [bert, x1, x2]
            else:
                x_out = x2

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
        def clm_layers(n_classes: int, link_function: str, use_tau: bool = True):
            _name = tf.compat.v1.get_default_graph().unique_name("dense_plus_clm")
            if self.mil_type is MILType.MI_NET:
                layers = [
                    BagWise(Dense(1)),
                    BagWise(BatchNormalization()),
                    BagWise(CLM(n_classes, link_function, use_tau)),
                ]
            else:
                layers = [
                    Dense(1),
                    BatchNormalization(),
                    CLM(n_classes, link_function, use_tau),
                ]

            return tf.keras.Sequential(layers, name=_name)

        layer = {
            OrdinalType.CORAL: coral.CoralOrdinal(self.n_classes),
            OrdinalType.CORN: coral.CornOrdinal(self.n_classes),
            OrdinalType.CLM_QWK_LOGIT: clm_layers(self.n_classes, "logit"),
            OrdinalType.CLM_QWK_PROBIT: clm_layers(self.n_classes, "probit"),
            OrdinalType.CLM_QWK_CLOGLOG: clm_layers(self.n_classes, "cloglog"),
        }
        return layer[self.ordinal_type]

    @property
    def ordinal_loss(self) -> tf.keras.layers.Layer:

        cost_matrix = tf.constant(
            make_cost_matrix(self.n_classes), dtype=tf.keras.backend.floatx()
        )

        loss = {
            OrdinalType.CORAL: coral.OrdinalCrossEntropy(num_classes=self.n_classes),
            OrdinalType.CORN: coral.CornOrdinalCrossEntropy(),
            OrdinalType.CLM_QWK_LOGIT: qwk_loss(cost_matrix),
            OrdinalType.CLM_QWK_PROBIT: qwk_loss(cost_matrix),
            OrdinalType.CLM_QWK_CLOGLOG: qwk_loss(cost_matrix),
        }
        return loss[self.ordinal_type]

    @property
    def ordinal_metrics(self) -> tf.keras.layers.Layer:
        # NOTE: will compute RMSE, Accuracy based on saved predictions later, because
        # coral/corn output returns 5 outputs, not 1
        metric = {
            OrdinalType.CORAL: coral.MeanAbsoluteErrorLabels(),
            OrdinalType.CORN: coral.MeanAbsoluteErrorLabels(corn_logits=True),
            OrdinalType.CLM_QWK_LOGIT: "accuracy",
            OrdinalType.CLM_QWK_PROBIT: "accuracy",
            OrdinalType.CLM_QWK_CLOGLOG: "accuracy",
        }
        return [metric[self.ordinal_type]]

