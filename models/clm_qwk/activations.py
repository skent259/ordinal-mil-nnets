"""
File copied from https://github.com/ayrna/deep-ordinal-clm/blob/master/src/activations.py
2022-06-12

Most content removed for clarity
"""
import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.compat.v1.distributions import Normal
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import Constant, RandomUniform
from tensorflow.math import igamma, igammac, lgamma
from tensorflow.raw_ops import MatrixBandPart


class CLM(keras.layers.Layer):
    """
	Proportional Odds Model activation layer.

    Modifications: 
    * remove everything that doesn't relate to 'logit', 'probit' or 'cloglog'
    * this removes parameter `p`, which used some parameters
    * convert to tensorflow v2.4.0
	"""

    def __init__(self, num_classes, link_function, use_tau, **kwargs):
        self.num_classes = num_classes
        self.dist = Normal(loc=0.0, scale=1.0)
        self.link_function = link_function
        self.use_tau = use_tau
        super(CLM, self).__init__(**kwargs)

    def _convert_thresholds(self, b, a):
        a = tf.pow(a, 2)
        thresholds_param = tf.concat([b, a], axis=0)
        th = tf.math.reduce_sum(
            MatrixBandPart(
                input=tf.ones([self.num_classes - 1, self.num_classes - 1]),
                num_lower=-1,
                num_upper=0,
            )
            * tf.reshape(
                tf.tile(thresholds_param, [self.num_classes - 1]),
                shape=[self.num_classes - 1, self.num_classes - 1],
            ),
            axis=1,
        )
        return th

    def _nnpom(self, projected, thresholds, m):
        # projected: [None, M]
        # thesholds: [num_classes, ]
        if self.use_tau == 1:
            projected = projected / self.tau

        # m = tf.shape(projected)[0]
        # m = self.m  # batch size

        thresh = tf.tile(tf.expand_dims(thresholds, axis=0), [m, 1])
        fx = tf.tile(projected, [1, self.num_classes - 1])
        z = thresh - fx

        if self.link_function == "probit":
            p = self.dist.cdf(z)
        elif self.link_function == "cloglog":
            p = 1 - tf.exp(-tf.exp(z))
        elif self.link_function == "logit":
            p = 1.0 / (1.0 + tf.exp(-z))

        # element i,j is p(y<=c_j) for batch i (NOTE: p(y<=c_Q) = 1))
        pcdf = tf.concat([p, tf.ones([m, 1])], axis=1)  # P()
        # go from p(y<=c_q) to p(y=c_q) through subtraction of subsequent values
        ppdf = tf.concat(
            [K.reshape(pcdf[:, 0], shape=[-1, 1]), pcdf[:, 1:] - pcdf[:, 0:-1]], axis=-1
        )

        return ppdf

    # def _nnpom_old(self, projected, thresholds):
    #     """
    #     Not used, only shown for a version of the old syntax
    #     """
    #     if self.use_tau == 1:
    #         projected = K.reshape(projected, shape=[-1]) / self.tau
    #     else:
    #         projected = K.reshape(projected, shape=[-1])

    #     m = K.shape(projected)[0]

    #     a = K.reshape(K.tile(thresholds, [m]), shape=[m, -1])
    #     b = K.transpose(
    #         K.reshape(K.tile(projected, [self.num_classes - 1]), shape=[-1, m])
    #     )
    #     z3 = a - b

    #     if self.link_function == "probit":
    #         a3T = self.dist.cdf(z3)
    #     elif self.link_function == "cloglog":
    #         a3T = 1 - K.exp(-K.exp(z3))
    #     elif self.link_function == "logit":
    #         a3T = 1.0 / (1.0 + K.exp(-z3))

    #     a3 = K.concatenate([a3T, K.ones([m, 1])], axis=1)  # P()
    #     a3 = K.concatenate(
    #         [K.reshape(a3[:, 0], shape=[-1, 1]), a3[:, 1:] - a3[:, 0:-1]], axis=-1
    #     )

    #     return a3

    def build(self, input_shape):
        self.thresholds_b = self.add_weight(
            "b_b_nnpom", shape=(1,), initializer=RandomUniform(minval=0, maxval=0.1),
        )
        self.thresholds_a = self.add_weight(
            "b_a_nnpom",
            shape=(self.num_classes - 2,),
            initializer=RandomUniform(
                minval=math.sqrt((1.0 / (self.num_classes - 2)) / 2),
                maxval=math.sqrt(1.0 / (self.num_classes - 2)),
            ),
        )

        if self.use_tau == 1:
            self.tau = self.add_weight(
                "tau_nnpom", shape=(1,), initializer=RandomUniform(minval=1, maxval=10),
            )
            self.tau = K.clip(self.tau, 1, 1000)
        # tf.TensorShape([None, 1])
        # self.m = tf.shape(input_shape)[0]
        # self.one_vec = tf.ones(self.m, dtype=tf.float32)[..., None]
        # self.one_vec = tf.Variable(one_vec, trainable=False, validate_shape=True)
        # self.m = tf.shape(input_shape)[0]

    def call(self, x, **kwargs):
        # TODO: maybe someone smarter than I can figure out how to
        # set this up with variable batch size. I've looked at these stackoverflow:
        # https://stackoverflow.com/questions/56101920/get-batch-size-in-keras-custom-layer-and-use-tensorflow-operations-tf-variable
        # https://stackoverflow.com/questions/70421012/how-to-define-a-new-tensor-with-a-dynamic-shape-to-support-batching-in-a-custom
        # https://github.com/tensorflow/tensorflow/blob/v2.4.0/tensorflow/python/keras/layers/wrappers.py#L86-L330
        batch_size = 1
        thresholds = self._convert_thresholds(self.thresholds_b, self.thresholds_a)
        return self._nnpom(x, thresholds, m=batch_size)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1].concatenate(self.num_classes)
        # return (input_shape[:-1], self.num_classes)

    def get_config(self):

        config = super().get_config().copy()
        config.update(
            {
                "num_classes": self.num_classes,
                "link_function": self.link_function,
                "use_tau": self.use_tau,
            }
        )
        return config
