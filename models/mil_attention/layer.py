"""
File copied from https://keras.io/examples/vision/attention_mil_classification/
2022-11-27
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def mil_attention_layer(
    in_layer,
    weight_params_dim,
    kernel_initializer="glorot_uniform",
    kernel_regularizer=keras.regularizers.l2(0.01),
    use_gated=False,
):
    """
    My re-implementation of the attention-based Deep MIL layer.

    Args:
      in_layer: keras.Layer instance near the end of the model. Expecting shape `(batch_size, bag_size, input_dim)`
      weight_params_dim: Positive Integer. Dimension of the weight matrix.
      kernel_initializer: Initializer for the `kernel` matrix.
      kernel_regularizer: Regularizer function applied to the `kernel` matrix.
      use_gated: Boolean, whether or not to use the gated mechanism.

    Returns:
      Keras model applied to the `in_layer`. Its output shape will be `(batch_size, input_dim)`
      The output is the attention scores multiplied by the input 

    This follows more closely to https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py. 
    Note that the `layers.dot` at the end does NOT do concatentation, as MILAttentionLayer does. Instead, it adds the 
    layers, which matches the layer in 
    Ilse, M., Tomczak, J., & Welling, M. (2018). Attention-based Deep Multiple Instance Learning. Proceedings of the 35th International Conference on Machine Learning, 2127-2136. https://proceedings.mlr.press/v80/ilse18a.html

    Followed some ideas from here to get custom name: https://keras.io/guides/functional_api/#all-models-are-callable-just-like-layers
    """
    BagWise = tf.keras.layers.TimeDistributed

    input = layers.Input(shape=in_layer.shape[1:])

    if not use_gated:
        att = BagWise(
            layers.Dense(
                weight_params_dim,
                use_bias=False,
                activation="tanh",
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )
        )(input)
    else:
        att_U = BagWise(
            layers.Dense(
                weight_params_dim,
                use_bias=False,
                activation="tanh",
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )
        )(input)
        att_V = BagWise(
            layers.Dense(
                weight_params_dim,
                use_bias=False,
                activation="sigmoid",
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )
        )(input)

        att = tf.keras.layers.multiply([att_U, att_V])

    att = BagWise(
        layers.Dense(
            1,
            use_bias=False,
            activation=None,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
        )
    )(att)
    att = layers.Flatten()(att)
    att = layers.Softmax()(att)
    output = layers.dot([input, att], axes=[1, 1])

    _name = tf.compat.v1.get_default_graph().unique_name("mil_attention_layer")
    # TODO: implement a way to pull the attention layer from this...
    return keras.Model(input, output, name=_name)(in_layer)


# Code below not used ---------------------------------------------------------


class MILAttentionLayer(layers.Layer):
    """Implementation of the attention-based Deep MIL layer.

    Args:
      weight_params_dim: Positive Integer. Dimension of the weight matrix.
      kernel_initializer: Initializer for the `kernel` matrix.
      kernel_regularizer: Regularizer function applied to the `kernel` matrix.
      use_gated: Boolean, whether or not to use the gated mechanism.

    Returns:
      List of 2D tensors with BAG_SIZE length.
      The tensors are the attention scores after softmax with shape `(batch_size, 1)`.
    """

    def __init__(
        self,
        weight_params_dim,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        use_gated=False,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.weight_params_dim = weight_params_dim
        self.use_gated = use_gated

        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        self.v_init = self.kernel_initializer
        self.w_init = self.kernel_initializer
        self.u_init = self.kernel_initializer

        self.v_regularizer = self.kernel_regularizer
        self.w_regularizer = self.kernel_regularizer
        self.u_regularizer = self.kernel_regularizer

    def build(self, input_shape):

        # Input shape.
        # List of 2D tensors with shape: (batch_size, input_dim).
        input_dim = input_shape[0][1]

        self.v_weight_params = self.add_weight(
            shape=(input_dim, self.weight_params_dim),
            initializer=self.v_init,
            name="v",
            regularizer=self.v_regularizer,
            trainable=True,
        )

        self.w_weight_params = self.add_weight(
            shape=(self.weight_params_dim, 1),
            initializer=self.w_init,
            name="w",
            regularizer=self.w_regularizer,
            trainable=True,
        )

        if self.use_gated:
            self.u_weight_params = self.add_weight(
                shape=(input_dim, self.weight_params_dim),
                initializer=self.u_init,
                name="u",
                regularizer=self.u_regularizer,
                trainable=True,
            )
        else:
            self.u_weight_params = None

        self.input_built = True

    def call(self, inputs):

        # Assigning variables from the number of inputs.
        instances = [self.compute_attention_scores(instance) for instance in inputs]

        # Apply softmax over instances such that the output summation is equal to 1.
        alpha = tf.math.softmax(instances, axis=0)

        return [alpha[i] for i in range(alpha.shape[0])]

    def compute_attention_scores(self, instance):

        # Reserve in-case "gated mechanism" used.
        original_instance = instance

        # tanh(v*h_k^T)
        instance = tf.math.tanh(tf.tensordot(instance, self.v_weight_params, axes=1))

        # for learning non-linear relations efficiently.
        if self.use_gated:

            instance = instance * tf.math.sigmoid(
                tf.tensordot(original_instance, self.u_weight_params, axes=1)
            )

        # w^T*(tanh(v*h_k^T)) / w^T*(tanh(v*h_k^T)*sigmoid(u*h_k^T))
        return tf.tensordot(instance, self.w_weight_params, axes=1)


def create_model(instance_shape):
    BAG_SIZE = 3
    # Extract features from inputs.
    inputs, embeddings = [], []
    shared_dense_layer_1 = layers.Dense(128, activation="relu")
    shared_dense_layer_2 = layers.Dense(64, activation="relu")
    for _ in range(BAG_SIZE):
        inp = layers.Input(instance_shape)
        flatten = layers.Flatten()(inp)
        dense_1 = shared_dense_layer_1(flatten)
        dense_2 = shared_dense_layer_2(dense_1)
        inputs.append(inp)
        embeddings.append(dense_2)

    # Invoke the attention layer.
    alpha = MILAttentionLayer(
        weight_params_dim=256,
        kernel_regularizer=keras.regularizers.l2(0.01),
        use_gated=True,
        name="alpha",
    )(embeddings)

    # Multiply attention weights with the input layers.
    multiply_layers = [
        layers.multiply([alpha[i], embeddings[i]]) for i in range(len(alpha))
    ]

    # Concatenate layers.
    concat = layers.concatenate(multiply_layers, axis=1)

    # Classification output node.
    output = layers.Dense(2, activation="softmax")(concat)

    return keras.Model(inputs, output)
