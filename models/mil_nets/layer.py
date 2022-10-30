import tensorflow as tf

# TODO: Set these up as thin wrappers of Keras layers for easier access

BagWise = tf.keras.layers.TimeDistributed

# class BagWise(tf.keras.layers.TimeDistributed):
#     def __init__(self, *args, **kwargs):
#         super().__init__(self, *args, **kwargs)


def MILPool(pooling_mode: str = "max"):
    layer = {
        "max": tf.keras.layers.GlobalMaxPool1D,
        "mean": tf.keras.layers.GlobalAveragePooling1D,
    }

    return layer[pooling_mode]
