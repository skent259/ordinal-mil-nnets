import ast
import sys

[sys.path.append(i) for i in [".", ".."]]  # need to access datasets and models module

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)

from models.clm_qwk.activations import CLM
from models.clm_qwk.losses import make_cost_matrix, qwk_loss
from models.clm_qwk.resnet import bagwise_residual_block
from models.dataset import DataSet, DataSetType
from models.generators import MILImageDataGenerator

ds = DataSet(DataSetType.FGNET, name="fgnet_bag_wr=0.5_size=4_i=0_j=0")

train_df = pd.read_csv(ds.params["dir"] + ds.train, dtype=str, index_col=0)
train_df = train_df.applymap(ast.literal_eval)
valid_df = pd.read_csv(ds.params["dir"] + ds.valid, dtype=str, index_col=0)
valid_df = valid_df.applymap(ast.literal_eval)
test_df = pd.read_csv(ds.params["dir"] + ds.test, dtype=str, index_col=0)
test_df = test_df.applymap(ast.literal_eval)

aug_args = {
    "horizontal_flip": True,
    "crop_range": 0.1,
    "contrast_lower": 0.5,
    "contrast_upper": 2.0,
    "brightness_delta": 0.2,
    "hue_delta": 0.1,
    "quality_min": 50,
    "quality_max": 100,
}

## Add generator functions
train_generator = MILImageDataGenerator(
    dataframe=train_df,
    directory=ds.params["dir"] + "images/",
    x_col="img_name",
    y_col="age_group",
    batch_size=1,
    shuffle=True,
    class_mode="categorical",
    target_size=ds.params["img_size"],
    **aug_args
)
valid_generator = MILImageDataGenerator(
    dataframe=valid_df,
    directory=ds.params["dir"] + "images/",
    x_col="img_name",
    y_col="age_group",
    batch_size=1,
    shuffle=True,
    class_mode="categorical",
    target_size=ds.params["img_size"],
)
test_generator = MILImageDataGenerator(
    dataframe=test_df,
    directory=ds.params["dir"] + "images/",
    x_col="img_name",
    y_col="age_group",
    batch_size=1,
    shuffle=True,
    class_mode="categorical",
    target_size=ds.params["img_size"],
)

n_classes = len(train_generator.class_indices)


## Build model

# NOTE: here we take advantage of TimeDistributed to carry the bag dimension
# TODO: turn these into thin wrappers of layers
BagWise = tf.keras.layers.TimeDistributed


def MILPool(pooling_mode: str = "max"):
    switch = {
        "max": tf.keras.layers.GlobalMaxPool1D,
        "mean": tf.keras.layers.GlobalAveragePooling1D,
    }

    return switch.get(pooling_mode, "Invalid Input")


# # Model 3: MI-net with softmax output (categorical), resnet blocks
# inputs = Input(shape=(None,) + ds.params["img_size"])
# # inputs, y = train_generator[0]

# x = BagWise(Conv2D(32, (7, 7), strides=2, padding="same", activation="relu"))(inputs)
# x = BagWise(MaxPooling2D(pool_size=(3, 3), strides=2))(x)

# x = bagwise_residual_block(x, 64, (3, 3), stride=1, nonlinearity="relu")
# x = bagwise_residual_block(x, 64, (3, 3), stride=1, nonlinearity="relu")

# x = bagwise_residual_block(x, 128, (3, 3), stride=2, nonlinearity="relu")
# x = bagwise_residual_block(x, 128, (3, 3), stride=1, nonlinearity="relu")
# x = bagwise_residual_block(x, 128, (3, 3), stride=1, nonlinearity="relu")

# x = bagwise_residual_block(x, 256, (3, 3), stride=2, nonlinearity="relu")
# x = bagwise_residual_block(x, 256, (3, 3), stride=1, nonlinearity="relu")
# x = bagwise_residual_block(x, 256, (3, 3), stride=1, nonlinearity="relu")

# x = bagwise_residual_block(x, 512, (3, 3), stride=2, nonlinearity="relu")
# x = bagwise_residual_block(x, 512, (3, 3), stride=1, nonlinearity="relu")
# x = bagwise_residual_block(x, 512, (3, 3), stride=1, nonlinearity="relu")

# x = BagWise(GlobalAveragePooling2D(data_format="channels_last"))(x)

# outputs = MILPool(pooling_mode="max")()(x)
# outputs = Dense(1)(outputs)
# outputs = BatchNormalization()(outputs)
# outputs = CLM(n_classes, "logit", use_tau=True)(outputs)

# cost_matrix = tf.constant(make_cost_matrix(n_classes), dtype=tf.keras.backend.floatx())

# model = tf.keras.Model(inputs=inputs, outputs=outputs, name="MI-net_corn_resnet")
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(lr=0.05),
#     loss=qwk_loss(cost_matrix),
#     metrics=["accuracy"],
# )

# # Model 2: mi-net with softmax output (categorical), resnet blocks
inputs = Input(shape=(None,) + ds.params["img_size"], batch_size=1)
# inputs, y = train_generator[0]

x = BagWise(Conv2D(32, (7, 7), strides=2, padding="same", activation="relu"))(inputs)
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

clm = tf.keras.Sequential(
    [
        BagWise(Dense(1)),
        BagWise(BatchNormalization()),
        BagWise(CLM(n_classes, "logit", use_tau=True)),
    ],
    name="dense-plus_clm",
)

outputs = clm(x)
outputs = MILPool(pooling_mode="max")()(outputs)

cost_matrix = tf.constant(make_cost_matrix(n_classes), dtype=tf.keras.backend.floatx())

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="MI-net_corn_resnet")
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.05),
    loss=qwk_loss(cost_matrix),
    metrics=["accuracy"],
)


model.summary()

## Train model
STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
model.fit(
    x=train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=valid_generator,
    validation_steps=STEP_SIZE_VALID,
    epochs=2,
)

# ## Evaluate model
# model.evaluate(valid_generator, steps=STEP_SIZE_VALID)

# model.evaluate(test_generator, steps=STEP_SIZE_TEST)

## Predict output
predictions = model.predict(test_generator, verbose=1)
predicted_class_indices = np.argmax(predictions, axis=1)

labels = train_generator.class_indices
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

results = test_df.copy()
results["predictions"] = predictions
results.to_csv(ds.params["dir"] + "results_clm_qwk-mil_nets.csv", index=False)
