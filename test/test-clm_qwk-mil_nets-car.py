import ast
import sys
from typing import Dict, List

[sys.path.append(i) for i in [".", ".."]]  # need to access datasets and models module

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

from models.clm_qwk.activations import CLM
from models.clm_qwk.losses import make_cost_matrix, qwk_loss
from models.dataset import DataSet, DataSetType
from models.generators import MILTabularDataGenerator

df = pd.read_csv("datasets/car/car-mil-or_wr=1_i=1.csv")

# TODO: fix this splitting to be based on bag...
bags = df["b"].unique()
np.random.shuffle(bags)
train, valid, test = np.split(bags, [int(0.8 * len(bags)), int(0.8 * len(bags))])

train_df = df.loc[
    df["b"].isin(train),
]
valid_df = df.loc[
    df["b"].isin(valid),
]
test_df = df.loc[
    df["b"].isin(test),
]


train_generator = MILTabularDataGenerator(
    dataframe=train_df,
    x_cols=list(df.columns[3:]),
    y_col="y",
    bag_col="b",
    batch_size=1,
    shuffle=True,
    class_mode="categorical",
)
valid_generator = MILTabularDataGenerator(
    dataframe=valid_df,
    x_cols=list(df.columns[3:]),
    y_col="y",
    bag_col="b",
    batch_size=1,
    shuffle=True,
    class_mode="categorical",
    class_indices=train_generator.class_indices,
)
test_generator = MILTabularDataGenerator(
    dataframe=test_df,
    x_cols=list(df.columns[3:]),
    y_col="y",
    bag_col="b",
    batch_size=1,
    shuffle=True,
    class_mode="categorical",
    class_indices=train_generator.class_indices,
)

n_classes = len(train_generator.class_indices)


## Build model

# NOTE: here we take advantage of TimeDistributed to carry the bag dimension
# TODO: turn these into thin wrappers of layers
BagWise = layers.TimeDistributed


def MILPool(pooling_mode: str = "max"):
    switch = {
        "max": layers.GlobalMaxPool1D,
        "mean": layers.GlobalAveragePooling1D,
    }

    return switch.get(pooling_mode, "Invalid Input")


# # Model 3: MI-net with softmax output (categorical), resnet blocks
inputs = layers.Input(shape=(None, 16))
# inputs, y = train_generator[0]

x = BagWise(layers.Dense(300))(inputs)
x = BagWise(layers.LeakyReLU())(x)
x = BagWise(layers.Dropout(0.2))(x)
x = BagWise(layers.BatchNormalization())(x)

x = BagWise(layers.Dense(200))(x)
x = BagWise(layers.LeakyReLU())(x)
x = BagWise(layers.Dropout(0.2))(x)
x = BagWise(layers.BatchNormalization())(x)

clm_layers = [
    layers.Dense(1),
    layers.BatchNormalization(),
    CLM(n_classes, "logit", use_tau=True),
]

outputs = MILPool(pooling_mode="max")()(x)
outputs = tf.keras.Sequential(clm_layers, name="dense-plus_clm")(outputs)

cost_matrix = tf.constant(make_cost_matrix(n_classes), dtype=tf.keras.backend.floatx())

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="MI-net_corn_resnet")
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.01),
    loss=qwk_loss(cost_matrix),
    metrics=["accuracy"],
)

model.summary()

# # Model 2: mi-net with softmax output (categorical), resnet blocks
# inputs = layers.Input(shape=(None, 16))
# # inputs, y = train_generator[0]

# x = BagWise(layers.Dense(300))(inputs)
# x = BagWise(layers.LeakyReLU())(x)
# x = BagWise(layers.Dropout(0.2))(x)
# x = BagWise(layers.BatchNormalization())(x)

# x = BagWise(layers.Dense(200))(x)
# x = BagWise(layers.LeakyReLU())(x)
# x = BagWise(layers.Dropout(0.2))(x)
# x = BagWise(layers.BatchNormalization())(x)

# clm = tf.keras.Sequential(
#     [
#         BagWise(Dense(1)),
#         BagWise(BatchNormalization()),
#         BagWise(CLM(n_classes, "logit", use_tau=True)),
#     ],
#     name="dense-plus_clm",
# )

# outputs = clm(x)
# outputs = MILPool(pooling_mode="max")()(outputs)

# cost_matrix = tf.constant(make_cost_matrix(n_classes), dtype=tf.keras.backend.floatx())

# model = tf.keras.Model(inputs=inputs, outputs=outputs, name="MI-net_corn_resnet")
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(lr=0.05),
#     loss=qwk_loss(cost_matrix),
#     metrics=["accuracy"],
# )

# model.summary()

## Train model
STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
model.fit(
    x=train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=test_generator,
    validation_steps=STEP_SIZE_TEST,
    epochs=100,
)

# ## Evaluate model
# model.evaluate(valid_generator, steps=STEP_SIZE_VALID)

model.evaluate(test_generator, steps=STEP_SIZE_TEST)

## Predict output
predictions = model.predict(test_generator, verbose=1)
predicted_class_indices = np.argmax(predictions, axis=1)

labels = train_generator.class_indices
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

results = test_df[["y", "b"]].copy().drop_duplicates()
results["predictions"] = predictions
results
# results.to_csv("car/results_clm_qwk-mil_nets.csv", index=False)
