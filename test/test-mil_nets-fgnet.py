import ast
import sys

[sys.path.append(i) for i in [".", ".."]]  # need to access datasets and models module

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import (
    Activation,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    GlobalMaxPool1D,
    Input,
    MaxPooling2D,
    TimeDistributed,
)
from tensorflow.keras.models import Sequential

from models.dataset import DataSet
from models.generators import MILImageDataGenerator

ds = DataSet(name="fgnet_bag", dir="datasets/fgnet/", img_size=(128, 128))

train_df = pd.read_csv(ds.dir + ds.train, dtype=str, index_col=0)
train_df = train_df.applymap(ast.literal_eval)
valid_df = pd.read_csv(ds.dir + ds.valid, dtype=str, index_col=0)
valid_df = valid_df.applymap(ast.literal_eval)
test_df = pd.read_csv(ds.dir + ds.test, dtype=str, index_col=0)
test_df = test_df.applymap(ast.literal_eval)


## Add generator functions
train_generator = MILImageDataGenerator(
    dataframe=train_df,
    directory=ds.dir + "images/",
    x_col="img_name",
    y_col="age_group",
    batch_size=1,
    shuffle=True,
    target_size=ds.img_size,
)
valid_generator = MILImageDataGenerator(
    dataframe=valid_df,
    directory=ds.dir + "images/",
    x_col="img_name",
    y_col="age_group",
    batch_size=1,
    shuffle=True,
    target_size=ds.img_size,
)
test_generator = MILImageDataGenerator(
    dataframe=test_df,
    directory=ds.dir + "images/",
    x_col="img_name",
    y_col="age_group",
    batch_size=1,
    shuffle=True,
    target_size=ds.img_size,
)


## Build model

# NOTE: here we take advantage of TimeDistributed to carry the bag dimension
BagWise = tf.keras.layers.TimeDistributed


def MILPool(pooling_mode: str = "max"):
    switch = {
        "max": tf.keras.layers.GlobalMaxPool1D,
        "mean": tf.keras.layers.GlobalAveragePooling1D,
    }

    return switch.get(pooling_mode, "Invalid Input")


# TODO: turn these into thin wrappers of layers

# Model 1: mi-net with softmax output (categorial)
# model = Sequential()
# model.add(Input(shape=(None,) + ds.img_size + (3,)))
# model.add(BagWise(Conv2D(32, (5, 5), padding="same", activation="relu")))
# model.add(BagWise(Conv2D(32, (5, 5), activation="relu")))
# model.add(BagWise(MaxPooling2D(pool_size=(3, 3))))
# model.add(Dropout(0.25))

# model.add(BagWise(Conv2D(64, (3, 3), padding="same", activation="relu")))
# model.add(BagWise(Conv2D(64, (3, 3), activation="relu")))
# model.add(BagWise(MaxPooling2D(pool_size=(3, 3))))
# model.add(Dropout(0.25))

# model.add(BagWise(Flatten()))
# model.add(BagWise(Dense(512, activation="relu")))
# model.add(Dropout(0.5))

# model.add(Dense(6, activation="softmax"))  # 6 for number of ordinal labels
# model.add(GlobalMaxPool1D())

# model.compile(
#     optimizers.RMSprop(lr=0.0001), loss="categorical_crossentropy", metrics=["accuracy"]
# )

# Model 2: MI-net with softmax output (categorical)
model = Sequential()
model.add(Input(shape=(None,) + ds.img_size + (3,)))
model.add(BagWise(Conv2D(32, (5, 5), padding="same", activation="relu")))
model.add(BagWise(Conv2D(32, (5, 5), activation="relu")))
model.add(BagWise(MaxPooling2D(pool_size=(3, 3))))
model.add(Dropout(0.25))

model.add(BagWise(Conv2D(64, (3, 3), padding="same", activation="relu")))
model.add(BagWise(Conv2D(64, (3, 3), activation="relu")))
model.add(BagWise(MaxPooling2D(pool_size=(3, 3))))
model.add(Dropout(0.25))

model.add(BagWise(Conv2D(64, (3, 3), padding="same", activation="relu")))
model.add(BagWise(Conv2D(64, (3, 3), activation="relu")))
model.add(BagWise(MaxPooling2D(pool_size=(3, 3))))
model.add(Dropout(0.25))

model.add(BagWise(Flatten()))
model.add(BagWise(Dense(256, activation="relu")))
model.add(Dropout(0.5))

model.add(MILPool(pooling_mode="max")())
model.add(Dense(6, activation="softmax"))  # 6 for number of ordinal labels

model.compile(
    optimizers.RMSprop(lr=0.0001), loss="categorical_crossentropy", metrics=["accuracy"]
)

# model.summary()


## Train model
STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
model.fit(
    x=train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=valid_generator,
    validation_steps=STEP_SIZE_VALID,
    epochs=3,
)

## Evaluate model
model.evaluate(valid_generator, steps=STEP_SIZE_VALID)

model.evaluate(test_generator, steps=STEP_SIZE_TEST)

## Predict output
pred = model.predict(test_generator, verbose=1)
predicted_class_indices = np.argmax(pred, axis=1)

labels = train_generator.class_indices
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

results = test_df.copy()
# results["y_true"] = results.age_group.apply(max)
results["predictions"] = predictions
results.to_csv(ds.dir + "results_mil_nets.csv", index=False)
