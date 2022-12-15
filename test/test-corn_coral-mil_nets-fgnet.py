import ast
import sys

[sys.path.append(i) for i in [".", ".."]]  # need to access datasets and models module

import coral_ordinal as coral
import pandas as pd
import scipy
import tensorflow as tf
from tensorflow.keras.layers import (
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
from models.dataset import DataSet, DataSetType
from models.generators import MILImageDataGenerator

ds = DataSet(DataSetType.FGNET, name="fgnet_bag_wr")

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
    class_mode="sparse",
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
    class_mode="sparse",
    target_size=ds.params["img_size"],
)
test_generator = MILImageDataGenerator(
    dataframe=test_df,
    directory=ds.params["dir"] + "images/",
    x_col="img_name",
    y_col="age_group",
    batch_size=1,
    shuffle=True,
    class_mode="sparse",
    target_size=ds.params["img_size"],
)

n_classes = len(train_generator.class_indices)


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

# # Model 2: MI-net with softmax output (categorical)
# model = Sequential()
# model.add(Input(shape=(None,) + ds.params["img_size"]))
# model.add(BagWise(Conv2D(32, (5, 5), padding="same", activation="relu")))
# model.add(BagWise(Conv2D(32, (5, 5), activation="relu")))
# model.add(BagWise(MaxPooling2D(pool_size=(3, 3))))
# model.add(Dropout(0.25))

# model.add(BagWise(Conv2D(64, (3, 3), padding="same", activation="relu")))
# model.add(BagWise(Conv2D(64, (3, 3), activation="relu")))
# model.add(BagWise(MaxPooling2D(pool_size=(3, 3))))
# model.add(Dropout(0.25))

# model.add(BagWise(Conv2D(64, (3, 3), padding="same", activation="relu")))
# model.add(BagWise(Conv2D(64, (3, 3), activation="relu")))
# model.add(BagWise(MaxPooling2D(pool_size=(3, 3))))
# model.add(Dropout(0.25))

# model.add(BagWise(Flatten()))
# model.add(BagWise(Dense(256, activation="relu")))
# model.add(Dropout(0.5))

# model.add(MILPool(pooling_mode="max")())
# # model.add(Dense(6, activation="softmax"))  # 6 for number of ordinal labels
# # model.add(coral.CoralOrdinal(n_classes))
# model.add(coral.CornOrdinal(n_classes))

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(lr=0.05),
#     # loss=coral.OrdinalCrossEntropy(num_classes=n_classes),
#     loss=coral.CornOrdinalCrossEntropy(),
#     metrics=[coral.MeanAbsoluteErrorLabels()],
# )

# # Model 3: MI-net with softmax output (categorical), resnet blocks
inputs = Input(shape=(None,) + ds.params["img_size"])
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

outputs = MILPool(pooling_mode="max")()(x)
# outputs = coral.CoralOrdinal(n_classes)(outputs)
outputs = coral.CornOrdinal(n_classes)(outputs)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="MI-net_corn_resnet")
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.05),
    # loss=coral.OrdinalCrossEntropy(num_classes=n_classes),
    loss=coral.CornOrdinalCrossEntropy(),
    metrics=[coral.MeanAbsoluteErrorLabels()],
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
    epochs=2,
)

## Evaluate model
model.evaluate(valid_generator, steps=STEP_SIZE_VALID)

model.evaluate(test_generator, steps=STEP_SIZE_TEST)

## Predict output
# For CORAL, outputs are P(y > r_k)
# ordinal_logits = model.predict(test_generator, verbose=1)
# cum_probs = pd.DataFrame(ordinal_logits).apply(scipy.special.expit)

# For CORN, outputs are conditional probs  P(y > r_k | y > r_{k-1})
ordinal_logits = model.predict(test_generator, verbose=1)
cum_probs = pd.DataFrame(coral.corn_cumprobs(ordinal_logits))

# cum_probs.round(3).head()
# tensor_probs = coral.ordinal_softmax(ordinal_logits)
# probs_df = pd.DataFrame(tensor_probs.numpy())
# probs_df.round(3).head()

predicted_class_indices = cum_probs.apply(lambda x: x > 0.5).sum(axis=1)
# predicted_class_indices = coral.cumprobs_to_label(cum_probs, threshold=0.5)
# predicted_class_indices = pd.Series(predicted_class_indices)
# don't add 1 because we are 0 indexing

labels = train_generator.class_indices
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

results = test_df.copy()
results["predictions"] = predictions
results.to_csv(ds.params["dir"] + "results_corn_coral-mil_nets.csv", index=False)
