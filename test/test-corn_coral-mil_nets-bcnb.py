import ast
import sys

[sys.path.append(i) for i in [".", ".."]]  # need to access datasets and models module

import coral_ordinal as coral
import pandas as pd
import scipy
import tensorflow as tf
from tensorflow.keras import layers
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

ds = DataSet(DataSetType.BCNB_ALN, name="bcnb_aln_i=0")

train_df = pd.read_csv(ds.params["dir"] + ds.train, dtype=str, index_col=0)
train_df = train_df.applymap(ast.literal_eval)
valid_df = pd.read_csv(ds.params["dir"] + ds.valid, dtype=str, index_col=0)
valid_df = valid_df.applymap(ast.literal_eval)
test_df = pd.read_csv(ds.params["dir"] + ds.test, dtype=str, index_col=0)
test_df = test_df.applymap(ast.literal_eval)

aug_args = {}

# small testing

## Add generator functions
train_generator = MILImageDataGenerator(
    dataframe=train_df,
    directory=ds.params["dir"] + "images/",
    x_col="img_name",
    y_col="aln_status",
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
    y_col="aln_status",
    batch_size=1,
    shuffle=True,
    class_mode="sparse",
    target_size=ds.params["img_size"],
)
test_generator = MILImageDataGenerator(
    dataframe=test_df,
    directory=ds.params["dir"] + "images/",
    x_col="img_name",
    y_col="aln_status",
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


# Model 1: MI-net with softmax output (categorical), VGG16
inputs = Input(shape=(None,) + ds.params["img_size"])

pre = tf.keras.layers.Lambda(lambda x: x * 255.0)(inputs)
pre = tf.keras.applications.vgg16.preprocess_input(pre)

x1 = BagWise(tf.keras.applications.VGG16(include_top=False, weights="imagenet"))(pre)
# NOTE: no adaptive average pooling to (7, 7, 512) size as in Xu, Zhu, Tang, et al. (2021)
x1 = BagWise(GlobalAveragePooling2D())(x1)
x2 = BagWise(Dense(2048))(x1)
x3 = BagWise(Dense(1028))(x2)

outputs = MILPool(pooling_mode="max")()(x3)
outputs = coral.CornOrdinal(n_classes)(outputs)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="MI-net_corn_vgg16")
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.001),
    # loss=coral.OrdinalCrossEntropy(num_classes=n_classes),
    loss=coral.CornOrdinalCrossEntropy(),
    metrics=[coral.MeanAbsoluteErrorLabels()],
)

model.summary()

# Model 2: MI-net with softmax output (categorical), resnet blocks
# inputs = Input(shape=(None,) + ds.params["img_size"])
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
# # outputs = coral.CoralOrdinal(n_classes)(outputs)
# outputs = coral.CornOrdinal(n_classes)(outputs)

# model = tf.keras.Model(inputs=inputs, outputs=outputs, name="MI-net_corn_resnet")
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(lr=0.05),
#     # loss=coral.OrdinalCrossEntropy(num_classes=n_classes),
#     loss=coral.CornOrdinalCrossEntropy(),
#     metrics=[coral.MeanAbsoluteErrorLabels()],
# )

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
# model.evaluate(valid_generator, steps=STEP_SIZE_VALID)

# model.evaluate(test_generator, steps=STEP_SIZE_TEST)

# X, y = test_generator[0]
# layer_name = "time_distributed_16"
# intermediate_layer_model = tf.keras.Model(
#     inputs=model.input,
#     outputs=model.get_layer(layer_name).output
# )
# intermediate_output = intermediate_layer_model.predict(X)

# print(intermediate_output)

# model.summary()

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
