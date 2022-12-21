import ast
import sys

[sys.path.append(i) for i in [".", ".."]]  # need to access datasets and models module

import coral_ordinal as coral
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras import layers

from models.dataset import DataSet, DataSetType
from models.generators import MILTextDataGenerator
from models.mil_nets import wrappers

ds = DataSet(DataSetType.AMREV_TV, name="amrev_TVs_i=0")

train_df = pd.read_csv(ds.params["dir"] + ds.train, dtype=str, index_col=0)
train_df = train_df.applymap(ast.literal_eval)
valid_df = pd.read_csv(ds.params["dir"] + ds.valid, dtype=str, index_col=0)
valid_df = valid_df.applymap(ast.literal_eval)
test_df = pd.read_csv(ds.params["dir"] + ds.test, dtype=str, index_col=0)
test_df = test_df.applymap(ast.literal_eval)

train_df = train_df.sample(n=100)
valid_df = valid_df.sample(n=20)
test_df = test_df.sample(n=200)

train_generator = MILTextDataGenerator(
    dataframe=train_df,
    x_col="review",
    y_col="rating",
    batch_size=1,
    shuffle=True,
    class_mode="categorical",
)
valid_generator = MILTextDataGenerator(
    dataframe=valid_df,
    x_col="review",
    y_col="rating",
    batch_size=1,
    shuffle=True,
    class_mode="categorical",
    class_indices=train_generator.class_indices,
)
test_generator = MILTextDataGenerator(
    dataframe=test_df,
    x_col="review",
    y_col="rating",
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
# BagWise = wrappers.TimeDistributed


def MILPool(pooling_mode: str = "max"):
    switch = {
        "max": layers.GlobalMaxPool1D,
        "mean": layers.GlobalAveragePooling1D,
    }

    return switch.get(pooling_mode, "Invalid Input")


# Load BERT from local download
bert_preprocess = hub.KerasLayer(
    hub.load("tfhub/bert_en_uncased_preprocess_3"), name="bert_preprocess"
)
bert_encoder = hub.KerasLayer(
    hub.load("tfhub/bert_en_uncased_L-12_H-768_A-12_4"), name="bert_encoder"
)


# Model 1: BERT base, with MI-net and CORN final layers
inputs = layers.Input(shape=(None,), dtype=tf.string, name="text_input")

bert = tf.squeeze(inputs, axis=0)  # collapse bag_size dimension
bert = bert_preprocess(bert)
bert = bert_encoder(bert)
bert = tf.expand_dims(bert["pooled_output"], axis=0)  # uncollapse bag_size dimension

x = BagWise(layers.Dense(300, activation="relu"))(bert)
x = BagWise(layers.Dropout(0.2))(x)
x = BagWise(layers.BatchNormalization())(x)

# TODO: add a few layers for deep supervision

outputs = MILPool(pooling_mode="max")()(x)
outputs = coral.CornOrdinal(n_classes)(outputs)

# Use inputs and outputs to construct a final model
model = tf.keras.Model(inputs=inputs, outputs=outputs, name="MI-net_corn_bert")

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.05),
    loss=coral.CornOrdinalCrossEntropy(),
    metrics=[coral.MeanAbsoluteErrorLabels()],
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
model.evaluate(test_generator, steps=STEP_SIZE_TEST)


## Predict output
# For CORAL, outputs are P(y > r_k)
# ordinal_logits = model.predict(test_generator, verbose=1)
# cum_probs = pd.DataFrame(ordinal_logits).apply(scipy.special.expit)

# For CORN, outputs are conditional probs  P(y > r_k | y > r_{k-1})
ordinal_logits = model.predict(test_generator, verbose=1)
cum_probs = pd.DataFrame(coral.corn_cumprobs(ordinal_logits))

predicted_class_indices = cum_probs.apply(lambda x: x > 0.5).sum(axis=1)
# predicted_class_indices = coral.cumprobs_to_label(cum_probs, threshold=0.5)
# predicted_class_indices = pd.Series(predicted_class_indices)
# don't add 1 because we are 0 indexing

labels = train_generator.class_indices
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

results = test_df.copy()
results["predictions"] = predictions
results
# results.to_csv(ds.params["dir"] + "results_corn_coral-mil_nets.csv", index=False)
