"""
See: https://www.section.io/engineering-education/classification-model-using-bert-and-tensorflow/
"""

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

df = pd.read_csv("datasets/spam/spam.csv")
df.head(5)

df["Category"].value_counts()

df_spam = df[df["Category"] == "spam"]
df_spam.shape

df_ham = df[df["Category"] == "ham"]
df_ham.shape

df_ham_downsampled = df_ham.sample(df_spam.shape[0])
df_ham_downsampled.shape

df_balanced = pd.concat([df_ham_downsampled, df_spam])
df_balanced.shape

df_balanced["Category"].value_counts()

df_balanced["spam"] = df_balanced["Category"].apply(lambda x: 1 if x == "spam" else 0)
df_balanced.sample(5)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df_balanced["Message"], df_balanced["spam"], stratify=df_balanced["spam"]
)

# bert_preprocess = hub.KerasLayer(
#     "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
# )
# bert_encoder = hub.KerasLayer(
#     "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
# )

# Load from local download
bert_preprocess = hub.KerasLayer(
    hub.load("tfhub/bert_en_uncased_preprocess_3"), name="bert_preprocess"
)
bert_encoder = hub.KerasLayer(
    hub.load("tfhub/bert_en_uncased_L-12_H-768_A-12_4"), name="bert_encoder"
)


# Bert layers
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)

# Neural network layers
l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs["pooled_output"])
l = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(l)

# Use inputs and outputs to construct a final model
model = tf.keras.Model(inputs=[text_input], outputs=[l])


model.summary()


METRICS = [
    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
]

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=METRICS)


model.fit(X_train, y_train, epochs=10)

model.evaluate(X_test, y_test)

y_predicted = model.predict(X_test)
y_predicted = y_predicted.flatten()


import numpy as np

y_predicted = np.where(y_predicted > 0.5, 1, 0)
y_predicted

sample_dataset = [
    "You can win alot of money, register in the link below",
    "You have an iphone 10, spin the image below to claim your prize and it willl be delivered in your door step",
    "You have an offer, the company will give you 50% off in every item purchased.",
    "Hey Bravin, dont be late for the meeting tomorrow, it will start at exactly 10:30am",
    "See you monday, we have alot to talk about the future of this company .",
]
model.predict(sample_dataset)
