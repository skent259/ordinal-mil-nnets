import numpy as np
import pandas as pd
from tensorflow.keras import optimizers
from tensorflow.keras.layers import (Activation, Conv2D, Dense, Dropout,
                                     Flatten, MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def append_ext(fn):
    return fn + ".png"


def remove_ext(x):
    return x.replace(".png", "")


dataset_dir = "datasets/cifar-10/"

traindf = pd.read_csv(dataset_dir + "trainLabels.csv", dtype=str)
testdf = pd.read_csv(dataset_dir + "sampleSubmission.csv", dtype=str)
traindf["id"] = traindf["id"].apply(append_ext)
testdf["id"] = testdf["id"].apply(append_ext)


## Add generator functions
datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.25)
train_generator = datagen.flow_from_dataframe(
    dataframe=traindf,
    directory=dataset_dir + "train/",
    x_col="id",
    y_col="label",
    subset="training",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(32, 32),
)
valid_generator = datagen.flow_from_dataframe(
    dataframe=traindf,
    directory=dataset_dir + "train/",
    x_col="id",
    y_col="label",
    subset="validation",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(32, 32),
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=testdf,
    directory=dataset_dir + "test/",
    x_col="id",
    y_col=None,
    batch_size=32,
    seed=42,
    shuffle=False,
    class_mode=None,
    target_size=(32, 32),
)


## Build model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same", input_shape=(32, 32, 3)))
model.add(Activation("relu"))
model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

model.compile(
    optimizers.RMSprop(lr=0.0001), loss="categorical_crossentropy", metrics=["accuracy"]
)


## Train model
STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
model.fit(
    x=train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=valid_generator,
    validation_steps=STEP_SIZE_VALID,
    epochs=10,
)

## Evaluate model
model.evaluate_generator(generator=valid_generator, steps=STEP_SIZE_VALID)

## Predict output
test_generator.reset()
pred = model.predict_generator(test_generator, steps=STEP_SIZE_TEST, verbose=1)
predicted_class_indices = np.argmax(pred, axis=1)

labels = train_generator.class_indices
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames = test_generator.filenames
results = pd.DataFrame({"id": filenames, "label": predictions})
results["id"] = results["id"].apply(remove_ext)
results.to_csv(dataset_dir + "results.csv", index=False)
