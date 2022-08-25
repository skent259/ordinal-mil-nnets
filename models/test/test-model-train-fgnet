import numpy as np
import pandas as pd
from models.dataset import DataSet
from tensorflow.keras import optimizers
from tensorflow.keras.layers import (
    Activation,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ds = DataSet(name="fgnet", img_size=(128, 128))

train_df = pd.read_csv(ds.dir + ds.train, dtype=str)
test_df = pd.read_csv(ds.dir + ds.test, dtype=str)


## Add generator functions
datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.25)
train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=ds.dir + "images/",
    x_col="img_name",
    y_col="age_group",
    subset="training",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=ds.img_size,
)
valid_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=ds.dir + "images/",
    x_col="img_name",
    y_col="age_group",
    subset="validation",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=ds.img_size,
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=ds.dir + "images/",
    x_col="img_name",
    y_col="age_group",
    batch_size=32,
    seed=42,
    shuffle=False,
    class_mode="categorical",
    target_size=ds.img_size,
)


## Build model
model = Sequential()
model.add(Conv2D(32, (5, 5), padding="same", input_shape=ds.img_size + (3,)))
model.add(Activation("relu"))
model.add(Conv2D(32, (5, 5)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(6, activation="softmax"))

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
    epochs=10,
)

## Evaluate model
model.evaluate(valid_generator, steps=STEP_SIZE_VALID)

model.evaluate(test_generator, steps=STEP_SIZE_TEST)

## Predict output
test_generator.reset()
pred = model.predict_generator(test_generator, verbose=1)
predicted_class_indices = np.argmax(pred, axis=1)

labels = train_generator.class_indices
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames = test_generator.filenames
results = pd.DataFrame({"id": filenames, "label": predictions})
results.to_csv(ds.dir + "results.csv", index=False)
