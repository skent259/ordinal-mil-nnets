import os

import numpy as np
import pandas as pd

proj_dir = os.getcwd()
data_dir = "datasets/aes"
os.chdir(data_dir)

df = pd.read_csv("aes.csv")

# split data into train, test, and validation sets
train, validate, test = np.split(
    df.sample(frac=1, random_state=42), [int(0.75 * len(df)), int(0.80 * len(df))]
)
print(len(train))
print(len(validate))
print(len(test))

train.to_csv("splits/aes_train.csv")
validate.to_csv("splits/aes_valid.csv")
test.to_csv("splits/aes_test.csv")

os.chdir(proj_dir)

