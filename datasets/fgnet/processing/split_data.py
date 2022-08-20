import os
import re

import numpy as np
import pandas as pd

proj_dir = os.getcwd()
data_dir = "datasets/fgnet"
os.chdir(data_dir)

# make data frame with image information
img_files = os.listdir("images")
reg_split = [re.split("([0-9]*)A([0-9]*).*.JPG", i) for i in img_files]
subject = [x[1] for x in reg_split]
age = np.array([int(x[2]) for x in reg_split])

# (0−3, 4−11, 12−16, 17−24, 25−40, > 40).
age_group = np.where(
    age <= 3,
    "00-03",
    np.where(
        age <= 11,
        "04-11",
        np.where(
            age <= 16,
            "12-16",
            np.where(age <= 24, "17-24", np.where(age <= 40, "25-40", "41+")),
        ),
    ),
)

df = pd.DataFrame(
    {"subject": subject, "age": age, "age_group": age_group}, index=img_files
)

# split data into train, test, and validation sets
train, validate, test = np.split(
    df.sample(frac=1, random_state=42), [int(0.8 * len(df)), int(0.85 * len(df))]
)
print(len(train))
print(len(validate))
print(len(test))

train.to_csv("fgnet_train.csv")
validate.to_csv("fgnet_valid.csv")
test.to_csv("fgnet_test.csv")

os.chdir(proj_dir)

