import glob
import os
import re

import numpy as np
import pandas as pd

proj_dir = os.getcwd()
data_dir = "datasets/afad"
os.chdir(data_dir)

# make data frame with image information
img_paths = glob.glob("**/*.jpg", recursive=True)
gender_dict = {"111": "male", "112": "female"}

reg = "images\/([0-9]*)\/([0-9]*)\/([0-9\-]*).jpg"
reg_split = [re.split(reg, i) for i in img_paths]

path = [re.sub("images\/", "", i) for i in img_paths]
file = [f"{x[3]}.jpg" for x in reg_split]
age = [int(x[1]) for x in reg_split]
gender = [gender_dict.get(x[2]) for x in reg_split]

df_full = pd.DataFrame({"file": file, "path": path, "age": age, "gender": gender})


# Following Shi, Cao, and Raschka (2022), we use 13 age labels (18-30)
df = df_full.query("age >= 18 and age <= 30")


# random data splits into 80% train, 15% test, and 5% validation sets
N_DATASETS_SPLIT = 2
for i in range(N_DATASETS_SPLIT):

    df_i = df.copy().sample(frac=1, random_state=42 + i)

    train, validate, test = np.split(df_i, [int(0.8 * len(df)), int(0.85 * len(df))])
    print(len(train))
    print(len(validate))
    print(len(test))

    train.to_csv(f"splits/afad_{i=}_train.csv")
    validate.to_csv(f"splits/afad_{i=}_valid.csv")
    test.to_csv(f"splits/afad_{i=}_test.csv")


os.chdir(proj_dir)

