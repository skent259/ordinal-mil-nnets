import os

import numpy as np
import pandas as pd

proj_dir = os.getcwd()
data_dir = "datasets/bcnb"
os.chdir(data_dir)


def random_under_sampler(df, y_col: str, limit: int = float("inf")):
    """
    Down-sample a data set to be split evenly by `y_col`.
    """

    df = df.copy()
    classes = np.unique(df[y_col])
    class_indices = dict(zip(classes, np.arange(len(classes))))

    y = df[y_col].apply(lambda x: class_indices[x]).reset_index(drop=True)

    low_n = min(y.value_counts())
    low_n = min(low_n, limit)

    def boostrap_sample(x: pd.Series, n: int):
        return x.sample(n=n, replace=True).index

    samples = [boostrap_sample(y[y == i], low_n) for i in range(len(classes))]
    samples_flat = [i for l in samples for i in l]

    return df.iloc[samples_flat].sample(frac=1)


# Read in clinical data
clin = pd.read_excel("patient-clinical-data.xlsx")

aln_names = {"N0": "N0", "N+(1-2)": "N1-2", "N+(>2)": "N2+"}  # need proper ordering
clin["ALN status 2"] = [aln_names[x] for x in clin["ALN status"]]

# Read in paper splits; set up 4 additional splits
# Append clinical information to the splits

N_DATASETS_SPLIT = 5

train, validate, test = [], [], []

paper_split_files = ["train_id.txt", "val_id.txt", "test_id.txt"]
paper_splits = [pd.read_table(f"splits/{i}", header=None) for i in paper_split_files]

train.append(clin.loc[paper_splits[0][0] - 1])
validate.append(clin.loc[paper_splits[1][0] - 1])
test.append(clin.loc[paper_splits[2][0] - 1])


for i in range(N_DATASETS_SPLIT - 1):
    df_i = clin.copy().sample(frac=1, random_state=42 - i)

    # 6:2:2 train:test:valid, based on website, but it's not exact

    train_i, validate_i, test_i = np.split(
        df_i, [int(0.596 * len(df_i)), int(0.794 * len(df_i))]
    )
    train.append(train_i)
    validate.append(validate_i)
    test.append(test_i)

# print([len(x) for x in train])
# print([len(x) for x in validate])
# print([len(x) for x in test])


# Find names of all images and parse the subject, other information
patches = pd.Series(os.listdir("images"), name="img_name")
subjects = patches.str.extract("([0-9]*)_.*.jpg")
subjects.rename(columns={0: "subject"}, inplace=True)

image_info = pd.concat([subjects, patches], axis=1)
image_info["subject"].value_counts()

# img_by_subject = image_info.groupby(["subject"])["img_name"].apply(list)
img_by_subject = image_info.groupby(["subject"]).agg(list)
img_by_subject["Patient ID"] = img_by_subject.index.astype(int)

# DECISION: Reduce down to at most 30 images in the bag...
# This may miss some, but the original paper does averaging at the end anyways, so this
# is probably okay
n_bag_max = 30
img_by_subject["img_name"] = img_by_subject["img_name"].apply(
    lambda x: x[0 : min(len(x), n_bag_max)]
)
assert img_by_subject["img_name"].apply(len).max() <= 50


# img_by_subject["img_name"].apply(len).max()
# Append columns for image information to the splits
cols = ["Patient ID", "ALN status 2", "Histological grading"]
train = [pd.merge(x[cols], img_by_subject, on="Patient ID") for x in train]
validate = [pd.merge(x[cols], img_by_subject, on="Patient ID") for x in validate]
test = [pd.merge(x[cols], img_by_subject, on="Patient ID") for x in test]


# Save output

for key, dfs in {"train": train, "valid": validate, "test": test}.items():

    for i, x in enumerate(dfs):
        new_cols = {
            "Patient ID": "subject",
            "ALN status 2": "aln_status",
            "Histological grading": "hist_grade",
        }
        x.rename(columns=new_cols, inplace=True)

        # turn to hard strings so can read with `ast.literal_eval`
        for col in new_cols.values():
            x[col] = x[col].apply(lambda x: f'"{x}"')

        y = x.loc[pd.notnull(x["hist_grade"])].copy()

        if key == "train":
            x_aln = random_under_sampler(x, "aln_status")
            y_hist = random_under_sampler(y, "hist_grade")
        elif key == "valid":
            x_aln = x.sample(n=50, ignore_index=True).copy()
            y_hist = y.sample(n=50, ignore_index=True).copy()
        elif key == "test":
            x_aln = x.copy()
            y_hist = y.copy()

        x_aln.to_csv(f"splits_bag/bcnb_aln_{i=}_{key}.csv")
        y_hist.to_csv(f"splits_bag/bcnb_hist_{i=}_{key}.csv")

        # super small data set for quick testing, can delete later
        if i == 0:
            x_aln = x_aln.sample(n=3, ignore_index=True)
            x_aln.to_csv(f"splits_bag/bcnb_aln_tiny_{i=}_{key}.csv")
            y_hist = y_hist.sample(n=3, ignore_index=True)
            y_hist.to_csv(f"splits_bag/bcnb_hist_tiny_{i=}_{key}.csv")


os.chdir(proj_dir)


# To read in:
# import ast
# tmp = pd.read_csv(out_fname, index_col=0).applymap(ast.literal_eval)
