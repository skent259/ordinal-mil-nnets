"""
To retrieve the data, users need to follow the instructions at the following link:
https://bupt-ai-cz.github.io/BCNB/

Download to datasets/bcnb/, then you should have the following files

datasets/bcnb/
├── dataset-splitting
│   ├── test_id.txt
│   ├── train_id.txt
│   └── val_id.txt
├── WSIs
│   ├── 1.jpg
│   ├── 1.json
│   ├── ...
│   ├── 1058.jpg
│   └── 1058.json
├── paper_patches.zip
├── patient-clinical-data.xlsx
└── README.txt

Then run this file to set up the right folder structure
"""

import os
import shutil

proj_dir = os.getcwd()
data_dir = "datasets/bcnb"
os.chdir(data_dir)

# unzip and extract images
shutil.unpack_archive("paper_patches.zip", ".")

# restructure files to match other datasets
shutil.move("dataset-splitting", "splits")
os.mkdir("images")
shutil.move("WSIs", "wsi")  # not currently used

# move patches/X/Y to images/Y
for patch_dir in os.listdir("patches"):
    for file in os.listdir("patches/" + patch_dir):
        shutil.move(f"patches/{patch_dir}/{file}", "images/")

# cleanup
os.remove("paper_patches.zip")
shutil.rmtree("patches")
os.chdir(proj_dir)
