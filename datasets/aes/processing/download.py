import ast
import os
import shutil
import urllib

import flickrapi
import numpy as np
import pandas as pd
import wget
from decouple import config
from PIL import Image

proj_dir = os.getcwd()
data_dir = "datasets/aes"
os.chdir(data_dir)

# download
url = "http://www.di.unito.it/~schifane/dataset/beauty-icwsm15/data/beauty-icwsm15-dataset.tsv.zip"
wget.download(url, ".")

# unzip and extract .tsv file
shutil.unpack_archive("beauty-icwsm15-dataset.tsv.zip", "processing/")
df = pd.read_csv("processing/beauty-icwsm15-dataset.tsv", sep="\t")

# Access Flickr api to download files
api_key = config("flickr_key", default="")
api_secret = config("flickr_secret", default="")

flickr = flickrapi.FlickrAPI(api_key, api_secret)


def get_url(pid):
    """
    generate flickr URL from given photo ID.

    Thanks to the following link:
    https://github.com/Raschka-research-group/corn-ordinal-neuralnet/blob/main/datasets/aes/processing-code/download_image.py
    """
    try:
        root = flickr.photos_getInfo(photo_id=pid)
    except flickrapi.exceptions.FlickrError:
        return ""
    info = root[0].attrib
    pid = info["id"]
    server = info["server"]
    secret = info["secret"]
    url = "https://live.staticflickr.com/" + server + "/" + pid + "_" + secret + ".jpg"
    return url


def fetch_save(pid: int) -> bool:
    """
    Citing the fetch code from the fetch-dataset.ipynb
    Returns True if downloaded the photo sucessfully
    
    Thanks to the following link, with minor modifications:
    https://github.com/Raschka-research-group/corn-ordinal-neuralnet/blob/main/datasets/aes/processing-code/download_image.py
    """
    url = get_url(pid)
    if url == "":
        return False
    filename_url = str(pid) + ".jpg"
    file_dest_index = os.path.join("images", filename_url)

    if not os.path.exists(file_dest_index):
        try:
            request = urllib.request.urlopen(url, timeout=15)
        except urllib.error.HTTPError:
            return False
        with open(file_dest_index, "wb") as f:
            try:
                f.write(request.read())
            except:
                print(f"error in {filename_url}")
                return False
        im = Image.open(file_dest_index)
        im.save(file_dest_index)

    return os.path.exists(file_dest_index)


# Download available files from df into images/
available = []
for i in range(df.shape[0]):
    pid = df.iloc[i][0]
    added = fetch_save(pid)
    if added:
        available.append(i)
    if i % 100 == 0:
        print(i)

print(f"Number of available instances: {len(available)}")

# Select available images
available_img = os.listdir("images")
available_img = [int(i.replace(".jpg", "")) for i in available_img]
ind = df["#flickr_photo_id"].isin(available_img)
df2 = df.loc[ind,].copy()

# Calculate beauty score and adjust columns
df2["beauty_scores"] = df2["beauty_scores"].apply(
    lambda x: ast.literal_eval("[" + x + "]")
)
df2["score"] = [np.array(x).mean() for x in df2.beauty_scores.values]
df2["score"] = df2["score"].round().astype(int) - 1
df2["img_name"] = df2["#flickr_photo_id"].astype(str) + ".jpg"

df2 = df2[["img_name", "category", "beauty_scores", "score"]].reset_index(drop=True)

print(df2)
df2.to_csv("aes.csv")


# cleanup
os.remove("beauty-icwsm15-dataset.tsv.zip")
os.chdir(proj_dir)

