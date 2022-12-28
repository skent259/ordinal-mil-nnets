import os
import shutil

import wget

proj_dir = os.getcwd()
data_dir = "datasets/imdb"
os.chdir(data_dir)

# download
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
wget.download(url, ".")

# unzip and extract images
shutil.unpack_archive("aclImdb_v1.tar.gz", ".")
os.rename("aclImdb", "reviews")

# cleanup
files_to_remove = [
    "aclImdb_v1.tar.gz",
    "reviews/imdb.vocab",
    "reviews/imdbEr.txt",
    "reviews/test/labeledBow.feat",
    "reviews/test/urls_neg.txt",
    "reviews/test/urls_pos.txt",
    "reviews/train/labeledBow.feat",
    "reviews/train/unsupBow.feat",
    "reviews/train/urls_neg.txt",
    "reviews/train/urls_pos.txt",
    "reviews/train/urls_unsup.txt",
]
for x in files_to_remove:
    os.remove(x)

os.chdir(proj_dir)
