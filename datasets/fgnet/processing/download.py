import os
import shutil

import wget

proj_dir = os.getcwd()
data_dir = "datasets/fgnet"
os.chdir(data_dir)

# download
url = "http://yanweifu.github.io/FG_NET_data/FGNET.zip"
wget.download(url, ".")

# unzip and extract images
shutil.unpack_archive("FGNET.zip", ".")
shutil.move("FGNET/images", "images")

# cleanup
os.remove("FGNET.zip")
shutil.rmtree("FGNET")
shutil.rmtree("__MACOSX")
os.chdir(proj_dir)
