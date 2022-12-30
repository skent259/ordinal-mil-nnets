import os
import shutil

import wget

proj_dir = os.getcwd()
data_dir = "datasets/afad"
os.chdir(data_dir)

# download files
url = "https://github.com/afad-dataset/tarball/raw/master"

alpha = "abcdefghijklmnopqrstuvwxyz"
extensions = [f"a{i}" for i in alpha] + [f"b{i}" for i in alpha[:20]]
# extensions = ["aa", "ab", ..., "bt"]
files = [f"AFAD-Full.tar.xz{x}" for x in extensions]

for f in files:
    try:
        wget.download(f"{url}/{f}", ".")
    except Exception as e:
        print(f"Failed to download {f}")

# combine the extensions, like "restore.sh"
os.system("cat AFAD-Full.tar.xz* > AFAD-Full.tar.xz")

# unzip and extract images
shutil.unpack_archive("AFAD-Full.tar.xz", ".")
shutil.move("AFAD-Full", "images")

# cleanup
os.remove("AFAD-Full.tar.xz")
os.remove(".DS_Store")
for f in files:
    os.remove(f)

os.chdir(proj_dir)
