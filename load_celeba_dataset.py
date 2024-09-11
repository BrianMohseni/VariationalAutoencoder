import gdown
import os
from zipfile import ZipFile

os.makedirs("celeba", exist_ok=True)

url = "https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684"
output = "celeba_gan/data.zip"
gdown.download(url, output, quiet=True)

with ZipFile("celeba/data.zip", "r") as zipobj:
    zipobj.extractall("celeba")

os.remove("celeba/data.zip")
