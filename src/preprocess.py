
change_dim = True #@param {type:"boolean"}
img_height = 400 #@param {type:"number"}
img_width =   600 #@param {type:"number"}


split= (.9, .1) #@param {type:"raw"}


import splitfolders

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio("data/data_raw", output="data/data_processed",
    seed=1337, ratio=(.9, .1), group_prefix=None, move=False) # default values


img_width =int(img_width)
img_height =int(img_height)

#!/usr/bin/python
from PIL import Image
import os, sys

def resize(path, moveto, size):
    dirs = os.listdir(path)
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            im.thumbnail(size, Image.ANTIALIAS)
            im.save(moveto+item)

size = img_width , img_height
if change_dim:
    resize("data/data_processed/train/original_images/",
            "data/data_processed/train/original_images/",
            size)
    resize("data/data_processed/train/label_images_semantic/",
            "data/data_processed/train/label_images_semantic/",
            size)

    resize("data/data_processed/val/original_images/",
            "data/data_processed/val/original_images/",
            size)

    resize("data/data_processed/val/label_images_semantic/",
            "data/data_processed/val/label_images_semantic/",
            size)