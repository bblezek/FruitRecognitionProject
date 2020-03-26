from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import os
import glob
import pathlib


fruit_folder = pathlib.Path('D:/Brin/fruit-recognition/')
image_count = len(glob.glob(os.path.join(fruit_folder, '*/*.png')))
print(image_count)

directories = np.array([item.name for item in fruit_folder.glob('*')])

print(directories)

list_ds = tf.data.Dataset.list_files(str(fruit_folder/'*/*'))

for f in list_ds.take(5):
    print(f.numpy())

def prepare_image(filepath):
    parts = tf.strings.split(filepath, os.path.sep)
    label = parts[-2]

    image = tf.io.read_file(filepath)
    image = tf.image.decode_png(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    # No need to change size because they are all the same
    return image, label

#tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

# Creating dataset using dataset map
labelled_ds = list_ds.map(prepare_image)
for image, label in labelled_ds.take(15):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())