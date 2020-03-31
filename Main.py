from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import glob
import pathlib
import tensorflow.keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
import joblib
import cv2
import tqdm
import random

fruit_folder = pathlib.Path('D:/Brin/fruit-recognition/')
file_list = glob.glob(os.path.join(fruit_folder, '*/*.png'))
random.shuffle(file_list)
image_count = len(file_list)
print(image_count)

directories = [item.name for item in fruit_folder.glob('*')]
category_map = {category: index for index, category in enumerate(directories)}
number_of_categories = len(directories)
print(directories)

def prepare_image_and_label(filepath):
    # Creating label from folder name
    fp = filepath  # str(filepath.numpy())
    os.path.split(fp)
    parts = fp.split(os.path.sep)
    label = parts[-2]

    # Then one-hot encoding the label
    label = category_map.get(label, -1)
    if label < 0:
        raise Exception(f"{filepath} didn't find a category")
    label = to_categorical(label, number_of_categories)

    # Then reading and resizing the image
    image = cv2.imread(fp)
    image = cv2.resize(image, dsize=(258, 320), interpolation=cv2.INTER_CUBIC)
    return image, label

# Create dataset
full_ds = []
images = []
labels = []

# "Pickling" dataset so we don't need to re-create it every time
full_ds_file = 'full-dataset-data.joblib'
if not os.path.exists(full_ds_file):
    for f in tqdm.tqdm(file_list):
        try:
            image, label = prepare_image_and_label(f)
            images.append(image)
            labels.append(label)
        # If file can't be "prepared"
        except Exception:
            print(f"{f} failed continuing")
    with open(full_ds_file, 'wb') as fid:
        joblib.dump({"images": images, "labels":labels},fid)
        # joblib.dump(full_ds, fid)
else:
    with open(full_ds_file, 'rb') as f:
        full_ds = joblib.load(f)
        images = full_ds["images"]
        labels = full_ds["labels"]
        full_ds.clear()

# np.random.shuffle(full_ds)
images = np.asarray(images)
labels = np.asarray(labels)

validation_size = int(0.15 * image_count)
test_size = int(0.15 * image_count)
train_size = image_count - validation_size - test_size

test_images = images[:test_size]
train_images = images[test_size:]
test_labels = labels[:test_size]
train_labels = labels[test_size:]

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(320, 258, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3,), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(15, activation='sigmoid'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


history = model.fit(train_images,
                    train_labels,
                    epochs=20,
                    batch_size=32,
                    validation_split=0.15,
                    shuffle=True
                    )

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.legend()
plt.figure()
