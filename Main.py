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
import sklearn
from sklearn.metrics import confusion_matrix
import seaborn as sns

fruit_folder = pathlib.Path('D:/Brin/fruit-recognition/')
# Creating list of all image file names
file_list = glob.glob(os.path.join(fruit_folder, '*/*.png'))

# Shuffling file names so we don't have to shuffle our dataset later
random.shuffle(file_list)
image_count = len(file_list)

# Creating classes
directories = [item.name for item in fruit_folder.glob('*')]
category_map = {category: index for index, category in enumerate(directories)}
number_of_categories = len(directories)

# Creates a numpy array of the image and adds it to a list of images
# Also one-hot encodes the label and adds it to a list of labels
def prepare_image_and_label(filepath):
    # Creating label from folder name
    fp = filepath
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
    image = cv2.resize(image, dsize=(258, 320), interpolation=cv2.INTER_AREA)
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
        # If image can't be "prepared"
        except Exception:
            print(f"{f} failed continuing")
    with open(full_ds_file, 'wb') as fid:
        joblib.dump({"images": images, "labels":labels},fid)
else:
    with open(full_ds_file, 'rb') as f:
        full_ds = joblib.load(f)
        images = full_ds["images"]
        labels = full_ds["labels"]
        full_ds.clear()

# Casting image and label lists into numpy arrays
images = np.asarray(images)
labels = np.asarray(labels)

batch_size=32

# Splitting dataset into testing and training sets
test_size = int(0.15 * image_count)
if test_size%batch_size != 0:
    test_size = test_size - (test_size%batch_size)
test_images = images[:test_size]
train_images = images[test_size:]
test_labels = labels[:test_size]
train_labels = labels[test_size:]

model = tf.keras.models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(320, 258, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3,), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(15, activation='softmax'))

model.summary()

epochs = 25
learning_rate = 0.00001
decay = learning_rate/epochs
metrics = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc')
]

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(learning_rate=learning_rate, momentum=0.99),
              metrics=metrics)

history = model.fit(train_images,
                    train_labels,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.15,
                    shuffle=True
                    )

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

#Plotting results of training
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.legend()
plt.figure()

plt.plot(epochs, acc, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

test_steps = int(test_size/batch_size)

# model.predict won't accept uint8 numpy arrays, so casting to float32
test_images_float = np.float32(test_images)

# Making predictions about test set
test_predictions = model.predict(test_images_float,
                                 batch_size=batch_size,
                                 steps=test_steps)

# Using model to evaluate test images
test_results = model.evaluate(test_images,
                              test_labels,
                              batch_size=batch_size,
                              verbose=0)

# Printing metric results for test images from evaluating model
for name, value in zip(model.metrics_names, test_results):
    print(name, ' : ', value)
print()

# Converting one-hot encoded labels to a number for confusion matrix
int_test_predictions = np.argmax(test_predictions, axis = 1)
int_test_labels = np.argmax(test_labels, axis=1)

# Mapping true test results versus what they were predicted to be
cm = confusion_matrix(int_test_labels,int_test_predictions, normalize='true')
plt.figure(figsize=(12,12))
sns.heatmap(cm, annot=True, fmt=".2f")
tick_marks = np.arange(len(directories))
plt.xticks(tick_marks, directories, rotation=45)
plt.yticks(tick_marks, directories, rotation=45)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
plt.savefig('Confusion_Matrix.png')