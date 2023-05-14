import os, warnings
from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.utils import image_dataset_from_directory
import pandas as pd


# Set Matplotlib Defaults
plt.rc("figure", autolayout=True)
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlesize=18, titlepad=10)
plt.rc("image", cmap="magma")
warnings.filterwarnings("ignore")


#Load Training and Validation sets

#######################################################################
# TO DO  : Sostituisci i dati caricati da kaggle con un altro dataset #
#######################################################################

ds_train_ = image_dataset_from_directory(
    'car-or-truck/train',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=True,
)
ds_valid_ = image_dataset_from_directory(
    'car-or-truck/valid',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=False,
)

# Data Pipeline
def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

###### TO DO -  Riattivare Dopo il loading del dataset #############
AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = (
  ds_train_.map(convert_to_float).cache().prefetch(buffer_size = AUTOTUNE)  
)

ds_valid= (
   ds_valid_.map(convert_to_float).cache().prefetch(buffer_size=AUTOTUNE)
)


# Ora Strutturiamo la Convnet

model = keras.Sequential([
    
    # INPUT LAYER
    layers.InputLayer(input_shape=[128,128,3]),

    # DATA AUGMENTATION
    layers.RandomContrast(factor=0.10),
    layers.RandomFlip(mode="horizontal"),
    layers.RandomRotation(factor = 0.10),


    # BASE

    # Block 1
    layers.BatchNormalization(renorm=True),
    layers.Conv2D(filters=64, kernel_size = 3, activation="relu", padding="same"),
    layers.MaxPool2D(),

    # Block 2
    layers.BatchNormalization(renorm=True),
    layers.Conv2D(filters=128, kernel_size = 3, activation="relu", padding="same"),
    layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding="same"),
    layers.MaxPool2D(),

    # Block 3
    layers.BatchNormalization(renorm=True),
    layers.Conv2D(filters=256, kernel_size=3, activation="relu", padding="same"),
    layers.Conv2D(filters=256, kernel_size =3, activation ="relu", padding="same"),
    layers.Conv2D(filters=256, kernel_size = 3, activation ="relu", padding="same"),
    layers.MaxPool2D(),

    # HEAD
    layers.BatchNormalization(renorm=True),
    layers.Flatten(),
    layers.Dense(10, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid"),

])


model.compile(
    optimizer = tf.keras.optimizers.Adam(epsilon=0.01),
    loss="binary_crossentropy",
    metrics=["binary_accuracy"]
)


history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=50,
)

model.save("saved_model/first-cnn-saved-model")

model.save("h5_model/first-cnn-h5-model")

# showing the history and plotting the loss/val_loss and acuracy/val_accuracy lines on graph
history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ["loss", "val_loss"]].plot()
plt.show()
history_frame.loc[:, ["binary_accuracy", "val_binary_accuracy"]].plot()
plt.show()


