import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from rdkit import Chem

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

from PIL.PngImagePlugin import PngImageFile, PngInfo

data_dir = Path("./imagesFromSmilesLite/")

images = sorted(list(map(str, list(data_dir.glob("*.png")))))
labels = [str(PngImageFile(f"{img}").text).split("'")[3] for img in images]
characters = set(char for label in labels for char in label)

print("Number of images found:", len(images))
print("Number of labels found: ", len(labels))
print("Number of unique characters: ", len(characters))
print("Characters present: ", characters)

batch_size = 32

img_width = 256
img_height = 256


downsample_factor = 16

max_length = max([len(label) for label in labels])


def pad_label(label):
    padded_label = label + (max_length - len(label))*'_'

    return padded_label


for label in labels:
    label = pad_label(label)


def gen_char_map_dic():
    i = 0
    char_map_dic = {}
    for c in characters:
        char_map_dic[c] = i
        i += 1
    char_map_dic['_'] = i

    return char_map_dic


char_map_dic = gen_char_map_dic()


def encode_label(label):
    label = pad_label(label)
    label = str(label)
    label = [char_map_dic[c] for c in label]
    encoded = np.zeros((max_length, len(char_map_dic.keys())))
    for j in range(max_length):
        encoded[j][label[j]] = 1

    return encoded


encoded_labels = [encode_label(label) for label in labels[:8192]]


def split_data(images, encoded_labels, train_size=0.9, shuffle=True):
    size = len(images)
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    train_samples = int(size * train_size)
    x_train, y_train = images[indices[:train_samples]], encoded_labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], encoded_labels[indices[train_samples:]]
    return x_train, x_valid, y_train, y_valid


x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(encoded_labels))


def encode_single_sample(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.transpose(img, perm=[1, 0, 2])
    # Map characters in label to nums
    #label = encode_label(label)
    input_label = label[:-1]
    output_label = label[1:]
    # Return a dictionary of key: value img: label
    return{"image": img, "input_label": input_label, "output_label": output_label}


def encode_samples(x, y):
    samples_dic = {"image": [], "input_label": [], "output_label": []}
    for xi, yi in zip(x, y):
        temp_dic = encode_single_sample(xi, yi)
        for(k, v) in temp_dic.items():
            samples_dic[k].append(v)

    for k, v in samples_dic.items():
        samples_dic[k] = np.array(v)

    return samples_dic


def build_model(d1, d2, kernel_size):
    # Inputs to the model
    input_img = layers.Input(
        shape=(img_width, img_height, 1), name="image", dtype="float32"
    )
    labels = layers.Input(name="input_label", shape=(d1, d2), dtype="float32")

    # First conv block
    x = layers.Conv2D(
        32,
        (kernel_size, kernel_size),
        kernel_initializer="glorot_uniform",
        padding="same",
        name="FirstConv",
    )(input_img)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(
        32,
        (kernel_size, kernel_size),
        kernel_initializer="glorot_uniform",
        padding="same",
        name="FirstConv1",
    )(x)
    x = layers.LeakyReLU()(x)

    # Second conv block (repeated 2x)
    x = layers.Conv2D(
        32,
        (kernel_size, kernel_size),
        kernel_initializer="glorot_uniform",
        padding="same",
        name="SecondConv",
    )(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    x = layers.Conv2D(
        32,
        (kernel_size, kernel_size),
        kernel_initializer="glorot_uniform",
        padding="same",
        name="SecondConv1",
    )(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Third conv block
    x = layers.Conv2D(
        64,
        (kernel_size, kernel_size),
        kernel_initializer="glorot_uniform",
        padding="same",
        name="ThirdConv",
    )(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Fourth conv block (repeated 2x) output should be fed into 8x8x512 context matrix
    x = layers.Conv2D(
        256,
        (kernel_size, kernel_size),
        kernel_initializer="glorot_uniform",
        padding="same",
        name="FourthConv",
    )(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)

    x = layers.Conv2D(
        256,
        (kernel_size, kernel_size),
        kernel_initializer="glorot_uniform",
        padding="same",
        name="FourthConv1",
    )(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D((2, 2), name="pool5")(x)
    '''
    # Fifth conv block (repeated 2x)
    x = layers.Conv2D(
        512,
        (5, 5),
        kernel_initializer="glorot_uniform",
        padding="same",
        name="Conv512_5x5",
    )(x)
    x = layers.PReLU()(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    x = layers.Conv2D(
        512,
        (5, 5),
        kernel_initializer="glorot_uniform",
        padding="same",
        name="Conv512_5x5",
    )(x)
    x = layers.PReLU()(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Sixth conv block
    x = layers.Conv2D(
        4096,
        (2, 2),
        kernel_initializer="glorot_uniform",
        padding="same",
        name="conv4096_2x2",
    )(x)
    x = layers.PReLU()(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)'''

    # Output of last maxpool now needs to be turned into 4096 state vector

    # We have used eight max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 16x smaller. The number of
    # filters in the last layer is 4096. Reshape accordingly before
    # Passing the output to the RNN part of the model
    #new_shape = ((img_width // 32), (img_height // 32) * 512)
    #x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    #x = layers.Flatten()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x1 = layers.Dense(64, activation="relu", name="dense1")(x)
    x1 = layers.Dropout(0.2)(x1)
    x2 = layers.Dense(64, activation="relu", name="dense2")(x)
    x2 = layers.Dropout(0.2)(x2)

    # RNNs
    y = layers.LSTM(64, return_sequences=True, dropout=0.25)(labels, initial_state=[x1, x2])
    y = layers.LSTM(64, return_sequences=True, dropout=0.25)(y)
    y = layers.LSTM(64, return_sequences=True, dropout=0.25)(y)

    # Output layer
    output = layers.Dense(len(characters) + 1, activation="softmax", name="output_label")(y)


    model = models.Model(
        inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
    )

    # Optimizer
    opt = keras.optimizers.Adam()
    # Compile the model and return
    model.compile(optimizer=opt, loss="categorical_crossentropy")
    return model


model = build_model(d1=None, d2=44, kernel_size=3)
model.summary()

epochs = 100
'''early_stopping_patience = 10
# Add early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
)'''

train_dataset = encode_samples(x_train, y_train)
validation_dataset = encode_samples(x_valid, y_valid)
print(train_dataset["output_label"].shape)
# Train the model
callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5", save_weights_only=True),
]
history = model.fit(
    [train_dataset["image"], train_dataset["input_label"]],
    train_dataset["output_label"],
    validation_data=([validation_dataset["image"], validation_dataset["input_label"]], validation_dataset["output_label"]),
    epochs=epochs,
    callbacks=callbacks,
)

prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="output_label").output
)
prediction_model.summary()