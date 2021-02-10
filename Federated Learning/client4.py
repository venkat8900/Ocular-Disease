import os

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
# from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

import flwr as fl

if __name__ == "__main__":

    # loop over the image paths
    for imagePath in imagePaths:
        # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]

        # load the image, swap color channels, and resize it to be a fixed
        # 224x224 pixels while ignoring aspect ratio
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        # update the data and labels lists, respectively
        data.append(image)
        if label == "agedegen":
            label = 1
        elif label == "cataract":
            label = 2
        elif label == "diabetes":
            label = 3
        else: 
            label = 0
        labels.append(label)

        # convert the data and labels to NumPy arrays while scaling the pixel
        # intensities to the range [0, 255]
        data = np.array(data) / 255.0
        labels = np.array(labels)

    # perform one-hot encoding on the labels
    # lb = LabelEncoder()
    # labels = lb.fit_transform(labels)
    # labels = to_categorical(labels)

    # partition the data into training and testing splits using 80% of
    # the data for training and the remaining 20% for testing
    (x_train, x_test, y_train, y_test) = train_test_split(data, labels,
        test_size=0.20, stratify=labels, random_state=42)

    y_train = to_categorical(y_train, num_classes = 4)
    y_test = to_categorical(y_test, num_classes = 4)

    # initialize the training data augmentation object
    # trainAug = ImageDataGenerator(rotation_range=15, fill_mode="nearest")

    # load the VGG16 network, ensuring the head FC layer sets are left off
    baseModel = VGG16(weights="imagenet", include_top=False,
        input_tensor=Input(shape=(224, 224, 3)))

    # construct the head of the model that will be placed on top of the
    # the base model
    headModel = baseModel.output
    #headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
    # headModel = AveragePooling2D(pool_size=(2, 2))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    #headModel = Dense(64, activation="relu")(headModel)
    # headModel = Dense(256)(headModel)
    # headModel = LeakyReLU(alpha=0.3)(headModel)
    headModel = Dropout(0.3)(headModel)
    #headModel = Dense(128)(headModel)
    #headModel = LeakyReLU(alpha=0.3)(headModel)
    #headModel = Dropout(0.5)(headModel)
    #headModel = Dense(64)(headModel)
    #headModel = LeakyReLU(alpha=0.3)(headModel)
    #headModel = Dropout(0.5)(headModel)
    headModel = Dense(4, activation="softmax")(headModel)

    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)

    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process

    for layer in baseModel.layers:
        layer.trainable = False

    # Load and compile Keras model
    # model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=4, weights=None)
    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

    # Load CIFAR-10 dataset
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Define Flower client
    class CifarClient(fl.client.NumPyClient):
        def get_parameters(self):  # type: ignore
            return model.get_weights()

        def fit(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            model.fit(x_train, y_train, epochs=1, batch_size=20, steps_per_epoch=3)
            # model.fit_generator(trainAug.flow(x_train, y_train, batch_size = 20, steps_per_epoch = len(x_train)//20,
              #  validation_data = (x_test, y_test), validation_steps = len(x_test)//20, epochs = 1))
            return model.get_weights(), len(x_train)

        def evaluate(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(x_test, y_test)
            return len(x_test), loss, accuracy

    # Start Flower client
    fl.client.start_numpy_client("[::]:8080", client=CifarClient())
