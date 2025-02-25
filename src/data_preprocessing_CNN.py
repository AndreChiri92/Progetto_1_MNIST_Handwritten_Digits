import numpy as np
from tensorflow.keras.datasets import mnist

def load_and_preprocess_data():
    # Carica il dataset MNIST
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Reshape per CNN
    train_images = train_images.reshape(-1, 28, 28, 1) / 255.0
    test_images = test_images.reshape(-1, 28, 28, 1) / 255.0

    return train_images, train_labels, test_images, test_labels
