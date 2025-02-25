import numpy as np
from tensorflow.keras.datasets import mnist

def load_and_preprocess_data():
    # Carica il dataset MNIST (training e test)
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Preprocessing delle immagini: rimodellare da (28, 28) a (28, 28, 1) per le CNN
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    # Normalizza i pixel delle immagini tra 0 e 1 (i pixel sono inizialmente tra 0 e 255)
    train_images, test_images = train_images / 255.0, test_images / 255.0

    return train_images, train_labels, test_images, test_labels
