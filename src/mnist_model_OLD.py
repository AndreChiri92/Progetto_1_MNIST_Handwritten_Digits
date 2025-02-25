import tensorflow as tf
from tensorflow.keras import layers, models


# Caricamento dei dati MNIST
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalizzazione delle immagini
train_images = train_images / 255.0
test_images = test_images / 255.0

# Creazione del modello
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Compilazione del modello
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Allenamento del modello
model.fit(train_images, train_labels, epochs=5)

# Valutazione del modello
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

# Salvataggio del modello
model.save('mnist_model.h5')  # Salva il modello in un file .h5