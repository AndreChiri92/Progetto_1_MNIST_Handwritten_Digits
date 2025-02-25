import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Caricare il dataset MNIST
data = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()

# Normalizzare i valori dei pixel tra 0 e 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Definire il modello della rete neurale
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Livello di input
    keras.layers.Dense(128, activation='relu'),  # Livello nascosto
    keras.layers.Dense(10, activation='softmax') # Livello di output
])

# Compilare il modello
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Addestrare il modello
model.fit(x_train, y_train, epochs=5)

# Valutare il modello
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Accuratezza sul test set: {test_acc:.4f}")

# Visualizzare alcune predizioni
def plot_image(index):
    plt.imshow(x_test[index], cmap=plt.cm.binary)
    plt.xlabel(f"Vero: {y_test[index]}")
    plt.show()
    
def predict_image(index):
    prediction = model.predict(np.array([x_test[index]]))
    print(f"Predizione: {np.argmax(prediction)}")
    
# Test su un'immagine a caso
index = np.random.randint(0, len(x_test))
plot_image(index)
predict_image(index)
