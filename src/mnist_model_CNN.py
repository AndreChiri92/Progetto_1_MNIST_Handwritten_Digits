import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data_preprocessing import load_and_preprocess_data

# Caricamento dei dati MNIST
(train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()

# Data Augmentation migliorata
datagen = ImageDataGenerator(
    rotation_range=15,    
    zoom_range=0.2,       
    width_shift_range=0.15, 
    height_shift_range=0.15,
    shear_range=0.1,      
    horizontal_flip=False
)
datagen.fit(train_images)

# Creazione del modello CNN migliorato
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation='relu'),

    layers.Conv2D(128, (3,3), activation='relu'),  # Nuovo livello convoluzionale
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),  # Dropout aumentato per ridurre overfitting
    layers.Dense(10, activation='softmax')
])

# Compilazione del modello con Adam e learning rate personalizzato
model.compile(optimizer=Adam(learning_rate=0.0005), 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

# Definizione dei callback per stabilizzare il training
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

# Allenamento del modello con Data Augmentation e callback
history = model.fit(datagen.flow(train_images, train_labels, batch_size=64), 
                    epochs=25, 
                    validation_data=(test_images, test_labels),
                    callbacks=[early_stopping, lr_scheduler])

# Valutazione del modello
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

# Salvataggio del modello migliorato
model.save('data/mnist_model_cnn.h5')

# 📊 Visualizzazione dei risultati dell'allenamento
plt.figure(figsize=(12, 5))

# Accuratezza
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')

# Perdita (Loss)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')

plt.show()
