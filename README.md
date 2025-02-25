# MNIST Handwritten Digits Classification

Questo progetto implementa un modello di deep learning per riconoscere cifre scritte a mano utilizzando il dataset MNIST.

## 📌 Descrizione del Progetto
Il dataset MNIST è uno dei più noti dataset di computer vision e contiene 70.000 immagini di cifre scritte a mano (0-9), ciascuna di dimensioni 28x28 pixel in scala di grigi.

Il progetto include:
- Preprocessing dei dati (normalizzazione e conversione delle etichette in one-hot encoding).
- Creazione e addestramento di un modello di rete neurale con TensorFlow/Keras.
- Salvataggio del modello addestrato.
- Caricamento del modello per effettuare previsioni.
- Visualizzazione di esempi di predizione.

## 📂 Struttura del Progetto
```
ML_AI_Projects/
│
├── Progetto_1_MNIST_Handwritten_Digits/
│   ├── data/                  # Dataset (se necessario)
│   ├── notebooks/             # Jupyter notebooks
│   │   └── MNIST_Model.ipynb  # Notebook principale
│   ├── src/                   # Codice Python
│   │   ├── mnist_model.py     # Definizione e addestramento del modello
│   │   ├── data_preprocessing.py  # Preprocessing dei dati
│   ├── requirements.txt       # Dipendenze Python
│   └── README.md              # Questo file
```

## 🚀 Installazione e Utilizzo
### 1️⃣ Clonare il repository
```bash
git clone <URL del repository>
cd Progetto_1_MNIST_Handwritten_Digits
```

### 2️⃣ Creare un ambiente virtuale (opzionale ma consigliato)
```bash
python -m venv venv
source venv/bin/activate  # Su macOS/Linux
venv\Scripts\activate     # Su Windows
```

### 3️⃣ Installare le dipendenze
```bash
pip install -r requirements.txt
```

### 4️⃣ Eseguire l'addestramento del modello
```bash
python src/mnist_model.py
```
Il modello verrà salvato come `mnist_model.h5`.

### 5️⃣ Avviare il notebook ed effettuare previsioni
Aprire `MNIST_Model.ipynb` in Jupyter Notebook o Jupyter Lab ed eseguire le celle per caricare il modello e testarlo.

## 📊 Risultati
Dopo l'addestramento, il modello raggiunge un'accuratezza di circa il **98%** sul set di test.

## 🔧 Tecnologie Utilizzate
- Python 3
- TensorFlow/Keras
- NumPy
- Matplotlib
- Jupyter Notebook

## 📌 Possibili Estensioni
- Utilizzo di una **rete neurale convoluzionale (CNN)** per migliorare l'accuratezza.
- Esperimenti con **data augmentation** per rendere il modello più robusto.
- Deployment del modello come **API REST** utilizzando Flask o FastAPI.

---

✉️ Per domande o suggerimenti, sentiti libero di contribuire o contattarmi!

