import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Charger les données à partir du fichier CSV
df = pd.read_csv('euromillions.csv')

# Extraire les numéros principaux et les numéros étoile de chaque tirage
X = df.iloc[:, 1:8].values

# Extraire les numéros gagnants pour la prédiction
y = df.iloc[:, 8:10].values

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliser les données
X_train = X_train / 50.0
X_test = X_test / 50.0
y_train = y_train / 12.0
y_test = y_test / 12.0

# Définir l'architecture du réseau de neurones
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(7,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2)
])

# Compiler le modèle
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Entraîner le modèle
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# Évaluer la précision du modèle sur l'ensemble de test
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print("MAE:", mae)
