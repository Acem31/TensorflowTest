import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Charger les données
df = pd.read_csv("euromillions.csv")

# Séparer les numéros principaux et les numéros étoile
X_main = df.iloc[:, :5]
X_star = df.iloc[:, 5:7]

# One-hot encode les numéros principaux
X_main = pd.get_dummies(X_main, prefix="main")

# Concaténer les numéros principaux et les numéros étoile
X = pd.concat([X_main, X_star], axis=1)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Normaliser les données
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_test.mean()) / X_test.std()

# Convertir les données en tableaux numpy
X_train = np.array(X_train)
X_test = np.array(X_test)

# Préparer les étiquettes des données d'entraînement et de test
y_train = np.zeros((X_train.shape[0], 13))
y_test = np.zeros((X_test.shape[0], 13))

for i, num in enumerate(df.iloc[:, 7]):
    y = np.zeros(13)
    for j in range(5):
        y[num[j]-1] = 1
    y_test[i] = y

for i, num in enumerate(df.iloc[:, 7]):
    y = np.zeros(13)
    for j in range(5):
        y[num[j]-1] = 1
    y_train[i] = y

# Construire le modèle
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=[50]),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(13, activation='softmax')
])

# Compiler le modèle
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entraîner le modèle
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Faire des prédictions sur les données de test
predictions = model.predict(X_test)

# Évaluer la précision des prédictions
accuracy = model.evaluate(X_test, y_test)[1]

print(f"Précision : {accuracy:.2%}")