import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical

# Charger les données depuis le fichier CSV
data = pd.read_csv('euromillions.csv', delimiter=';')
numbers = data.iloc[:, :5].values  # Utilisez les 5 premiers numéros pour former un tuple

# Normalisation des données
scaler = MinMaxScaler()
numbers_normalized = scaler.fit_transform(numbers)

# Préparer les données pour l'apprentissage
X, y = [], []
sequence_length = 5  # Longueur de la séquence
for i in range(len(numbers_normalized) - sequence_length - 1):
    X.append(numbers_normalized[i:i+sequence_length])
    y.append(numbers_normalized[i+sequence_length])

X = np.array(X)
y = np.array(y)

# Convertir les étiquettes en format catégoriel
y_categorical = to_categorical(y, num_classes=33)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train_categorical, y_test_categorical = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Construire le modèle de classification
model = Sequential()
model.add(Dense(units=128, activation='relu', input_dim=5))
model.add(Dropout(0.2))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=33, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(X_train, y_train_categorical, epochs=50, batch_size=32, validation_split=0.2)

# Évaluer le modèle sur l'ensemble de test
loss, accuracy = model.evaluate(X_test, y_test_categorical)
print('Loss:', loss)
print('Accuracy:', accuracy)

# Utiliser le modèle pour la prédiction
last_five_numbers = numbers_normalized[-sequence_length:]
last_five_numbers = last_five_numbers.reshape(1, sequence_length, 5)
next_number_probabilities = model.predict(last_five_numbers)
predicted_numbers = np.random.choice(range(1, 51), size=5, p=next_number_probabilities[0][-1])

print("Numéros prédits :", predicted_numbers)
print("Probabilités pour les prochains numéros :", next_number_probabilities)
