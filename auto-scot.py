import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Charger les données depuis le fichier CSV
data = pd.read_csv('euromillions.csv', delimiter=';')
numbers = data.iloc[:, :5].values  # Utilisez les 5 premiers numéros pour former un tuple

# Normalisation des données
scaler = MinMaxScaler()
numbers_normalized = scaler.fit_transform(numbers)

# Préparer les données pour l'apprentissage
X, y = [], []
for i in range(len(numbers_normalized) - 6):  # Utilisez les 6 derniers tirages pour la prédiction du prochain
    X.append(numbers_normalized[i:i+5])
    y.append(np.argmax(numbers_normalized[i+5]))

X = np.array(X)
y = np.array(y)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construire le modèle de classification
model = Sequential()
model.add(Dense(units=128, activation='relu', input_dim=5))
model.add(Dropout(0.2))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=33, activation='softmax'))  # 33 classes pour les numéros 1 à 50 de l'Euromillions

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Évaluer le modèle sur l'ensemble de test
loss, accuracy = model.evaluate(X_test, y_test)
print('Loss:', loss)
print('Accuracy:', accuracy)

# Utiliser le modèle pour la prédiction
last_five_numbers = numbers_normalized[-5:]
last_five_numbers = last_five_numbers.reshape(1, 5)
next_number_probabilities = model.predict(last_five_numbers)
predicted_number = np.argmax(next_number_probabilities)

print("Probabilités pour les prochains numéros :", next_number_probabilities)
print("Numéro prédit :", predicted_number)
