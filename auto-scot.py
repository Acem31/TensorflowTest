import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense

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
    y.append(numbers_normalized[i+5])

X = np.array(X)
y = np.array(y)

# Remodeler les données d'entraînement pour être en 3D
X = X.reshape(X.shape[0], X.shape[1], 1)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construire le modèle LSTM
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(5, 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Entraîner le modèle
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Évaluer le modèle sur l'ensemble de test
loss = model.evaluate(X_test, y_test)
print('Loss:', loss)

# Utiliser le modèle pour la prédiction
last_five_numbers = numbers_normalized[-5:]
last_five_numbers = last_five_numbers.reshape(1, 5, 1)
next_numbers_prediction = model.predict(last_five_numbers)
rounded_predictions = scaler.inverse_transform(next_numbers_prediction.reshape(1, -1))

print("Prédiction pour les 5 prochains numéros :", rounded_predictions)
