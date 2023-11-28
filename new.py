import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Charger les données
data = pd.read_csv('euromillions.csv', sep=';', header=None)

# Prétraitement des données
main_numbers = data.iloc[:, 0:6]
bonus_numbers = data.iloc[:, 6:8]
sequences = pd.concat([main_numbers, bonus_numbers], axis=1)

# Normaliser les données
scaler = StandardScaler()
sequences = scaler.fit_transform(sequences)

# Préparer les données pour l'apprentissage
X, y = [], []
sequence_length = 10

for i in range(len(sequences) - sequence_length):
    X.append(sequences[i:i+sequence_length])
    y.append(sequences[i+sequence_length])

X = np.array(X)
y = np.array(y)

# Diviser les données en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construire le modèle LSTM
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(sequence_length, X.shape[2])))
model.add(Dense(32, activation='relu'))  # Ajustez le nombre de neurones et la fonction d'activation
model.add(Dense(X.shape[2]))  # Assurez-vous que le nombre de neurones correspond à votre sortie
model.compile(optimizer='adam', loss='mse')

# Entraîner le modèle
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Évaluer le modèle sur l'ensemble de test (facultatif)
loss = model.evaluate(X_test, y_test)
print(f"Loss on test set: {loss}")

# Faire une prédiction pour le prochain tirage
last_sequence = sequences[-sequence_length:].reshape(1, sequence_length, X.shape[2])
predicted_numbers = model.predict(last_sequence)

# Inverser la normalisation pour obtenir les numéros prédits
predicted_numbers = scaler.inverse_transform(predicted_numbers)

print("Numéros prédits pour le prochain tirage:")
print(predicted_numbers)
