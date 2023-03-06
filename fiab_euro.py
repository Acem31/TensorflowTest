import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Importer les données
df = pd.read_csv('euromillions.csv')

# Supprimer les colonnes inutiles
df = df.drop(['id', 'date', 'star1', 'star2'], axis=1)

# Convertir les données en tableau numpy
data = df.values

# Séparer les données en entrée et en sortie
X = data[:, :-1]
y = data[:, -1]

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliser les données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Définir le modèle
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))

# Compiler le modèle
model.compile(optimizer='adam', loss='mse')

# Définir l'early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Entraîner le modèle
model.fit(X_train, y_train, validation_split=0.2, epochs=100, callbacks=[early_stopping])

# Prédire les résultats
predictions = model.predict(X_test)

# Evaluer le modèle sur les données de test
score = model.evaluate(X_test, y_test)
print("MSE :", score)

# Prédire le résultat pour un tirage spécifique (ici le 100ème)
input_data = X[100].reshape(1, -1)
output_data = model.predict(input_data)
print("Prédiction pour le tirage numéro 100 :", output_data[0, 0])
print("Résultat réel :", y[100])
