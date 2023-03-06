import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Charger les données depuis le fichier csv
df = pd.read_csv('euromillions.csv', sep=';', header=None)

# Sélectionner les 7 colonnes pertinentes pour les tirages
df = df.iloc[:, :7]

# Diviser les données en entrées (les 5 premières colonnes) et en sortie (la dernière colonne)
X = df.iloc[:, :5].values
y = df.iloc[:, 6].values

# Diviser les données en ensembles de train/test
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

# Prédire les résultats sur les données de test
predictions = model.predict(X_test)

# Evaluer le modèle sur les données de test
score = model.evaluate(X_test, y_test)
print(f"Score (MSE) : {score:.4f}")
