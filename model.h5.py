import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Lecture du fichier CSV
data = pd.read_csv("euromillions.csv", header=None)

# Transformation des données en un tableau numpy
numbers = np.array(data)

# Création des jeux de données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(numbers[:, :5], numbers[:, 5:], test_size=0.2, random_state=42)

# Construction du modèle
model = Sequential()
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='linear'))

# Compilation du modèle
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# Définition du rappel EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Entraînement du modèle
model.fit(X_train, y_train, validation_split=0.2, epochs=100, callbacks=[early_stopping])

# Évaluation de la performance du modèle sur le jeu de test
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test MAE:', score[1])

# Prédiction sur un nouveau tirage (par exemple le dernier tirage de la loterie)
new_numbers = np.array([[4, 12, 25, 46, 48]])
prediction = model.predict(new_numbers)
print(prediction)
