import csv
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense

# Charger les données en tant que tuples
data = []
with open('euromillions.csv', 'r') as file:
    reader = csv.reader(file, delimiter=';')
    for row in reader:
        numbers = tuple(map(int, row[:5]))
        result = tuple(map(int, row[5:]))
        data.append((numbers, result))

# Convertir les tuples en DataFrame
data_df = pd.DataFrame(data, columns=['numbers', 'result'])

# Extraire X et y à partir du DataFrame
X = np.array([np.array(x) for x in data_df['numbers']])
y = np.array([np.array(r) for r in data_df['result']])

# Sélectionner toutes les lignes sauf les 10 dernières pour l'entraînement
X_train = X[:-10]
y_train = y[:-10]

# Sélectionner les 10 dernières lignes pour les tests
X_test = X[-10:]
y_test = y[-10:]

# Fonction pour créer le modèle Keras
def create_model():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(2))  # Deux sorties pour le second tirage
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

target_success_rate = 0.75
success_rate = 0.0
num_iterations = 0

while success_rate < target_success_rate:
    num_iterations += 1
    
    model = create_model()  # Recréer le modèle à chaque itération
    
    # Entraîner le modèle
    model.fit(X_train, y_train, epochs=1, verbose=0)
    
    # Faire des prédictions sur les données de test
    predictions = model.predict(X_test)
    
    # Calculer le taux de réussite
    success_rate = accuracy_score(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1))
    
    print('Taux de réussite à l\'itération', num_iterations, ':', success_rate)

print('Nombre d\'itérations nécessaires pour atteindre le seuil de réussite :', num_iterations)

# Intégrer les 10 dernières lignes dans l'apprentissage
X_train = np.concatenate((X_train, X_test), axis=0)
y_train = np.concatenate((y_train, y_test), axis=0)

# Réentraîner le modèle avec les nouvelles données
model = create_model()
model.fit(X_train, y_train, epochs=1, verbose=1)

# Faire une prédiction
def make_prediction(model, X):
    # Faire la prédiction
    predictions = model.predict(X)
    # Afficher les prédictions
    print('Prédictions :', predictions)

print('Seuil de réussite atteint. Faisons une prédiction.')
make_prediction(model, X_test)
