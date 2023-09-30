import csv
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense

def reformat_labels(labels):
    reformatted_labels = []
    for label in labels:
        # Assurez-vous que chaque tuple a exactement 5 éléments
        if len(label) != 5:
            print(f"Erreur : Tuple avec une dimension différente de 5 : {label}")
            continue
        numbers = tuple(label)
        reformatted_labels.append(numbers)
    return np.array(reformatted_labels)

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

# Reformater les étiquettes pour l'entraînement et les tests
y_train = reformat_labels(y[:-10])  # Utiliser les données sauf les 10 dernières
y_test = reformat_labels(y[-10:])  # Utiliser les 10 dernières données

# Sélectionner toutes les lignes sauf les 10 dernières pour l'entraînement
X_train = X[:-10]
y_train = y_train

# Sélectionner les 10 dernières lignes pour les tests
X_test = X[-10:]
y_test = y_test

# Fonction pour créer le modèle Keras
def create_model():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(5))  # Modifier en 5 pour correspondre au nombre de numéros
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
    
    # Convertir les prédictions en tuples de 5 numéros
    predicted_tuples = [tuple(map(int, prediction)) for prediction in predictions]
    
    # Comparer les prédictions avec les 10 dernières lignes du fichier
    correct_predictions = 0
    for i in range(10):
        # Vous avez 5 numéros dans chaque tuple, donc on garde uniquement les 5 premiers
        if predicted_tuples[i] == tuple(map(int, y_test[i][:5])):
            correct_predictions += 1
    
    # Calculer le taux de réussite
    success_rate = correct_predictions / 10
    
    print('Taux de réussite à l\'itération', num_iterations, ':', success_rate)

print('Nombre d\'itérations nécessaires pour atteindre le seuil de réussite :', num_iterations)

# Intégrer les 10 dernières lignes dans l'apprentissage
X_train = np.concatenate((X_train, X_test), axis=0)
y_train = np.concatenate((y_train, y_test), axis=0)

# Réentraîner le modèle avec les nouvelles données
model = create_model()
model.fit(X_train, y_train, epochs=1, verbose=1)

# Faire une prédiction
def make_prediction(model):
    # Générer une prédiction
    prediction = model.predict(np.zeros((1, 5)))  # Ici, nous utilisons un tableau de zéros comme entrée, vous pouvez changer cela
    # Afficher la prédiction
    print('Prédiction :', tuple(map(int, prediction[0])))

print('Seuil de réussite atteint. Faisons une prédiction.')
make_prediction(model)
