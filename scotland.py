import csv
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor
from keras.layers import Activation

# Charger les données depuis le fichier CSV
data = []
with open('euromillions.csv', 'r') as file:
    reader = csv.reader(file, delimiter=';')
    for row in reader:
        numbers = list(map(int, row))
        data.append(numbers)

# Préparer les données pour l'apprentissage
X = []
y = []
for i in range(len(data) - 1):
    X.append(data[i][:5])
    y.append(data[i + 1][:5])  # Les 5 numéros suivants sont la sortie
X = np.array(X)
y = np.array(y)

def find_optimal_hyperparameters(X_train, y_train):
    # Définition de la fonction pour créer le modèle

    def create_model_linear(learning_rate=0.001, epochs=50, batch_size=16):
        model = Sequential()
        model.add(LSTM(50, input_shape=(5, 1)))
        model.add(Dense(5))
        model.add(Activation('linear'))
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model
    
    def create_model_tanh(learning_rate=0.001, epochs=50, batch_size=16):
        model = Sequential()
        model.add(LSTM(50, input_shape=(5, 1)))
        model.add(Dense(5))
        model.add(Activation('tanh'))
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model
    
    def create_model_relu(learning_rate=0.001, epochs=50, batch_size=16):
        model = Sequential()
        model.add(LSTM(50, input_shape=(5, 1)))
        model.add(Dense(5))
        model.add(Activation('relu'))
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model
    
    # Hyperparamètres à explorer
    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'epochs': [50, 100, 200],
        'batch_size': [16, 32, 64]
    }
    
    # Liste des fonctions de création de modèle pour chaque valeur d'activation
    model_functions = [create_model_linear, create_model_tanh, create_model_relu]
    
    # Créer un modèle basé sur KerasRegressor pour la recherche d'hyperparamètres
    model = KerasRegressor(build_fn=model_functions[0], verbose=0)
    
    # Recherche des meilleures combinaisons d'hyperparamètres
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Obtenez les meilleurs hyperparamètres
    best_params = grid_search.best_params_

    # Créez un modèle avec les meilleurs hyperparamètres en utilisant la fonction de création de modèle correspondante
    final_model = KerasRegressor(build_fn=model_functions[best_params['activation']], verbose=0, **best_params)
    final_model.model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'])
    
    return final_model

# Exemple d'utilisation de la fonction
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
optimal_model = find_optimal_hyperparameters(X_train, y_train)

# Seuil de distance pour continuer l'apprentissage
seuil_distance = 5.0

while True:
    last_five_numbers = np.array(data[-1][:5]).reshape(1, 5, 1)
    next_numbers_prediction = optimal_model.predict(last_five_numbers)

    # Calcul de la distance euclidienne entre la prédiction et la dernière ligne du CSV
    distance = np.linalg.norm(next_numbers_prediction[0] - data[-1][:5])

    print("Prédiction pour les 5 prochains numéros :", next_numbers_prediction[0])
    print("Dernière ligne du CSV :", data[-1][:5])
    print("Distance euclidienne avec la dernière ligne du CSV :", distance)

    if distance < seuil_distance:
        break

print("Le modèle a atteint un résultat satisfaisant.")
