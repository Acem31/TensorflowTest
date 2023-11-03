import csv
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor
from keras.layers import Activation
from scipy.stats import uniform, randint

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

def create_model(learning_rate=0.001, activation='linear', units=50):
    model = Sequential()
    model.add(LSTM(units, input_shape=(5, 1))
    model.add(Dense(5))
    model.add(Activation(activation))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

# Hyperparamètres à explorer
param_dist = {
    'learning_rate': uniform(0.0001, 0.1),
    'activation': ['linear', 'tanh', 'relu'],
    'units': randint(20, 100),
    'epochs': randint(50, 201),
    'batch_size': randint(16, 65)
}

# Créer un modèle basé sur KerasRegressor pour la recherche d'hyperparamètres
model = KerasRegressor(build_fn=create_model, verbose=0)

# Recherche des meilleures combinaisons d'hyperparamètres avec RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                   n_iter=100, scoring='neg_mean_squared_error', n_jobs=-1)
random_search.fit(X_train, y_train)

# Obtenez les meilleurs hyperparamètres
best_params = random_search.best_params_

# Créez un modèle avec les meilleurs hyperparamètres
final_model = KerasRegressor(build_fn=create_model, verbose=0, **best_params)
final_model.model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'])

# Seuil de distance pour continuer l'apprentissage
seuil_distance = 5.0

while True:
    last_five_numbers = np.array(data[-1][:5]).reshape(1, 5, 1)
    next_numbers_prediction = final_model.predict(last_five_numbers)

    # Calcul de la distance euclidienne entre la prédiction et la dernière ligne du CSV
    distance = np.linalg.norm(next_numbers_prediction[0] - data[-1][:5])

    print("Prédiction pour les 5 prochains numéros :", next_numbers_prediction[0])
    print("Dernière ligne du CSV :", data[-1][:5])
    print("Distance euclidienne avec la dernière ligne du CSV :", distance)

    if distance < seuil_distance:
        break

print("Le modèle a atteint un résultat satisfaisant.")
