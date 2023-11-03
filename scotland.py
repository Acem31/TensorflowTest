import csv
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras.layers import Activation
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def build_hyper_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units', min_value=20, max_value=100, step=1), input_shape=(5, 1)))
    model.add(Dense(5))
    model.add(Activation(hp.Choice('activation', values=['linear', 'tanh', 'relu'])))
    optimizer = Adam(learning_rate=hp.Float('learning_rate', min_value=0.0001, max_value=0.1, sampling='log'))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

tuner = RandomSearch(
    build_hyper_model,
    objective='val_loss',
    max_trials=100,  # Nombre de modèles à essayer
    directory='my_dir',  # Répertoire pour enregistrer les résultats
    overwrite=False  # Assurez-vous que les résultats précédents ne sont pas écrasés
)

# Divisez les données en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

tuner.search(X_train, y_train, epochs=hp.get('epochs'), validation_data=(X_val, y_val))

# Obtenez les meilleurs hyperparamètres
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Créez un modèle avec les meilleurs hyperparamètres
model = tuner.hypermodel.build(best_hps)
model.fit(X_train, y_train, epochs=best_hps.get('epochs'), batch_size=best_hps.get('batch_size'))

tuner.results_summary()

# Seuil de distance pour continuer l'apprentissage
seuil_distance = 5.0

while True:
    last_five_numbers = np.array(data[-1][:5]).reshape(1, 5, 1)
    next_numbers_prediction = model.predict(last_five_numbers)

    # Calcul de la distance euclidienne entre la prédiction et la dernière ligne du CSV
    distance = np.linalg.norm(next_numbers_prediction[0] - data[-1][:5])

    print("Prédiction pour les 5 prochains numéros :", next_numbers_prediction[0])
    print("Dernière ligne du CSV :", data[-1][:5])
    print("Distance euclidienne avec la dernière ligne du CSV :", distance)

    if distance < seuil_distance:
        break

print("Le modèle a atteint un résultat satisfaisant.")
