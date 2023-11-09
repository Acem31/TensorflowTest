import csv
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras.layers import Activation
from kerastuner.tuners import RandomSearch
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from kerastuner.engine.hyperparameters import HyperParameters

# Charger les données depuis le fichier CSV
data = []
with open('euromillions.csv', 'r') as file:
    reader = csv.reader(file, delimiter=';')
    for row in reader:
        numbers = list(map(int, row[:5]))  # Utilisez les 5 premiers numéros pour former un tuple
        data.append(numbers)

# Préparer les données pour l'apprentissage
X = []
y = []
for i in range(len(data) - 1):
    X.append(data[i])
    y.append(data[i + 1])  # Les 5 numéros suivants sont la sortie
X = np.array(X)
y = np.array(y)

# Remodeler les données d'entraînement pour être en 3D
X = X.reshape(X.shape[0], X.shape[1], 1)
y = y.reshape(y.shape[0], y.shape[1], 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def build_hyper_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units_1', min_value=20, max_value=200, step=1), input_shape=(5, 1), return_sequences=True))
    dropout_rate_1 = hp.Float('dropout_rate_1', min_value=0.0, max_value=0.5, step=0.1)
    model.add(Dropout(rate=dropout_rate_1))  # Couche de régularisation
    model.add(LSTM(units=hp.Int('units_2', min_value=20, max_value=200, step=1), return_sequences=True))
    dropout_rate_2 = hp.Float('dropout_rate_2', min_value=0.0, max_value=0.5, step=0.1)
    model.add(Dropout(rate=dropout_rate_2))  # Couche de régularisation
    model.add(LSTM(units=hp.Int('units_3', min_value=20, max_value=200, step=1), return_sequences=True))
    dropout_rate_3 = hp.Float('dropout_rate_3', min_value=0.0, max_value=0.5, step=0.1)
    model.add(Dropout(rate=dropout_rate_3))
    model.add(Dense(1))
    model.add(Activation(hp.Choice('activation', values=['linear', 'tanh', 'relu'])))
    optimizer = Adam(learning_rate=hp.Float('learning_rate', min_value=0.0001, max_value=2, sampling='log'))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

tuner = RandomSearch(
    build_hyper_model,
    objective='val_loss',
    max_trials=532,  # Nombre de modèles à essayer
    directory='my_dir',  # Répertoire pour enregistrer les résultats
    project_name='euromillions'
)

# Divisez les données en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

tuner.search(X_train, y_train, epochs=200, validation_data=(X_val, y_val))
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Obtenez les meilleurs hyperparamètres
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Créez un modèle avec les meilleurs hyperparamètres
model = tuner.hypermodel.build(best_hps)
model.fit(X_train, y_train, epochs=200, batch_size=1032, validation_data=(X_val, y_val), callbacks=[early_stopping])

tuner.results_summary()

# Seuil de distance pour continuer l'apprentissage
seuil_distance = 5.0

while True:
    last_five_numbers = np.array(data[-1]).reshape(1, 5)
    next_numbers_prediction = model.predict(last_five_numbers)
    rounded_predictions = np.round(next_numbers_prediction[0])

    # Calcul de la distance euclidienne entre la prédiction et la dernière ligne du CSV
    distance = np.linalg.norm(rounded_predictions - data[-1])

    print("Prédiction pour les 5 prochains numéros :", rounded_predictions)
    print("Dernière ligne du CSV :", data[-1])
    print("Distance euclidienne avec la dernière ligne du CSV :", distance)

    if distance < seuil_distance:
        break

print("Le modèle a atteint un résultat satisfaisant.")
