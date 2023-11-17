import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from kerastuner.tuners import RandomSearch

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

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliser les données (il est important de normaliser pour le Deep Learning)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Réorganiser les données pour qu'elles soient compatibles avec un modèle RNN
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])  # Ajoute une dimension temporelle
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Fonction pour construire le modèle
def build_model(hp):
    model = Sequential()
    model.add(GRU(units=hp.Int('units', min_value=32, max_value=128, step=32), input_shape=(1, 5), activation='relu'))
    model.add(Dense(5))
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])))
    return model

# Initialiser le tuner
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,  # Nombre total de modèles à essayer
    directory='tuner_dir',  # Répertoire pour enregistrer les résultats du tuner
    project_name='euromillions_tuning'  # Nom du projet tuner
)

# Rechercher les meilleurs paramètres
tuner.search(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[EarlyStopping(patience=5)])

# Obtenir les meilleurs paramètres
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Construire le modèle avec les meilleurs paramètres
best_model = tuner.hypermodel.build(best_hps)

# Entraîner le meilleur modèle
best_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Normaliser les données pour la prédiction
last_five_numbers = np.array(data[-1]).reshape(1, 1, -1)
last_five_numbers = np.squeeze([scaler.transform(last_five_numbers[:, i, :]) for i in range(last_five_numbers.shape[1])])

# Seuil de distance pour continuer l'apprentissage
seuil_distance = 10.0
distance = 15

while distance > seuil_distance:
    # Prédiction avec le modèle
    next_numbers_prediction = best_model.predict(last_five_numbers.reshape(1, 1, -1))
    rounded_predictions = np.round(next_numbers_prediction)

    # Calcul de la distance euclidienne entre la prédiction et la dernière ligne du CSV
    distance = np.linalg.norm(rounded_predictions - data[-1])

    print("Prédiction pour les 5 prochains numéros :", rounded_predictions)
    print("Dernière ligne du CSV :", data[-1])
    print("Distance euclidienne avec la dernière ligne du CSV :", distance)

    if distance < seuil_distance:
        # Charger toutes les données du fichier CSV
        all_data = []
        with open('euromillions.csv', 'r') as file:
            reader = csv.reader(file, delimiter=';')
            for row in reader:
                numbers = list(map(int, row[:5]))
                all_data.append(numbers)

        # Ajouter les nouvelles données au jeu de données existant
        X_extended = np.concatenate((X, np.array(all_data[:-1])))
        y_extended = np.concatenate((y, np.array(all_data[1:])))

        # Diviser les données étendues en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X_extended, y_extended, test_size=0.2, random_state=42)

        # Normaliser les données
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Réorganiser les données pour qu'elles soient compatibles avec un modèle RNN
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

        # Réentraîner le modèle avec les données étendues
        best_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

        # Réinitialiser le processus de prédiction avec la dernière ligne du CSV
        last_five_numbers = np.array(all_data[-1]).reshape(1, 1, -1)
        last_five_numbers = np.squeeze([scaler.transform(last_five_numbers[:, i, :]) for i in range(last_five_numbers.shape[1])])

    else:
        # Ré-entraîner le modèle avec les nouvelles données
        best_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=2)

        # Préparer les nouvelles données pour la prédiction
        last_five_numbers = np.array(data[-1]).reshape(1, 1, -1)
        last_five_numbers = np.squeeze([scaler.transform(last_five_numbers[:, i, :]) for i in range(last_five_numbers.shape[1])])

# Une dernière prédiction après la fin de la boucle
final_prediction = best_model.predict(last_five_numbers.reshape(1, 1, -1))
final_rounded_prediction = np.round(final_prediction)

print("Dernière prédiction pour les 5 prochains numéros :", final_rounded_prediction)
