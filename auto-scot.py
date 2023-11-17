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

# Initialiser une variable pour suivre la distance précédente
previous_distance = float('inf')

# Initialiser une variable pour compter le nombre d'itérations consécutives où la distance augmente
consecutive_increases = 0

# Seuil de distance pour continuer l'apprentissage
seuil_distance = 5.0

while True:
    # Prédiction avec le modèle
    next_numbers_prediction = best_model.predict(last_five_numbers.reshape(1, 1, -1))
    rounded_predictions = np.round(next_numbers_prediction)

    # Calcul de la distance euclidienne entre la prédiction et la dernière ligne du CSV
    distance = np.linalg.norm(rounded_predictions - data[-1])

    print("Prédiction pour les 5 prochains numéros :", rounded_predictions)
    print("Dernière ligne du CSV :", data[-1])
    print("Distance euclidienne avec la dernière ligne du CSV :", distance)

    if distance < seuil_distance:
        # Si la distance est inférieure au seuil, réinitialiser le compteur
        consecutive_increases = 0
    elif distance > previous_distance:
        # Si la distance augmente, incrémenter le compteur
        consecutive_increases += 1
    else:
        # Sinon, réinitialiser le compteur
        consecutive_increases = 0

    if consecutive_increases >= 3:
        # Si la distance augmente pendant trois itérations consécutives, arrêter la boucle
        print("La distance a augmenté pendant trois itérations consécutives. Arrêt de l'apprentissage.")
        break

    # Ré-entraîner le modèle avec les nouvelles données
    best_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=2)

    # Préparer les nouvelles données pour la prédiction
    last_five_numbers = np.array(data[-1]).reshape(1, 1, -1)
    last_five_numbers = np.squeeze([scaler.transform(last_five_numbers[:, i, :]) for i in range(last_five_numbers.shape[1])])

    # Mettre à jour la distance précédente
    previous_distance = distance
    
# Entraîner le meilleur modèle sur l'intégralité du CSV
best_model.fit(X, y, epochs=100, batch_size=32, verbose=2)

# Préparer les données pour la prédiction avec le modèle final
last_five_numbers = np.array(data[-1]).reshape(1, 1, -1)
last_five_numbers = np.squeeze([scaler.transform(last_five_numbers[:, i, :]) for i in range(last_five_numbers.shape[1])])

# Faire une prédiction avec le modèle final
final_prediction = best_model.predict(last_five_numbers.reshape(1, 1, -1))
rounded_final_prediction = np.round(final_prediction)

print("Prédiction finale pour les 5 prochains numéros :", rounded_final_prediction)
