import csv
import numpy as np
from sklearn.model_selection import train_test_split
from autokeras import StructuredDataRegressor, HyperParameters
from autokeras.tuners import BayesianOptimization
from keras.callbacks import EarlyStopping

# Charger les données depuis le fichier CSV
data = []
with open('euromillions.csv', 'r') as file:
    reader = csv.reader(file, delimiter=';')
    for row in reader:
        numbers = list(map(int, row[:5]))
        data.append(numbers)

# Préparer les données pour l'apprentissage
X = []
y = []
for i in range(len(data) - 1):
    X.append(data[i])
    y.append(data[i + 1])

X = np.array(X)
y = np.array(y)

# Remodeler les données d'entraînement pour être en 3D
X = X.reshape(X.shape[0], -1)
y = y.reshape(y.shape[0], -1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Utilisez AutoKeras pour la recherche de modèle
def search_space(hp: HyperParameters):
    model = StructuredDataRegressor(
        max_trials=150,
        directory='autokeras',
        objective='val_loss',
        overwrite=True,
        seed=42,
        hyperparameters=hp,
    )
    return model

# Exemple avec BayesianOptimization
tuner = BayesianOptimization(
    hypermodel=search_space,
    objective='val_loss',
    max_trials=150,
    num_initial_points=10,
    alpha=1e-4,
    directory='autokeras',
    overwrite=True,
)
clf = StructuredDataRegressor(tuner=tuner)

# Divisez les données en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Fit le modèle AutoKeras
clf.fit(X_train, y_train, epochs=200, validation_data=(X_val, y_val), callbacks=[EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)])

# Seuil de distance pour continuer l'apprentissage
seuil_distance = 5.0

while True:
    last_five_numbers = np.array(data[-1]).reshape(1, -1)
    next_numbers_prediction = clf.predict(last_five_numbers)
    rounded_predictions = np.round(next_numbers_prediction[0])

    # Calcul de la distance euclidienne entre la prédiction et la dernière ligne du CSV
    distance = np.linalg.norm(rounded_predictions - data[-1])

    print("Prédiction pour les 5 prochains numéros :", rounded_predictions)
    print("Dernière ligne du CSV :", data[-1])
    print("Distance euclidienne avec la dernière ligne du CSV :", distance)

    if distance < seuil_distance:
        break

print("Le modèle a atteint un résultat satisfaisant.")
