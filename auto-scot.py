import autokeras as ak
import csv
import numpy as np
from sklearn.model_selection import train_test_split

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

# Divisez les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Utiliser AutoKeras pour la régression avec sortie de probabilité
regression = ak.StructuredDataRegressor(max_trials=150, overwrite=True, seed=42, objective="val_accuracy")

# Rechercher le meilleur modèle
regression.fit(X_train, y_train, epochs=200, validation_split=0.2)

# Obtenir les prédictions de probabilité sur l'ensemble de test
probabilities = regression.predict(X_test, return_probabilities=True)
print(probabilities)
