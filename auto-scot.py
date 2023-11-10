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

# Utiliser AutoKeras pour la classification avec CNN et sortie de probabilité
# Remarque : Il est recommandé d'utiliser les modèles ImageRegressor ou ImageClassifier
# pour les tâches impliquant des données structurées avec une représentation image-like.
# Utilisez la transformation 'reshape' pour préparer les données comme une image.
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Utiliser AutoKeras pour la classification
clf = ak.ImageClassifier(
    max_trials=150,
    overwrite=True,
    seed=42,
    objective="val_loss",
    multilabel=True,  # Pour la sortie de probabilité
    metrics=['accuracy'],  # Ajout de la métrique accuracy
)

# Rechercher le meilleur modèle
clf.fit(X_train, y_train, epochs=200, validation_split=0.2)

# Obtenir les prédictions de probabilité sur l'ensemble de test
probabilities = clf.predict(X_test, return_probabilities=True)
print(probabilities)
