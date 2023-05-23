import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

# Charger les données CSV
data = []
with open('euromillions.csv', 'r') as file:
    csv_reader = csv.reader(file, delimiter=';')
    for row in csv_reader:
        series = [int(num) for num in row[:5]]
        data.append(series)

# Diviser les données en ensembles d'apprentissage et de test
X = data[:-1]  # Séries d'apprentissage (toutes sauf la dernière)
y = [series[0] for series in data[1:]]  # Numéro cible (premier numéro de chaque série)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle SVM
model = SVR()
model.fit(X_train, y_train)

# Prédire le prochain numéro
next_series = data[-1]
next_number = model.predict([next_series])[0]

print("La prédiction du prochain numéro est :", next_number)
