import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Charger les données CSV
data = []
with open('euromillions.csv', 'r') as file:
    csv_reader = csv.reader(file, delimiter=';')
    for row in csv_reader:
        series = [int(num) for num in row[:5]]
        data.append(series)

# Convertir les séries de chiffres en listes
X = [series[:-1] for series in data[:-1]]  # Séries d'apprentissage (toutes sauf la dernière)
y = [series[-1] for series in data[:-1]]   # Numéro cible correspondant au dernier chiffre de chaque série

# Diviser les données en ensembles d'apprentissage et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle SVM
model = SVC()
model.fit(X_train, y_train)

# Prédire la prochaine série
next_series = data[-1][:-1]
predicted_number = model.predict([next_series])[0]

print("La prédiction du prochain numéro est :", predicted_number)
