import csv
import lightgbm as lgb
import pandas as pd

# Chargement des données
data = []
with open('euromillions.csv', 'r') as file:
    reader = csv.reader(file, delimiter=';')
    for row in reader:
        numbers = tuple(map(int, row[:5]))
        data.append(numbers)

# Préparation des données pour LightGBM
features = pd.DataFrame(data, columns=['num1', 'num2', 'num3', 'num4'])
target = [row[4] for row in data]

# Création du modèle LightGBM
model = lgb.LGBMClassifier(num_leaves=31, objective='multiclass')

# Entraînement du modèle
model.fit(features, target)

# Récupération de la dernière ligne du CSV
last_row = data[-1]

# Utilisation du modèle pour prédire le prochain numéro
last_row_features = pd.DataFrame([last_row], columns=['num1', 'num2', 'num3', 'num4'])
next_number = model.predict(last_row_features)[0]

print("Prochain numéro prédit:", next_number)
