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
features = pd.DataFrame(data, columns=['num1', 'num2', 'num3', 'num4', 'target'])
# 'target' représente la colonne 5 de la dernière ligne, mais ce n'est pas utilisé pour la prédiction

# Création du modèle LightGBM
model = lgb.LGBMClassifier(num_leaves=31, objective='multiclass')

# Entraînement du modèle
model.fit(features.drop('target', axis=1), features['target'])

# Récupération de la dernière ligne du CSV
last_row = data[-1]

# Utilisation du modèle pour prédire les cinq colonnes de la dernière ligne
last_row_features = pd.DataFrame([last_row], columns=['num1', 'num2', 'num3', 'num4', 'target'])
predicted_numbers = model.predict(last_row_features.drop('target', axis=1))

print("Colonnes prédites pour la dernière ligne du CSV:", predicted_numbers)
