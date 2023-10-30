import csv
import lightgbm as lgb
import pandas as pd
import numpy as np

# Chargement des données
data = []
with open('euromillions.csv', 'r') as file:
    reader = csv.reader(file, delimiter=';')
    for row in reader:
        numbers = tuple(map(int, row[:5]))
        data.append(numbers)

# Préparation des données pour LightGBM
df = pd.DataFrame(data, columns=['num1', 'num2', 'num3', 'num4', 'target'])
# 'target' représente la colonne 5 de la dernière ligne, mais ce n'est pas utilisé pour la prédiction

# Création du modèle LightGBM
models = []
for col in df.columns:
    if col != 'target':
        model = lgb.LGBMClassifier(num_leaves=31, objective='multiclass')
        model.fit(df.drop('target', axis=1), df[col])
        models.append(model)

# Récupération de la dernière ligne du CSV
last_row = data[-1]

# Utilisation du modèle pour prédire les cinq colonnes de la dernière ligne
predicted_numbers = [model.predict(pd.DataFrame([last_row], columns=['num1', 'num2', 'num3', 'num4']).drop('target', axis=1))[0] for model in models]

# Création d'un tuple avec les résultats
predicted_tuple = tuple(predicted_numbers)

print("Tuple prédit pour la dernière ligne du CSV:", predicted_tuple)
