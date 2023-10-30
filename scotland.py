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
df = pd.DataFrame(data, columns=['num1', 'num2', 'num3', 'num4', 'target'])

# Création du modèle LightGBM pour chaque colonne
models = {}
for col in df.columns:
    model = lgb.LGBMClassifier(num_leaves=31, objective='multiclass')
    model.fit(df.drop('target', axis=1), df[col])
    models[col] = model

# Récupération de la dernière ligne du CSV
last_row = data[-1]

# Utilisation des modèles pour prédire chaque colonne de la dernière ligne
predicted_numbers = {}
for col, model in models.items():
    predicted_numbers[col] = model.predict(pd.DataFrame([last_row], columns=['num1', 'num2', 'num3', 'num4', 'target'))[0]

# Création d'un tuple avec les résultats
predicted_tuple = (predicted_numbers['num1'], predicted_numbers['num2'], predicted_numbers['num3'], predicted_numbers['num4'], predicted_numbers['target'])

print("Tuple prédit pour la dernière ligne du CSV:", predicted_tuple)
