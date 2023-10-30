import csv
import lightgbm as lgb
import numpy as np

# Chargement des données
data = []
with open('euromillions.csv', 'r') as file:
    reader = csv.reader(file, delimiter=';')
    for row in reader:
        numbers = list(map(int, row[:5]))
        data.append(numbers)

# Créez un tableau pour stocker les prédictions de chaque colonne
predicted_columns = []

# Entraînez un modèle LightGBM pour chaque colonne
for col in range(5):
    # Préparez les données d'entraînement et de test
    X_train = [row[:col] for row in data[:-1]]
    y_train = [row[col] for row in data[1:]]
    X_test = data[-1][:col]  # Accédez à la colonne appropriée dans la dernière ligne

    # Créez un dataset LightGBM
    lgb_train = lgb.Dataset(X_train, y_train)

    # Paramètres du modèle LightGBM
    params = {
        "objective": "regression",
        "metric": "l2",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
    }

    # Entraînez le modèle
    num_round = 100
    bst = lgb.train(params, lgb_train, num_round)

    # Faites une prédiction pour la colonne actuelle
    predicted_value = int(bst.predict([X_test])[0])  # Utilisez [X_test] pour prédire une seule valeur
    predicted_columns.append(predicted_value)

# Affichez les prédictions finales
print(predicted_columns)
