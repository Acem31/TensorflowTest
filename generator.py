import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Lire le CSV
data = pd.read_csv('euromillions.csv', sep=';', header=None)

# Séparer les données en fonction des colonnes d'entrée (X) et des colonnes de sortie (y)
X = data.iloc[:, 0:5]
y = data.iloc[:, 0:5]

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer un modèle RandomForestRegressor avec des hyperparamètres prédéfinis
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Liste pour stocker les prédictions pour chaque colonne
predictions = []

# Entraîner le modèle et faire des prédictions pour chaque colonne
for col in range(5):
    model.fit(X_train, y_train.iloc[:, col])
    y_pred = model.predict(X_test)
    predictions.append(y_pred)

# Calculer la précision pour chaque colonne (RMSE)
accuracies = [mean_squared_error(y_test.iloc[:, col], pred, squared=False) for col, pred in enumerate(predictions)]

# Imprimer les prédictions et précisions pour chaque colonne
for col in range(5):
    print(f"Prédiction pour la colonne {col + 1} :", predictions[col])
    print(f"Précision de la prédiction pour la colonne {col + 1} (RMSE) :", accuracies[col])
