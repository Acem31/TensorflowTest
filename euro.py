import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Chargement des données depuis le fichier CSV
data = pd.read_csv("euromillions.csv", sep=";", header=None)

# Sélection des 5 premières colonnes
data = data.iloc[:, :5]

# Conversion de la colonne cible en chaînes de caractères
data.iloc[:, 4] = data.iloc[:, 4].astype(str)

# Suppression des lignes contenant des valeurs NaN
data = data.dropna()

# Séparation des fonctionnalités (X) et de la variable cible (y)
X = data.iloc[:, :4]
y = data.iloc[:, 4]

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création d'une liste pour stocker les prédictions de chaque chiffre
predictions = []

# Entraînement et prédiction pour chaque chiffre de la séquence
for i in range(5):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train.str[i])
    y_pred = model.predict(X_test)
    predictions.append(y_pred)

# Calcul de l'erreur quadratique moyenne pour l'ensemble des prédictions
mse = mean_squared_error(y_test, pd.DataFrame(predictions).T)
print("MSE (Mean Squared Error):", mse)

# Prédiction avec le modèle optimisé
new_data = pd.DataFrame([[16, 29, 32, 36], [7, 13, 39, 47]], columns=X.columns)
new_data = new_data.astype(str)
new_predictions = []

# Prédiction pour chaque chiffre de la séquence
for i in range(5):
    new_pred = model.predict(new_data)
    new_predictions.append(new_pred)

print("Prédictions:", new_predictions)
