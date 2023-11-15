import csv
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

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

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliser les données (il est important de normaliser pour XGBoost)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Créer et entraîner le modèle XGBoost
model = XGBRegressor(objective ='reg:squarederror')
model.fit(X_train, y_train)

# Seuil de distance pour continuer l'apprentissage
seuil_distance = 5.0

while True:
    last_five_numbers = np.array(data[-1]).reshape(1, -1)
    last_five_numbers = scaler.transform(last_five_numbers)  # Normaliser les données
    next_numbers_prediction = model.predict(last_five_numbers)
    rounded_predictions = np.round(next_numbers_prediction)

    # Calcul de la distance euclidienne entre la prédiction et la dernière ligne du CSV
    distance = np.linalg.norm(rounded_predictions - data[-1])

    print("Prédiction pour les 5 prochains numéros :", rounded_predictions)
    print("Dernière ligne du CSV :", data[-1])
    print("Distance euclidienne avec la dernière ligne du CSV :", distance)

    if distance < seuil_distance:
        break

    # Ré-entraîner le modèle avec les nouvelles données
    model.fit(X_train, y_train)

print("Le modèle a atteint un résultat satisfaisant.")
