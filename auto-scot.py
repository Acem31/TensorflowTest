import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

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

# Normaliser les données (il est important de normaliser pour le Deep Learning)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Réorganiser les données pour qu'elles soient compatibles avec un modèle RNN
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])  # Ajoute une dimension temporelle
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Créer le modèle GRU
model = Sequential()
model.add(GRU(64, input_shape=(1, 5), activation='relu'))
model.add(Dense(5))  # Le nombre de neurones de sortie doit être égal à la dimension de la sortie

# Compiler le modèle
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))

# Entraîner le modèle
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2)

# Seuil de distance pour continuer l'apprentissage
seuil_distance = 5.0

while True:
    last_five_numbers = np.array(data[-1]).reshape(1, 1, -1)
    last_five_numbers = np.squeeze([scaler.transform(last_five_numbers[:, i, :]) for i in range(last_five_numbers.shape[1])])
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
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=2)

print("Le modèle a atteint un résultat satisfaisant.")
