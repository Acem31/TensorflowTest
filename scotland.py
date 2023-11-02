import csv
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Charger les données depuis le fichier CSV
data = []
with open('euromillions.csv', 'r') as file:
    reader = csv.reader(file, delimiter=';')
    for row in reader:
        numbers = list(map(int, row))
        data.append(numbers)

# Préparer les données pour l'apprentissage
X = []
y = []
for i in range(len(data) - 1):
    X.append(data[i][:5])
    y.append(data[i + 1][:5])  # Les 5 numéros suivants sont la sortie
X = np.array(X)
y = np.array(y)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner un modèle LSTM
model = Sequential()
model.add(LSTM(50, input_shape=(5, 1)))
model.add(Dense(5, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')

# Seuil de distance pour continuer l'apprentissage
seuil_distance = 5.0

while True:
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    
    last_five_numbers = np.array(data[-1][:5]).reshape(1, 5, 1)
    next_numbers_prediction = model.predict(last_five_numbers)
    
    # Calcul de la distance euclidienne entre la prédiction et la dernière ligne du CSV
    distance = np.linalg.norm(next_numbers_prediction[0] - data[-1][:5])
    
    print("Prédiction pour les 5 prochains numéros :", next_numbers_prediction[0])
    print("Distance euclidienne avec la dernière ligne du CSV :", distance)
    
    if distance < seuil_distance:
        break

print("Le modèle a atteint un résultat satisfaisant.")
