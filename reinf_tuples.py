import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Charger les données CSV et prétraiter
data = pd.read_csv('euromillions.csv', sep=';', header=None)

if data.shape[1] < 5:
    print("Le CSV doit avoir au moins 5 colonnes.")
    exit()
    
data = data.iloc[:, :5]  # Garder seulement les 5 premières colonnes
data.columns = [f'Num{i+1}' for i in range(5)]  # Renommer les colonnes

# Fonction pour préparer les séquences
def prepare_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i+seq_length]
        label = data.iloc[i+seq_length]['Num1']
        sequences.append(seq.values)
        targets.append(label)
    return np.array(sequences), np.array(targets)

# Préparer les séquences pour l'entraînement
seq_length = 10  # Vous pouvez choisir la longueur que vous préférez
X, y = prepare_sequences(data, seq_length)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Définir le modèle RNN
model = Sequential([
    LSTM(128, input_shape=(seq_length, 5)),
    Dense(1)
])

# Compiler le modèle
model.compile(optimizer=Adam(), loss='mean_squared_error')

# Initialiser le taux de réussite à un faible nombre pour entrer dans la boucle
accuracy = 0.0

# Boucle d'apprentissage jusqu'à atteindre le taux de réussite cible
while accuracy < 0.75:
    # Entraîner le modèle
    model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=1)
    
    # Prédire sur les données de test
    predictions = model.predict(X_test)
    rounded_predictions = [round(pred[0]) for pred in predictions]
    
    # Calculer le taux de réussite
    accuracy = accuracy_score(y_test, rounded_predictions)
    print('Taux de réussite :', accuracy)

# Prendre la dernière ligne du CSV comme entrée pour la prédiction
last_line = data.iloc[-1, :5].values.reshape(1, seq_length, 5)

# Prédire les prochains numéros basés sur la dernière ligne
predicted_number = int(model.predict(last_line)[0, 0])
print('Prédiction du prochain numéro :', predicted_number)
