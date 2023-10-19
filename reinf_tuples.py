import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import top_k_categorical_accuracy
import tensorflow as tf

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

def calculate_top_k_accuracy(y_true, y_pred, k=5):
    top_k = tf.nn.top_k(y_pred, k)
    top_k_indices = top_k.indices
    true_indices = tf.argmax(y_true, axis=1)
    matches = tf.equal(tf.expand_dims(true_indices, -1), top_k_indices)
    top_k_matches = tf.reduce_any(matches, axis=1)
    return tf.reduce_mean(tf.cast(top_k_matches, tf.float32))

# Paramètres initiaux
best_accuracy = 0.0
best_precision = 0.0
best_topk_accuracy = 0.0
best_model = None
best_epochs = 10
best_batch_size = 32
target_accuracy = 0.40

# Hyperparamètres
epochs = 10
batch_size = 32

# Créer un fichier de résultats
with open("results.txt", "w") as results_file:
    results_file.write("Epochs, Batch Size, Accuracy, Precision, Top-K Accuracy\n")

    while best_accuracy < target_accuracy:
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

        # Entraîner le modèle
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

        # Prédire sur les données de test
        predictions = model.predict(X_test)
        rounded_predictions = [round(pred[0]) for pred in predictions]
        
        # Calculer les métriques
        accuracy = accuracy_score(y_test, rounded_predictions)
        precision = precision_score(y_test, rounded_predictions, average='weighted')
        topk_accuracy = calculate_top_k_accuracy(y_test, predictions, k=5)
        print(f'Taux de réussite avec {epochs} époques et {batch_size} taille de lot : {accuracy}')
        
        # Écrire les résultats dans le fichier
        results_file.write(f"{epochs}, {batch_size}, {accuracy}, {precision}, {topk_accuracy}\n")

        # Mettre à jour les meilleures métriques et le meilleur modèle
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_precision = precision
            best_topk_accuracy = topk_accuracy
            best_model = model
            best_epochs = epochs
            best_batch_size = batch_size

        # Ajuster les hyperparamètres pour la prochaine itération
        epochs *= 2  # Double le nombre d'époques
        batch_size = int(batch_size * 1.5)  # Augmente la taille du lot de 50%

# Utiliser le meilleur modèle trouvé
model = best_model

# Prendre la dernière ligne du CSV comme entrée pour la prédiction
last_line = data.iloc[-1, :5].values.reshape(1, seq_length, 5)

# Prédire les prochains numéros basés sur la dernière ligne
predicted_number = int(model.predict(last_line)[0, 0])
print('Meilleur taux de réussite atteint :', best_accuracy)
print('Meilleur nombre d\'époques :', best_epochs)
print('Meilleure taille de lot :', best_batch_size)
print('Meilleure précision :', best_precision)
print('Meilleure Top-K Accuracy :', best_topk_accuracy)
print('Prédiction du prochain numéro :', predicted_number)
