import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

# Charger les données CSV et prétraiter
data = pd.read_csv('euromillions.csv', sep=';', header=None)

if data.shape[1] < 5:
    print("Le CSV doit avoir au moins 5 colonnes.")
    exit()

data = data.iloc[:, :5]  # Garder seulement les 5 premières colonnes
data.columns = [f'Num{i + 1}' for i in range(5)]  # Renommer les colonnes

# Fonction pour préparer les séquences
def prepare_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i + seq_length]
        label = data.iloc[i + seq_length]['Num1']
        sequences.append(seq.values)
        targets.append(label)
    return np.array(sequences), np.array(targets)

# Paramètres initiaux
best_accuracy = 0.0
best_precision = 0.0
best_model = None
best_epochs = 0
best_batch_size = 0
target_accuracy = 0.20  # Nouvelle cible de précision (20 %)

# Hyperparamètres à explorer
param_grid = {
    'epochs': [5120, 10240, 20480],
    'batch_size': [1228, 2456, 4912],
    'learning_rate': [0.001, 0.01, 0.1],
    'regularization': [0.001, 0.01, 0.1]
}

# Créer un fichier de résultats
with open("results.txt", "a") as results_file:
    results_file.write("Epochs, Batch Size, Accuracy, Precision\n")

    for epochs in param_grid['epochs']:
        for batch_size in param_grid['batch_size']:
            for learning_rate in param_grid['learning_rate']:
                for regularization in param_grid['regularization']:
                    # Préparer les séquences pour l'entraînement
                    seq_length = 10  # Vous pouvez choisir la longueur que vous préférez
                    X, y = prepare_sequences(data, seq_length)

                    # Diviser les données en ensembles d'entraînement et de test
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Créer un modèle de régression logistique multinomiale
                    input_layer = Input(shape=(seq_length, 5))
                    flatten = Flatten()(input_layer)
                    output_layer = Dense(34, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(regularization))(flatten)

                    model = Model(inputs=input_layer, outputs=output_layer)
                    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

                    # Créer un encodeur one-hot pour les étiquettes
                    encoder = OneHotEncoder(sparse=False)
                    y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
                    y_test_encoded = encoder.transform(y_test.reshape(-1, 1))

                    # Entraîner le modèle
                    model.fit(X_train, y_train_encoded, epochs=epochs, batch_size=batch_size, verbose=1)

                    # Prédire sur les données de test
                    predictions = model.predict(X_test)

                    # Évaluer le modèle
                    accuracy = accuracy_score(np.argmax(y_test_encoded, axis=1), np.argmax(predictions, axis=1))
                    precision = precision_score(np.argmax(y_test_encoded, axis=1), np.argmax(predictions, axis=1), average='weighted')
                    print(f'Taux de réussite avec {epochs} époques, {batch_size} taille de lot, LR {learning_rate}, Reg {regularization}: {accuracy}')

                    # Écrire les résultats dans le fichier
                    results_file.write(f"{epochs}, {batch_size}, {accuracy}, {precision}\n")

                    # Mettre à jour les meilleures métriques et le meilleur modèle
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_epochs = epochs
                        best_batch_size = batch_size

# Utiliser le meilleur modèle trouvé
model = best_model

# Prendre la dernière ligne du CSV comme entrée pour la prédiction
last_line_data = data.iloc[-1, :5].values
last_line = last_line_data.reshape(1, seq_length, 5)

# Prédire les prochains numéros basés sur la dernière ligne
predictions = model.predict(last_line)
predicted_number = np.argmax(predictions, axis=1)[0]

print('Meilleur taux de réussite atteint :', best_accuracy)
print('Meilleur learning rate :', best_learning_rate)
print('Meilleure régularisation :', best_regularization)
print('Prédiction du prochain numéro :', predicted_number)
