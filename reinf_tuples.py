import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# Ignorer les messages d'erreur TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

# Fonction pour évaluer un modèle avec des hyperparamètres donnés
def evaluate_model(params):
    # Paramètres d'hyperparamètres
    epochs = params['epochs']
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    regularization = params['regularization']

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
    model.fit(X_train, y_train_encoded, epochs=epochs, batch_size=batch_size, verbose=0)

    # Prédire sur les données de test
    predictions = model.predict(X_test)

    # Évaluer le modèle
    accuracy = accuracy_score(np.argmax(y_test_encoded, axis=1), np.argmax(predictions, axis=1))
    precision = precision_score(np.argmax(y_test_encoded, axis=1), np.argmax(predictions, axis=1), average='weighted')

    return {'loss': -accuracy, 'status': STATUS_OK}

# Espace de recherche pour les hyperparamètres
space = {
    'epochs': hp.choice('epochs', [5120, 10240, 20480]),
    'batch_size': hp.choice('batch_size', [1228, 2456, 4912]),
    'learning_rate': hp.loguniform('learning_rate', -5, -1),
    'regularization': hp.loguniform('regularization', -6, -2)
}

# Créer un objet Trials pour suivre les résultats
trials = Trials()

# Utiliser l'optimisation bayésienne pour trouver les meilleurs hyperparamètres
best = fmin(fn=evaluate_model, space=space, algo=tpe.suggest, max_evals=100, trials=trials)

# Extraire les meilleurs hyperparamètres
best_epochs = param_grid['epochs'][best['epochs']]
best_batch_size = param_grid['batch_size'][best['batch_size']]
best_learning_rate = best['learning_rate']
best_regularization = best['regularization']

# Utiliser les meilleurs hyperparamètres pour entraîner le modèle
best_params = {
    'epochs': best_epochs,
    'batch_size': best_batch_size,
    'learning_rate': best_learning_rate,
    'regularization': best_regularization
}
best_model = evaluate_model(best_params)
print('Meilleur taux de réussite atteint :', -best_model['loss'])

# Utiliser le meilleur modèle trouvé
model = best_model

# Prendre la dernière ligne du CSV comme entrée pour la prédiction
last_line_data = data.iloc[-1, :5].values
last_line = last_line_data.reshape(1, seq_length, 5)

# Prédire les prochains numéros basés sur la dernière ligne
predictions = model.predict(last_line)
predicted_number = np.argmax(predictions, axis=1)[0]

print('Meilleur nombre d\'époques :', best_epochs)
print('Meilleure taille de lot :', best_batch_size)
print('Prédiction du prochain numéro :', predicted_number)
