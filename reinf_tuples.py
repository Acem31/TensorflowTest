import os
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Ignorer les messages d'erreur TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Charger les données CSV et prétraiter
data = pd.read_csv('euromillions.csv', sep=';', header=None)

if data.shape[1] < 5:
    print("Le CSV doit avoir au moins 5 colonnes.")
    exit()

data = data.iloc[:, :5]  # Garder seulement les 5 premières colonnes
data.columns = [f'Num{i + 1}' for i in range(5)]  # Renommer les colonnes

# Créer un modèle pour générer des prédictions
def create_model(seq_length):
    input_layer = Input(shape=(seq_length, 5))
    flatten = Flatten()(input_layer)
    output_layer = Dense(5, activation='softmax')(flatten)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Préparer les données d'entraînement
def prepare_training_data(data, seq_length):
    sequences = []
    next_tuples = []  # Liste pour stocker les tuples de 5 numéros
    
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i + seq_length]
        next_nums = data.iloc[i + seq_length:i + seq_length + 5].values
        sequences.append(seq.values)
        
        # Stocker les 5 numéros sous forme de tuple
        next_tuple = tuple(next_nums[0])
        next_tuples.append(next_tuple)
    
    X = np.array(sequences)
    y = np.array(next_tuples)
    
    return X, y

# Entraîner le modèle
seq_length = 10  # Longueur de la séquence
X, y = prepare_training_data(data, seq_length)
model = create_model(seq_length)

# Entraîner le modèle
model.fit(X, y, epochs=100, batch_size=32)

# Générer une prédiction pour les prochains numéros
def generate_prediction(model, last_rows, seq_length):
    last_rows = np.array(last_rows).reshape(1, seq_length, 5)
    predictions = model.predict(last_rows)
    return predictions

# Prédire un ensemble de 5 numéros pour les 10 dernières lignes du CSV
last_rows = data.iloc[-seq_length:].values
predictions = generate_prediction(model, last_rows, seq_length)
# Sélectionnez les 5 numéros les plus probables
top_predictions = predictions[0].argsort()[-5:][::-1]
print("Prédiction des prochains 5 numéros :", top_predictions)
