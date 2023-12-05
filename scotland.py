import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Concatenate
from kerastuner.tuners import BayesianOptimization
from tensorflow.keras.callbacks import EarlyStopping

# Charger les données
data = pd.read_csv('euromillions.csv', sep=';', header=None)

# Prétraitement des données
main_numbers = data.iloc[:, 0:6]
bonus_numbers = data.iloc[:, 6:8]
sequences = pd.concat([main_numbers, bonus_numbers], axis=1)

# Normaliser les données
scaler = MinMaxScaler()
sequences = scaler.fit_transform(sequences)

# Préparer les données pour l'apprentissage
X, y = [], []
sequence_length = 2

for i in range(len(sequences) - sequence_length):
    X.append(sequences[i:i+sequence_length])
    y.append(sequences[i+sequence_length])

X = np.array(X)
y = np.array(y)

# Diviser les données en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

def build_hyper_model(hp):
    model = Sequential()
    model.add(LSTM(
        units=hp.Int('units', min_value=10, max_value=100, step=1),
        activation=hp.Choice('lstm_activation', values=['relu', 'tanh', 'sigmoid']),
        input_shape=(sequence_length, X.shape[2])
    ))
    model.add(Dense(
        units=hp.Int('dense_units', min_value=10, max_value=100, step=1),
        activation=hp.Choice('Dense_activation', values=['relu', 'tanh', 'sigmoid'])
    ))
    
    # Sortie pour la valeur moyenne
    mean_output = Dense(X.shape[2], name='mean_output')(model.layers[-1].output)
    
    # Sortie pour l'écart-type (activation softplus pour des valeurs positives)
    std_output = Dense(X.shape[2], activation='softplus', name='std_output')(model.layers[-1].output)

    # Concaténer les deux sorties
    final_output = Concatenate(name='final_output')([mean_output, std_output])

    # Compiler le modèle
    model = Model(inputs=model.input, outputs=final_output)
    optimizer_choice = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])
    optimizer = 'adam' if optimizer_choice == 'adam' else ('sgd' if optimizer_choice == 'sgd' else 'rmsprop')
    model.compile(optimizer=optimizer, loss='mse')
    return model

# Initialiser le tuner BayesianOptimization
tuner = BayesianOptimization(
    build_hyper_model,
    objective='val_loss',
    num_initial_points=10,
    alpha=1e-4,
    beta=2.6,
    max_trials=100
)

# Rechercher les meilleurs hyperparamètres
tuner.search(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Récupérer le modèle avec les meilleurs hyperparamètres
best_model = tuner.get_best_models(num_models=1)[0]

# Évaluer le modèle sur l'ensemble de test
loss = best_model.evaluate(X_test, y_test)
print(f"Loss on test set: {loss}")

# Faire une prédiction pour le prochain tirage
last_sequence = sequences[-sequence_length:].reshape(1, sequence_length, X.shape[2])
predicted_values = best_model.predict(last_sequence)

# Récupérer les prédictions de la valeur moyenne et de l'écart-type
predicted_mean = predicted_values[:, :X.shape[2]]
predicted_std = predicted_values[:, X.shape[2]:]

# Ajouter l'écart-type aux prédictions de la valeur moyenne
predicted_numbers = predicted_mean + predicted_std * np.random.normal(size=predicted_mean.shape)

# Inverser la normalisation pour obtenir les numéros prédits
predicted_numbers = scaler.inverse_transform(predicted_numbers)

print("Numéros prédits pour le prochain tirage:")
print(predicted_numbers)
