import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, AdamW, Adadelta, Adagrad, Adamax, Adafactor, Nadam, Ftrl
from keras_tuner.tuners import BayesianOptimization
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
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
sequence_length = 7

for i in range(len(sequences) - sequence_length):
    X.append(sequences[i:i+sequence_length])
    y.append(sequences[i+sequence_length])
X = np.array(X)
y = np.array(y)
# Diviser les données en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

def custom_loss(y_true, y_pred):
    # Split y_pred into mean and std
    mean_pred = y_pred[:, :X.shape[2]]
    std_pred = y_pred[:, X.shape[2]:]

    if mean_pred.shape != (None, 7):
        raise ValueError("Invalid shape for mean prediction. Expected (1, 7).")

    # Use mean_squared_error for the mean part
    mean_loss = mean_squared_error(y_true, mean_pred)

    # Calculate the loss for the std part (you may use a different loss here)
    std_loss = tf.reduce_mean(tf.square(std_pred - tf.math.reduce_std(y_true, axis=1, keepdims=True)))

    if tf.reduce_any(tf.math.is_nan(mean_loss)) or tf.reduce_any(tf.math.is_nan(std_loss)):
    # Return a very high loss to indicate that this configuration is not valid
        return 1e6
    else:
    # Return the combined loss
        return mean_loss + std_loss

def build_hyper_model(hp):
    model = Sequential()
    model.add(LSTM(
        units=hp.Int('units', min_value=7, max_value=700, step=1),
        activation=hp.Choice('lstm_activation', values=['softmax', 'softplus', 'softsign', 'tanh', 'selu', 'elu', 'exponential', 'leaky_relu', 'relu6', 'silu', 'gelu', 'hard_sigmoid', 'linear', 'mish', 'log_softmax']),
        input_shape=(sequence_length, X.shape[2])
    ))
    # Ajouter la première couche Dense supplémentaire
    model.add(Dense(
        units=hp.Int('dense_units_1', min_value=7, max_value=700, step=1),
        activation=hp.Choice('Dense_activation_1', values=['softmax', 'softplus', 'softsign', 'tanh', 'selu', 'elu', 'exponential', 'leaky_relu', 'relu6', 'silu', 'gelu', 'hard_sigmoid', 'linear', 'mish', 'log_softmax'])
    ))

    # Ajouter la deuxième couche Dense supplémentaire
    model.add(Dense(
        units=hp.Int('dense_units_2', min_value=7, max_value=700, step=1),
        activation=hp.Choice('Dense_activation_2', values=['softmax', 'softplus', 'softsign', 'tanh', 'selu', 'elu', 'exponential', 'leaky_relu', 'relu6', 'silu', 'gelu', 'hard_sigmoid', 'linear', 'mish', 'log_softmax'])
    ))

    # Couche Dense originale (valeur moyenne)
    model.add(Dense(
        units=hp.Int('dense_units', min_value=7, max_value=700, step=1),
        activation=hp.Choice('Dense_activation', values=['softmax', 'softplus', 'softsign', 'tanh', 'selu', 'elu', 'exponential', 'leaky_relu', 'relu6', 'silu', 'gelu', 'hard_sigmoid', 'linear', 'mish', 'log_softmax'])
    ))

    # Sortie pour la valeur moyenne
    mean_output = Dense(X.shape[2], name='mean_output')(model.layers[-1].output)

    # Sortie pour l'écart-type (activation softplus pour des valeurs positives)
    std_output = Dense(X.shape[2], activation='softplus', name='std_output')(model.layers[-1].output)

    # Concaténer les deux sorties
    final_output = Concatenate(name='final_output')([mean_output, std_output])

    # Compiler le modèle
    model = Model(inputs=model.input, outputs=final_output)
    optimizer_choice = hp.Choice('optimizer', values=['SGD', 'RMSprop', 'Adam', 'AdamW', 'Adadelta', 'Adagrad', 'Adamax', 'Adafactor', 'Nadam', 'Ftrl'])
    
    if optimizer_choice == 'SGD':
        optimizer = SGD(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log'))
    elif optimizer_choice == 'RMSprop':
        optimizer = RMSprop(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log'))
    elif optimizer_choice == 'Adam':
        optimizer = Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log'))
    elif optimizer_choice == 'AdamW':
        optimizer = AdamW(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log'))
    elif optimizer_choice == 'Adadelta':
        optimizer = Adadelta(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log'))
    elif optimizer_choice == 'Adagrad':
        optimizer = Adagrad(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log'))
    elif optimizer_choice == 'Adamax':
        optimizer = Adamax(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log'))
    elif optimizer_choice == 'Adafactor':
        optimizer = Adafactor(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log'))
    elif optimizer_choice == 'Nadam':
        optimizer = Nadam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log'))
    elif optimizer_choice == 'Ftrl':
        optimizer = Ftrl(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log'))
    model.compile(optimizer=optimizer, loss=custom_loss)
    return model

# Initialiser le tuner BayesianOptimization
tuner = BayesianOptimization(
    build_hyper_model,
    objective='val_loss',
    num_initial_points=10,
    alpha=1e-4,
    beta=2.6,
    max_trials=200
)

# Rechercher les meilleurs hyperparamètres
tuner.search(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

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

# Arrondir les numéros prédits
predicted_numbers_rounded = np.round(predicted_numbers)

print("Numéros prédits (arrondis) pour le prochain tirage:")
print(predicted_numbers_rounded)
