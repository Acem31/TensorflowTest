import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate
from kerastuner.tuners import BayesianOptimization
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder

# Charger les données
data = pd.read_csv('euromillions.csv', sep=';', header=None)

# Prétraitement des données
main_numbers = data.iloc[:, 0:6]
bonus_numbers = data.iloc[:, 6:8]
sequences = pd.concat([main_numbers, bonus_numbers], axis=1)

# Normaliser les données
scaler = StandardScaler()
sequences = scaler.fit_transform(sequences)

# Préparer les données pour l'apprentissage
X, y = [], []
sequence_length = 10

for i in range(len(sequences) - sequence_length):
    X.append(sequences[i:i+sequence_length])
    y.append(sequences[i+sequence_length])

X = np.array(X)
y = np.array(y)

# Diviser les données en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Utiliser mlxtend pour extraire les motifs fréquents
te = TransactionEncoder()
te_ary = te.fit(X.flatten()).transform(X.flatten())
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)

# Ajouter des caractéristiques basées sur les motifs fréquents
pattern_features = frequent_itemsets.values[:, 1:]  # Utiliser les motifs fréquents comme caractéristiques
X_with_patterns = np.concatenate((X, pattern_features[:, np.newaxis, :]), axis=2)

def build_hyper_model(hp):
    lstm_model = Sequential()
    lstm_model.add(LSTM(hp.Int('units', min_value=10, max_value=100, step=1), activation='relu', input_shape=(sequence_length, X.shape[2])))
    lstm_model.add(Dense(hp.Int('dense_units', min_value=10, max_value=100, step=1), activation='relu'))

    # Inclure l'optimizer comme hyperparamètre
    optimizer_choice = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])
    
    # Initialiser optimizer avec une valeur par défaut (par exemple, 'adam')
    optimizer = 'adam'

    if optimizer_choice == 'sgd':
        optimizer = 'sgd'
    elif optimizer_choice == 'rmsprop':
        optimizer = 'rmsprop'
    
    lstm_model.compile(optimizer=optimizer, loss='mse')

    # Modèle basé sur les motifs fréquents
    input_patterns = Input(shape=(5,))  # Adapter à votre nombre de motifs
    dense_patterns = Dense(10, activation='relu')(input_patterns)
    pattern_model = Model(inputs=input_patterns, outputs=dense_patterns)

    # Concaténer les sorties des deux modèles
    combined_model = Concatenate()([lstm_model.output, pattern_model.output])

    # Ajouter une couche dense supplémentaire si nécessaire
    combined_model = Dense(10, activation='relu')(combined_model)

    # Couche de sortie
    output_layer = Dense(X.shape[2])(combined_model)

    model = Model(inputs=[lstm_model.input, input_patterns], outputs=output_layer)
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
tuner.search([X_train, pattern_features[y_train.index]], y_train, epochs=100, batch_size=32, validation_data=([X_test, pattern_features[y_test.index]], y_test))

# Récupérer le modèle avec les meilleurs hyperparamètres
best_model = tuner.get_best_models(num_models=1)[0]

# Évaluer le modèle sur l'ensemble de test
loss = best_model.evaluate([X_test, pattern_features[y_test.index]], y_test)
print(f"Loss on test set: {loss}")

# Faire une prédiction pour le prochain tirage
last_sequence = sequences[-sequence_length:].reshape(1, sequence_length, X.shape[2])
last_pattern_features = np.random.rand(1, 5)  # Exemple aléatoire
predicted_numbers = best_model.predict([last_sequence, last_pattern_features])

# Inverser la normalisation pour obtenir les numéros prédits
predicted_numbers = scaler.inverse_transform(predicted_numbers)

print("Numéros prédits pour le prochain tirage:")
print(predicted_numbers)
