import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense

data = pd.read_csv('euromillions.csv')

train_data = data.iloc[:-5]  # Toutes les lignes sauf les 5 dernières
test_data = data.iloc[-5:]   # Les 5 dernières lignes

def create_model():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(1))  # Pour la régression
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

learning_rate = 0.01
discount_factor = 0.95

target_success_rate = 0.75

model = create_model()  # Utilisez votre fonction create_model()

success_rate = 0.0
num_iterations = 0

while success_rate < target_success_rate:
    num_iterations += 1
    
    predictions = model.predict(X_test)
    predictions = predictions.flatten()
    
    y_test_pred = (predictions > 0.5).astype(int)  # Binaire dans cet exemple
    success_rate = accuracy_score(y_test, y_test_pred)
    
    reward = success_rate * 100  # Vous pouvez définir votre propre fonction de récompense
    
    target = np.array([reward] * len(predictions))  # La récompense est la même pour chaque action
    model.fit(X_test, target, epochs=1, verbose=0)
    
    print('Taux de réussite à l\'itération', num_iterations, ':', success_rate)

print('Nombre d\'itérations nécessaires pour atteindre le seuil de réussite :', num_iterations)

def make_prediction(model, X):
    predictions = model.predict(X)
    print('Prédictions :', predictions)

print('Seuil de réussite atteint. Faisons une prédiction.')
make_prediction(model, X_test)
