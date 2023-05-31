import csv
import random

# Chargement des données existantes
data = []
with open('euromillions.csv', 'r') as file:
    reader = csv.reader(file, delimiter=';')
    for row in reader:
        numbers = tuple(map(int, row[:5]))
        data.append(numbers)

# Fonction pour mettre à jour la matrice de transition avec les nouvelles données
def update_transition_matrix(new_data, transition_matrix):
    for i in range(len(new_data) - 1):
        current_tuple = new_data[i]
        next_tuple = new_data[i + 1]

        if current_tuple in transition_matrix:
            if next_tuple in transition_matrix[current_tuple]:
                transition_matrix[current_tuple][next_tuple] += 1
            else:
                transition_matrix[current_tuple][next_tuple] = 1
        else:
            transition_matrix[current_tuple] = {next_tuple: 1}

    return transition_matrix

# Fonction pour normaliser les probabilités de transition
def normalize_transition_probabilities(transition_matrix):
    for current_tuple in transition_matrix:
        total_transitions = sum(transition_matrix[current_tuple].values())
        for next_tuple in transition_matrix[current_tuple]:
            transition_matrix[current_tuple][next_tuple] /= total_transitions

    return transition_matrix

# Fonction pour prédire le prochain tuple basé sur la matrice de transition
def predict_next_tuple(current_tuple, transition_matrix):
    if current_tuple in transition_matrix:
        next_tuple = random.choices(
            list(transition_matrix[current_tuple].keys()),
            list(transition_matrix[current_tuple].values())
        )[0]
    else:
        next_tuple = random.choice(data)

    return next_tuple

# Initialisation de la matrice de transition
transition_matrix = {}

# Boucle d'apprentissage
for _ in range(10):  # Répéter le processus d'apprentissage 10 fois (à ajuster selon les besoins)
    # Collecte de nouvelles données (supposons que vous ayez de nouvelles données dans une liste appelée "new_data")
    new_data = [27;29;32;33;47]  # Remplacez [...] par vos nouvelles données
    data.extend(new_data)

    # Mise à jour de la matrice de transition avec les nouvelles données
    transition_matrix = update_transition_matrix(new_data, transition_matrix)

    # Normalisation des probabilités de transition
    transition_matrix = normalize_transition_probabilities(transition_matrix)

# Récupération de la dernière ligne du CSV
last_row = data[-1]

# Prédiction du prochain tuple basé sur la dernière ligne et la matrice de transition améliorée
current_tuple = last_row
predicted_sequence = list(current_tuple)

for _ in range(4):  # Répéter quatre fois au lieu de cinq
    next_tuple = predict_next_tuple(current_tuple, transition_matrix)
    predicted_sequence.extend(next_tuple)
    current_tuple = next_tuple

# Afficher la séquence prédite de cinq chiffres
predicted_sequence = predicted_sequence[-5:]  # Garder uniquement les cinq derniers chiffres
print(predicted_sequence)
