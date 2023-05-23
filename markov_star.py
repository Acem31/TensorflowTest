import csv
import random

# Chargement des données
data = []
with open('euromillions.csv', 'r') as file:
    reader = csv.reader(file, delimiter=';')
    for row in reader:
        numbers = tuple(map(int, row[:5]))
        data.append(numbers)

# Récupération de la dernière ligne du CSV
last_row = data[-1]

# Construction de la matrice de transition
transition_matrix = {}
for i in range(len(data) - 1):
    current_tuple = data[i]
    next_tuple = data[i + 1]

    if current_tuple in transition_matrix:
        if next_tuple in transition_matrix[current_tuple]:
            transition_matrix[current_tuple][next_tuple] += 1
        else:
            transition_matrix[current_tuple][next_tuple] = 1
    else:
        transition_matrix[current_tuple] = {next_tuple: 1}

# Normaliser les probabilités de transition
for current_tuple in transition_matrix:
    total_transitions = sum(transition_matrix[current_tuple].values())
    for next_tuple in transition_matrix[current_tuple]:
        transition_matrix[current_tuple][next_tuple] /= total_transitions

# Prédiction du prochain tuple basé sur la dernière ligne
current_tuple = last_row
predicted_sequence = list(current_tuple)

for _ in range(4):  # Répéter quatre fois au lieu de cinq
    if current_tuple in transition_matrix:
        next_tuple = random.choices(
            list(transition_matrix[current_tuple].keys()),
            list(transition_matrix[current_tuple].values())
        )[0]
    else:
        next_tuple = random.choice(data)

    predicted_sequence.extend(next_tuple)
    current_tuple = next_tuple

# Afficher la séquence prédite de cinq chiffres
predicted_sequence = predicted_sequence[-5:]  # Garder uniquement les cinq derniers chiffres
print(predicted_sequence)
