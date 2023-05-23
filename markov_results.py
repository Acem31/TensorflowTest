import subprocess
from collections import Counter

def execute_markov_number():
    process = subprocess.Popen(['python', 'markov_number.py'], stdout=subprocess.PIPE)
    output, error = process.communicate()
    output = output.decode().strip()
    # Modifier la façon dont les tuples sont extraits de la sortie
    numbers = tuple(map(int, output[1:-1].split(',')))
    return numbers


def execute_markov_star():
    process = subprocess.Popen(['python', 'markov_star.py'], stdout=subprocess.PIPE)
    output, error = process.communicate()
    output = output.decode().strip()
    # Modifier la façon dont les tuples sont extraits de la sortie
    numbers = tuple(map(int, output.split(":")[1].strip()[1:-1].split(',')))
    return numbers


def get_most_frequent_numbers(tuples):
    columns = zip(*tuples)
    most_frequent_numbers = [Counter(column).most_common(1)[0][0] for column in columns]
    return tuple(most_frequent_numbers)

# Exécuter markov_number.py 500 fois
markov_number_tuples = []
for _ in range(500):
    numbers = execute_markov_number()
    markov_number_tuples.append(numbers)

# Sélectionner les chiffres les plus fréquents de chaque colonne
most_frequent_markov_number = get_most_frequent_numbers(markov_number_tuples)

# Exécuter markov_star.py 500 fois
markov_star_tuples = []
for _ in range(500):
    numbers = execute_markov_star()
    markov_star_tuples.append(numbers)

# Sélectionner les chiffres les plus fréquents de chaque colonne
most_frequent_markov_star = get_most_frequent_numbers(markov_star_tuples)

print("Chiffres les plus fréquents (markov_number):", most_frequent_markov_number)
print("Chiffres les plus fréquents (markov_star):", most_frequent_markov_star)
