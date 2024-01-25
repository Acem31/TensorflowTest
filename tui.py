import subprocess

# Résultat cible
target_result = [14, 23, 39, 48, 50, 3, 12]

# Initialiser le tableau des occurrences
occurrences = {i: 0 for i in range(8)}

# Nombre maximal d'itérations
max_iterations = 1000
iteration = 1

while occurrences[7] == 0 and iteration < max_iterations:
    # Lancer le script scotland.py
    result = subprocess.run(["python3.10", "auto-scot.py"], capture_output=True, text=True)

    # Extraire la chaîne représentant les numéros prédits
    output_lines = result.stdout.strip().split('\n')
    predicted_str = output_lines[-1].split(': ')[-1]

    # Convertir la chaîne en une liste
    predicted_numbers_rounded = list(map(float, predicted_str.strip('[[]]').split()))

    # Séparer les numéros prédits
    predicted_numbers_first_part = predicted_numbers_rounded[:5]
    predicted_numbers_second_part = predicted_numbers_rounded[5:]

    # Comparaison des 5 premiers numéros
    common_elements_first_part = set(target_result[:5]) & set(predicted_numbers_first_part)

    # Comparaison des 2 derniers numéros
    common_elements_second_part = set(target_result[5:]) & set(predicted_numbers_second_part)

    # Ajouter les occurrences aux totaux
    occurrences[len(common_elements_first_part) + len(common_elements_second_part)] += 1

    # Afficher le tableau des occurrences avec pourcentage
    print("\nOccurrences (itération", iteration, "):")
    print("-------------")
    for i in range(8):
        percentage = (occurrences[i] / iteration) * 100 if iteration > 0 else 0
        print(f"{i} corrects: {occurrences[i]} ({percentage:.2f}%)")

    # Incrémenter le nombre d'itérations
    iteration += 1

# Afficher le tableau des occurrences final avec pourcentage
print("\nOccurrences finales:")
print("-------------")
for i in range(8):
    percentage = (occurrences[i] / iteration) * 100 if iteration > 0 else 0
    print(f"{i} corrects: {occurrences[i]} ({percentage:.2f}%)")
