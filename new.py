import subprocess
import numpy as np

# Fonction pour compter les occurrences
def count_occurrences(target, predictions):
    count = 0
    for prediction in predictions:
        if np.array_equal(prediction, target):
            count += 1
    return count

# Fonction pour afficher le tableau résumant les occurrences
def print_occurrences_table(occurrences):
    print("Occurrences:")
    print("-------------")
    print("0 corrects: ", occurrences[0])
    print("1 correct: ", occurrences[1])
    print("2 corrects: ", occurrences[2])
    print("3 corrects: ", occurrences[3])
    print("4 corrects: ", occurrences[4])
    print("5 corrects: ", occurrences[5])
    print("6 corrects: ", occurrences[6])
    print("7 corrects: ", occurrences[7])

# Fonction principale
def main():
    # Initialiser le tableau des occurrences
    occurrences = {i: 0 for i in range(8)}

    # Nombre maximal d'itérations
    max_iterations = 1000
    iteration = 0

    while iteration < max_iterations:
        # Lancer le script scotland.py
        result = subprocess.run(["python3.10", "scotland.py"], capture_output=True, text=True)

        # Extraire les numéros prédits
        predicted_numbers_rounded = eval(result.stdout)

        # Résultat cible
        target_result = np.array([4, 7, 18, 39, 50, 3, 8])

        # Compter les occurrences
        correct_numbers = count_occurrences(target_result, predicted_numbers_rounded)
        occurrences[correct_numbers] += 1

        # Afficher le tableau des occurrences
        print_occurrences_table(occurrences)

        # Vérifier si les numéros prédits sont corrects
        if correct_numbers == 7:
            print("Le résultat cible a été atteint! Arrêt du programme.")
            break

        iteration += 1

# Exécuter le programme principal
if __name__ == "__main__":
    main()
