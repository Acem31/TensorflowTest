import subprocess

desired_result = [2, 13, 37, 38, 48, 5, 9]
num_attempts = 0

while True:
    # Exécuter le programme scotland.py
    result = subprocess.run(['python', 'scotland.py'], capture_output=True, text=True)

    # Vérifier si le résultat correspond à ce qui est recherché
    if result.returncode == 0:
        output_lines = result.stdout.strip().split('\n')
        last_line = output_lines[-1]
        predicted_numbers_rounded = [int(num) for num in last_line.split(';')]

        if predicted_numbers_rounded == desired_result:
            print(f"Résultat trouvé après {num_attempts} tentatives.")
            break

    # Si le résultat n'est pas celui recherché, incrémenter le nombre de tentatives
    num_attempts += 1

print("Fin du programme.")
