import curses
import subprocess
import os
import sys
import pty
import signal  # Ajouter cette ligne pour gérer l'arrêt du programme
import csv
import threading
import time

# Variable pour suivre si le programme est en cours d'exécution
program_running = False

# Créez une fonction pour mettre à jour le tableau avec des données actuelles
def update_table(table, data):
    # Assurez-vous que le nombre de lignes à mettre à jour ne dépasse pas la taille du tableau
    rows_to_update = min(len(data), table_height)

    # Mettez à jour le tableau avec les données
    for i in range(rows_to_update):
        table.addstr(i * 2 + 1, 1, data[i], curses.color_pair(2))

    table.refresh()

# Créez une fonction pour mettre à jour le tableau en arrière-plan
def update_table_periodically():
    while True:
        # Charger les données actuelles du fichier CSV
        with open('results.csv', newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            data = list(csv_reader)
            if len(data) >= 2:
                row = data[-1]
                update_table(table, row)
        
        # Attendez un certain intervalle avant la prochaine mise à jour
        time.sleep(5)  # par exemple, mettez à jour toutes les 5 secondes


# Fonction pour afficher la fenêtre TUI
def create_tui_window(stdscr):
    global program_running  # Utilisez la variable globale

    def start_program():
        global program_running
        program_running = True
        right_win.addstr(1, 2, "Lancement du programme...", curses.color_pair(2))
        right_win.refresh()

        # Obtenir la hauteur de la fenêtre de droite
        max_y, max_x = right_win.getmaxyx()

        # Utiliser un terminal virtuel pour exécuter le script
        master, slave = pty.openpty()
        cmd = ["python3.10", "reinf_tuples.py"]
        p = subprocess.Popen(cmd, stdout=slave, stderr=slave, preexec_fn=lambda: curses.resizeterm(max_y, curses.COLS // 3))

        # Lire la sortie du terminal virtuel et afficher dans la fenêtre de droite
        first_line = True  # Pour suivre la première ligne
        lines = []  # Liste pour stocker les lignes à afficher
        while True:
            try:
                output = os.read(master, 1024).decode("utf-8")
                if not output:
                    break
                if first_line:
                    right_win.addstr(1, 2, " " * (curses.COLS // 3 - 4), curses.color_pair(2))
                    first_line = False 
                lines.append(output)
                if len(lines) > max_y - 8:
                    # Si le nombre de lignes dépasse la hauteur de la fenêtre, faire défiler
                    lines.pop(0)
                right_win.clear()
                for i, line in enumerate(lines):
                    right_win.addstr(2 + i, 2, line.strip(), curses.color_pair(2))
                right_win.refresh()
            except OSError:
                break

        # Attendre la fin du programme
        p.wait()
        program_running = False  # Mettez à jour l'état du programme

    def stop_program():
        if program_running:
            # Envoyer un signal pour demander l'arrêt du programme
            p.send_signal(signal.SIGINT)

    # Démarrer la bibliothèque curses
    curses.curs_set(0)  # Masquer le curseur
    stdscr.clear()       # Effacer l'écran

    # Activer le mode de clavier spécial pour gérer les touches spéciales
    stdscr.keypad(1)

    # Initialiser les couleurs si le terminal le permet
    curses.start_color()
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)
    curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)
    
    # Créez la fenêtre left_win avec des marges à gauche et à droite
    left_win = stdscr.subwin(curses.LINES, curses.COLS // 3 - 2, 0, 0)
    left_win.bkgd(' ', curses.color_pair(1))
    left_win.box()

    # Ajouter un bouton "Appuyez sur F pour lancer le programme" dans la fenêtre de gauche
    left_win.addstr(1, 2, "Appuyez sur F pour lancer le programme", curses.color_pair(2))

    # Ajouter un message pour indiquer l'état du programme
    left_win.addstr(3, 2, "État du programme: En attente", curses.color_pair(2))
    left_win.addstr(5, 2, "Appuyez sur C pour arrêter le programme", curses.color_pair(2))

    # Créer une fenêtre à droite (1/3 de largeur)
    right_win = stdscr.subwin(curses.LINES, 2 * (curses.COLS // 3 - 2), 0, curses.COLS // 3 + 2)
    right_win.bkgd(' ', curses.color_pair(2))  # Arrière-plan en blanc sur noir
    right_win.box()

    # Créer le tableau de 12 lignes au bas de la fenêtre de gauche
    table_height = 12
    table_width = curses.COLS // 3 - 6  # Largeur du tableau
    table_start_y = curses.LINES - table_height - 2  # Position en Y pour le tableau
    table_start_x = 3  # Marge à gauche pour le tableau
    table = left_win.subwin(table_height, table_width, table_start_y, table_start_x)

    # Charger les données du fichier CSV
    with open('results.csv', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        data = list(csv_reader)

        # Assurez-vous que le fichier CSV a au moins 2 lignes (header + données)
        if len(data) >= 2:
        # Récupérer le header depuis la première ligne
            header = data[0]
        
        # Récupérer les données depuis la dernière ligne
            row = data[-1]
    
        # Remplir le tableau avec les données du header et de la dernière ligne
        for i in range(len(header)):
            table.addstr(i * 2, 1, header[i], curses.color_pair(2))
            table.addstr(i * 2 + 1, 1, row[i], curses.color_pair(2))


    # Mettre à jour l'affichage
    stdscr.refresh()
    left_win.refresh()
    right_win.refresh()

    # Attente de l'appui sur la touche 'F' pour lancer le programme
    while True:
        key = stdscr.getch()
        if key == ord('F') or key == ord('f'):
            start_program()  # Lancer le programme
            left_win.addstr(3, 2, "État du programme: En cours d'exécution", curses.color_pair(2))
            left_win.refresh()

        elif key == ord('C') or key == ord('c'):
            stop_program()  # Arrêter le programme
            left_win.addstr(3, 2, "État du programme: Arrêté", curses.color_pair(2))
            left_win.refresh()

        elif key in (ord('q'), ord('Q')):
            break

    # Désactiver le mode de clavier spécial avant de quitter
    stdscr.keypad(0)

    # Restaurer les paramètres du terminal
    curses.endwin()

if __name__ == "__main__":
    # Créez un thread pour mettre à jour le tableau en arrière-plan
    update_thread = threading.Thread(target=update_table_periodically)
    update_thread.daemon = True  # Le thread s'exécutera en arrière-plan

    # Démarrer le thread
    update_thread.start()


# Exécuter la fenêtre TUI
if __name__ == "__main__":
    curses.wrapper(create_tui_window)
