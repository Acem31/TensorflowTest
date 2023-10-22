import curses
import subprocess
import os
import sys
import pty
import signal  # Ajouter cette ligne pour gérer l'arrêt du programme
import csv

# Variable pour suivre si le programme est en cours d'exécution
program_running = False

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

    # Lecture du contenu du CSV et affichage dans la fenêtre de gauche
    try:
        with open('results.csv', 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                left_win.addstr(1, 2, ', '.join(row), curses.color_pair(2))
    except FileNotFoundError:
        # Si le fichier n'existe pas, affichez des valeurs par défaut
        left_win.addstr(1, 2, "Epochs,Batch Size,Learning Rate,Regularization,Accuracy,Precision", curses.color_pair(2))

    # Ajouter un bouton "Appuyez sur F pour lancer le programme" dans la fenêtre de gauche
    left_win.addstr(1, 2, "Appuyez sur F pour lancer le programme", curses.color_pair(2))

    # Ajouter un message pour indiquer l'état du programme
    left_win.addstr(3, 2, "État du programme: En attente", curses.color_pair(2))
    left_win.addstr(5, 2, "Appuyez sur C pour arrêter le programme", curses.color_pair(2))

    # Créer une fenêtre à droite (1/3 de largeur)
    right_win = stdscr.subwin(curses.LINES, 2 * (curses.COLS // 3 - 2), 0, curses.COLS // 3 + 2)
    right_win.bkgd(' ', curses.color_pair(2))  # Arrière-plan en blanc sur noir
    right_win.box()

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

# Exécuter la fenêtre TUI
if __name__ == "__main__":
    curses.wrapper(create_tui_window)
