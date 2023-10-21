import curses
import subprocess
import os
import sys
import pty
import time

# Fonction pour afficher la fenêtre TUI
def create_tui_window(stdscr):
    # Démarrer la bibliothèque curses
    curses.curs_set(0)  # Masquer le curseur
    stdscr.clear()       # Effacer l'écran

    # Activer le mode de clavier spécial pour gérer les touches spéciales
    stdscr.keypad(1)

    # Initialiser les couleurs si le terminal le permet
    curses.start_color()
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)
    curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)

    # Créer une fenêtre à gauche (2/3 de largeur)
    left_win = stdscr.subwin(curses.LINES, curses.COLS // 3, 0, 0)
    left_win.bkgd(' ', curses.color_pair(1))  # Arrière-plan en blanc sur bleu
    left_win.box()

    # Ajouter un bouton "Appuyez sur F pour lancer le programme" dans la fenêtre de gauche
    left_win.addstr(1, 2, "Appuyez sur F pour lancer le programme", curses.color_pair(2))

    # Créer une fenêtre à droite (1/3 de largeur)
    right_win = stdscr.subwin(curses.LINES, 2 * (curses.COLS // 3), 0, curses.COLS // 3)
    right_win.bkgd(' ', curses.color_pair(2))  # Arrière-plan en blanc sur noir
    right_win.box()

    # Créer une sous-fenêtre dans la fenêtre de droite pour le terminal virtuel
    max_y, max_x = right_win.getmaxyx()
    terminal_win = right_win.subwin(max_y - 2, max_x - 2, 2, 2)
    terminal_win.scrollok(1)  # Activer le défilement automatique

    # Mettre à jour l'affichage
    stdscr.refresh()
    left_win.refresh()
    right_win.refresh()

    # Attente de l'appui sur la touche 'F' pour lancer le programme
    while True:
        key = stdscr.getch()
        if key == ord('F') or key == ord('f'):
            # Lancer le programme ici
            right_win.addstr(1, 2, "Lancement du programme...", curses.color_pair(2))
            right_win.refresh()

            # Exécution du programme dans la sous-fenêtre du terminal virtuel
            terminal_master, _ = pty.openpty()
            cmd = ["python3.10", "reinf_tuples.py"]
            p = subprocess.Popen(cmd, stdout=terminal_master, stderr=terminal_master, stdin=terminal_master, shell=False, universal_newlines=True)
            
            # Lecture et affichage de la sortie du programme dans la sous-fenêtre
            while p.poll() is None:
                output = os.read(terminal_master, 1024).decode("utf-8")
                terminal_win.addstr(output, curses.color_pair(2))
                terminal_win.refresh()
                time.sleep(0.1)  # Ajout d'un délai pour éviter de surcharger la boucle

        elif key in (ord('q'), ord('Q')):
            break

    # Désactiver le mode de clavier spécial avant de quitter
    stdscr.keypad(0)

    # Restaurer les paramètres du terminal
    curses.endwin()

# Exécuter la fenêtre TUI
if __name__ == "__main__":
    curses.wrapper(create_tui_window)
