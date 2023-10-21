import curses
import subprocess
import os
import sys
import pty

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

    # Obtenir les dimensions de la fenêtre de droite
    max_y, max_x = right_win.getmaxyx()

    # Calculer la taille de la sous-fenêtre (ajustez-la selon vos besoins)
    terminal_height = max_y - 5
    terminal_width = max_x - 4
    
    # Vérifier que la taille de la sous-fenêtre est valide
    if terminal_height <= 0 or terminal_width <= 0:
        # Gérer la taille invalide ici, par exemple, en affichant un message d'erreur.
        # Vous pouvez également ajuster la taille en conséquence.
    else:
        # Créer la sous-fenêtre
        terminal_win = right_win.subwin(terminal_height, terminal_width, 2, 2)
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
                    if len(lines) > max_y - 6:
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

        elif key in (ord('q'), ord('Q')):
            break

    # Désactiver le mode de clavier spécial avant de quitter
    stdscr.keypad(0)

    # Restaurer les paramètres du terminal
    curses.endwin()

# Exécuter la fenêtre TUI
if __name__ == "__main__":
    curses.wrapper(create_tui_window)
