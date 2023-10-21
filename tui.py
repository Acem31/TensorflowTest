import curses
import subprocess
import tempfile
import os

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

            # Créer un fichier temporaire pour capturer la sortie du programme
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_filename = temp_file.name

                # Exécutez votre script ici en redirigeant la sortie vers le fichier temporaire
                process = subprocess.Popen(["python3.10", "reinf_tuples.py"], stdout=temp_file, stderr=temp_file)

                # Attendez que le programme se termine
                process.wait()

                # Fermez le fichier temporaire
                temp_file.close()

                # Lire la sortie du fichier temporaire et l'afficher ligne par ligne
                with open(temp_filename, "r") as output_file:
                    for i, line in enumerate(output_file):
                        right_win.addstr(3 + i, 2, line.strip(), curses.color_pair(2))
                        right_win.refresh()

                # Supprimez le fichier temporaire
                os.remove(temp_filename)

        elif key in (ord('q'), ord('Q')):
            break

    # Désactiver le mode de clavier spécial avant de quitter
    stdscr.keypad(0)

    # Restaurer les paramètres du terminal
    curses.endwin()

# Exécuter la fenêtre TUI
if __name__ == "__main__":
    curses.wrapper(create_tui_window)
