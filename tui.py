import curses
import subprocess

def create_tui_window(stdscr):
    # Initialisation de Curses
    stdscr.clear()
    curses.curs_set(0)
    stdscr.refresh()

    # Activer les couleurs
    curses.start_color()
    curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)  # Couleur des bordures (rouge sur noir)
    curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)  # Couleur du texte (blanc sur noir)

    # Taille de l'écran
    sh, sw = stdscr.getmaxyx()

    # Fenêtre de gauche
    left_window = stdscr.subwin(sh, sw * 2 // 3, 0, 0)
    left_window.attron(curses.color_pair(1))  # Appliquer la couleur des bordures
    left_window.border()  # Affiche une bordure
    left_window.attroff(curses.color_pair(1))
    left_window.attron(curses.color_pair(2))  # Appliquer la couleur du texte
    left_window.addstr(2, 2, "Appuyez sur 'F' pour lancer le programme")
    left_window.attroff(curses.color_pair(2))

    # Fenêtre de droite
    right_window = stdscr.subwin(sh, sw // 3, 0, sw * 2 // 3)
    right_window.attron(curses.color_pair(1))  # Appliquer la couleur des bordures
    right_window.border()  # Affiche une bordure
    right_window.attroff(curses.color_pair(1))
    right_window.attron(curses.color_pair(2))  # Appliquer la couleur du texte
    right_window.addstr(2, 2, "Hyperparamètres en cours d'utilisation:")
    right_window.addstr(4, 2, "Taux de réussite en cours: 0.0")
    right_window.attroff(curses.color_pair(2))

    # Rafraîchir l'écran
    stdscr.refresh()

    # Attendre l'entrée de l'utilisateur
    while True:
        key = stdscr.getch()
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('F') or key == ord('f'):
            execute_program()

def execute_program():
    subprocess.run(["python3.10", "reinf_tuples.py"])

if __name__ == "__main__":
    curses.wrapper(create_tui_window)
