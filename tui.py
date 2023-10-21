import curses
import subprocess

def create_tui_window(stdscr):
    # Initialisation de Curses
    stdscr.clear()
    curses.curs_set(0)
    stdscr.refresh()

    # Taille de l'écran
    sh, sw = stdscr.getmaxyx()

    # Fenêtre de gauche
    left_window = stdscr.subwin(sh, sw * 2 // 3, 0, 0)
    left_window.box()
    left_window.addstr(2, 2, "Appuyez sur 'F' pour lancer le programme")

    # Fenêtre de droite
    right_window = stdscr.subwin(sh, sw // 3, 0, sw * 2 // 3)
    right_window.box()
    right_window.addstr(2, 2, "Hyperparamètres en cours d'utilisation:")
    right_window.addstr(4, 2, "Taux de réussite en cours: 0.0")

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
