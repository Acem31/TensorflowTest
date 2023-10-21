import curses
import subprocess

# Fonction pour afficher la fenêtre TUI
def create_tui_window(stdscr):
    # Démarrer la bibliothèque curses
    curses.curs_set(0)  # Masquer le curseur
    stdscr.clear()       # Effacer l'écran

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
            
            # Exécutez votre script ici (remplacez la commande par votre script)
            try:
                output = subprocess.check_output(["python", "reinf_tuples.py"])
                right_win.addstr(3, 2, "Programme terminé.", curses.color_pair(2))
            except subprocess.CalledProcessError as e:
                right_win.addstr(3, 2, f"Erreur : {e}", curses.color_pair(2))
            right_win.refresh()
        elif key in (ord('q'), ord('Q')):
            break

    # Restaurer les paramètres du terminal
    curses.endwin()

# Exécuter la fenêtre TUI
if __name__ == "__main__":
    curses.wrapper(create_tui_window)
