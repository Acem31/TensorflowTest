import curses
import subprocess
import os
import pty
import signal
import csv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import threading

# Variables pour suivre l'état du programme
program_running = False
table_data = []

# Hauteur du tableau
table_height = 12
# En-têtes de tableau par défaut
table_headers = ["Epochs", "Batch Size", "Learning Rate", "Regularization", "Accuracy", "Precision"]

def update_table(table, data):
    # Effacer le contenu actuel du tableau
    table.clear()

    # Afficher les en-têtes du tableau
    for i, col_name in enumerate(table_headers):
        table.addstr(i * 2, 1, col_name, curses.color_pair(2))

    # Mettez à jour le tableau avec les données
    for i, row_data in enumerate(data):
        table.addstr(i * 2 + 1, 1, row_data, curses.color_pair(2))

    table.refresh()

# Créez une classe de gestionnaire d'événements pour surveiller le fichier CSV
class CSVHandler(FileSystemEventHandler):
    def __init__(self, table):
        self.table = table

    def on_modified(self, event):
        if event.src_path == 'results.csv':
            # Ajoutez une pause pour la synchronisation
            time.sleep(0.1)

            # Charger les données actuelles du fichier CSV
            with open('results.csv', newline='') as csvfile:
                csv_reader = csv.reader(csvfile)
                data = list(csv_reader)
                if data:
                    row = data[-1]  # Récupérer les données depuis la dernière ligne
                else:
                    row = ["0"] * len(table_headers)  # Remplacer les données vides par des zéros
                update_table(self.table, row)
                table_data.clear()
                table_data.extend(data)

# Fonction pour lancer le programme
def start_program():
    global program_running
    program_running = True
    right_win.addstr(1, 2, "Lancement du programme...", curses.color_pair(2))
    right_win.refresh()

    max_y, max_x = right_win.getmaxyx()

    master, slave = pty.openpty()
    cmd = ["python3.10", "reinf_tuples.py"]
    process = subprocess.Popen(
        cmd, stdout=slave, stderr=slave, preexec_fn=lambda: curses.resizeterm(max_y, curses.COLS // 3)
    )

    first_line = True
    lines = []
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
                lines.pop(0)
            right_win.clear()
            for i, line in enumerate(lines):
                right_win.addstr(2 + i, 2, line.strip(), curses.color_pair(2))
            right_win.refresh()
        except OSError:
            break

    process.wait()
    program_running = False

# Fonction pour arrêter le programme
def stop_program():
    global program_running
    if program_running:
        try:
            p.terminate()
        except ProcessLookupError:
            pass

# Fonction pour afficher la fenêtre TUI
def create_tui_window(stdscr):
    global program_running

    curses.curs_set(0)
    stdscr.clear()
    stdscr.keypad(1)

    curses.start_color()
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)
    curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)

    left_win = stdscr.subwin(curses.LINES, curses.COLS // 3 - 2, 0, 0)
    left_win.bkgd(' ', curses.color_pair(1))
    left_win.box()

    # Ajouter du texte centré avec fond bleu
    text_centered = "Appuyez sur F pour lancer le programme"
    left_win.addstr(1, (curses.COLS // 3 - len(text_centered)) // 2, text_centered, curses.color_pair(1))

    # Ajouter du texte centré avec fond bleu et blanc
    text_centered = "État du programme: En attente"
    left_win.addstr(2, (curses.COLS // 3 - len(text_centered)) // 2, text_centered, curses.color_pair(1))

    # Modifier la couleur du texte "Appuyez sur C pour arrêter le programme" et centrer
    text_centered = "Appuyez sur C pour arrêter le programme"
    left_win.addstr(3, (curses.COLS // 3 - len(text_centered)) // 2, text_centered, curses.color_pair(1))

    right_win = stdscr.subwin(curses.LINES, 2 * (curses.COLS // 3 - 2), 0, curses.COLS // 3 + 2)
    right_win.bkgd(' ', curses.color_pair(2))
    right_win.box()

    # Créer un tableau initial avec les en-têtes
    table_width = curses.COLS // 3 - 6
    table_start_y = curses.LINES - table_height - 2
    table_start_x = 3
    table = left_win.subwin(table_height, table_width, table_start_y, table_start_x)

    # Créer un observateur watchdog pour surveiller les modifications du fichier CSV
    observer = Observer()
    observer.schedule(CSVHandler(table), path='.', recursive=False)
    observer.start()

    stdscr.refresh()
    left_win.refresh()
    right_win.refresh()

    while True:
        key = stdscr.getch()
        if key == ord('F') or key == ord('f'):
            if not program_running:
                # Démarrer le programme dans un thread
                program_thread = threading.Thread(target=start_program)
                program_thread.start()
                text_centered = "État du programme: En cours d'exécution"
                left_win.addstr(2, (curses.COLS // 3 - len(text_centered)) // 2, text_centered, curses.color_pair(1))
                left_win.refresh()
        elif key == ord('C') or key == ord('c'):
            stop_program()
            text_centered = "État du programme: Arrêté"
            left_win.addstr(2, (curses.COLS // 3 - len(text_centered)) // 2, text_centered, curses.color_pair(1))
            left_win.refresh()
        elif key in (ord('q'), ord('Q')):
            break

    stdscr.keypad(0)
    curses.endwin()

if __name__ == "__main__":
    curses.wrapper(create_tui_window)
