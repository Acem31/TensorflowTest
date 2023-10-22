import curses
import subprocess
import os
import sys
import pty
import signal
import csv
import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Variable pour suivre si le programme est en cours d'exécution
program_running = False

# Variable pour la hauteur du tableau
table_height = 12

def update_table(table, data, header):
    # Effacer le contenu actuel du tableau
    table.erase()

    # Assurez-vous que le nombre de lignes à mettre à jour ne dépasse pas la taille du tableau
    rows_to_update = min(len(data), table_height)

    # Réafficher les en-têtes du tableau
    for i, col_name in enumerate(header):
        table.addstr(i * 2, 1, col_name, curses.color_pair(2))

    # Mettez à jour le tableau avec les données
    for i in range(rows_to_update):
        table.addstr(i * 2 + 1, 1, data[i], curses.color_pair(2))

    table.refresh()

# Créez une classe de gestionnaire d'événements pour surveiller le fichier CSV
class CSVHandler(FileSystemEventHandler):
    def __init__(self, table):
        self.table = table

    def on_modified(self, event):
        if event.src_path == 'results.csv':
            # Charger les données actuelles du fichier CSV
            with open('results.csv', newline='') as csvfile:
                csv_reader = csv.reader(csvfile)
                data = list(csv_reader)
                if len(data) >= 2:
                    row = data[-1]
                    update_table(self.table, row)

# Fonction pour afficher la fenêtre TUI
def create_tui_window(stdscr):
    global program_running

    def start_program():
        global program_running
        program_running = True
        right_win.addstr(1, 2, "Lancement du programme...", curses.color_pair(2))
        right_win.refresh()

        max_y, max_x = right_win.getmaxyx()

        master, slave = pty.openpty()
        cmd = ["python3.10", "reinf_tuples.py"]
        p = subprocess.Popen(cmd, stdout=slave, stderr=slave, preexec_fn=lambda: curses.resizeterm(max_y, curses.COLS // 3))

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

        p.wait()
        program_running = False

    def stop_program():
        if program_running:
            p.send_signal(signal.SIGINT)

    curses.curs_set(0)
    stdscr.clear()
    stdscr.keypad(1)

    curses.start_color()
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)
    curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)

    left_win = stdscr.subwin(curses.LINES, curses.COLS // 3 - 2, 0, 0)
    left_win.bkgd(' ', curses.color_pair(1))
    left_win.box()

    left_win.addstr(1, 2, "Appuyez sur F pour lancer le programme", curses.color_pair(2))

    left_win.addstr(3, 2, "État du programme: En attente", curses.color_pair(2))
    left_win.addstr(5, 2, "Appuyez sur C pour arrêter le programme", curses.color_pair(2))

    right_win = stdscr.subwin(curses.LINES, 2 * (curses.COLS // 3 - 2), 0, curses.COLS // 3 + 2)
    right_win.bkgd(' ', curses.color_pair(2))
    right_win.box()

    table_height = 12
    table_width = curses.COLS // 3 - 6
    table_start_y = curses.LINES - table_height - 2
    table_start_x = 3
    table = left_win.subwin(table_height, table_width, table_start_y, table_start_x)

    # Créer un observateur watchdog pour surveiller les modifications du fichier CSV
    observer = Observer()
    observer.schedule(CSVHandler(table), path='.', recursive=False)
    observer.start()

    # Charger les données initiales du fichier CSV
    with open('results.csv', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        data = list(csv_reader)
        if len(data) >= 2:
            header = data[0]  # Récupérer le header depuis la première ligne
            row = data[-1]  # Récupérer les données depuis la dernière ligne
            update_table(table, row, header)  # Passer le header à la fonction


    stdscr.refresh()
    left_win.refresh()
    right_win.refresh()

    while True:
        key = stdscr.getch()
        if key == ord('F') or key == ord('f'):
            start_program()
            left_win.addstr(3, 2, "État du programme: En cours d'exécution", curses.color_pair(2))
            left_win.refresh()

        elif key == ord('C') or key == ord('c'):
            stop_program()
            left_win.addstr(3, 2, "État du programme: Arrêté", curses.color_pair(2))
            left_win.refresh()

        elif key in (ord('q'), ord('Q')):
            break

    stdscr.keypad(0)
    curses.endwin()

if __name__ == "__main__":
    curses.wrapper(create_tui_window)
