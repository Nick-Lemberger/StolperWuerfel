#!/usr/bin/env python3
"""Zeigt auf jedem Panel groß seine Flächennummer (0..5) – zum Zuordnen der Panels.

Jede Fläche bekommt eine eigene Farbe; ein kleines Quadrat markiert die Ecke mit
Pixel (x=0, y=0). Erscheinen die Ziffern gespiegelt, ist die Pixelreihenfolge der
Panels gespiegelt zur Annahme der Skripte – dann bitte Bescheid geben.
Beenden mit Strg+C.
"""
import time
import numpy as np
import cube

UDP_IP = '192.168.178.50'
UDP_PORT = 2000

COLORS = [(255, 0, 0), (0, 255, 0), (0, 40, 255), (255, 190, 0), (0, 230, 255), (255, 0, 200)]

# 5x7-Ziffern
GLYPHS = {
    0: ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    1: ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    2: ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    3: ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    4: ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    5: ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
}
SCALE = 7


def make_frame():
    frame = np.zeros((6, 64, 64, 3), dtype=np.uint8)
    gw, gh = 5 * SCALE, 7 * SCALE
    x0, y0 = (64 - gw) // 2, (64 - gh) // 2
    for f in range(6):
        col = np.array(COLORS[f], dtype=np.uint8)
        frame[f] = (np.array(COLORS[f]) * 0.06).astype(np.uint8)       # schwacher Hintergrund
        for r, row in enumerate(GLYPHS[f]):
            for c, bit in enumerate(row):
                if bit == "1":
                    # Frame-Index [Fläche, x, y]: x = Spalte, y = Zeile
                    frame[f, x0 + c * SCALE: x0 + (c + 1) * SCALE, y0 + r * SCALE: y0 + (r + 1) * SCALE] = col
        frame[f, 1:5, 1:5] = (200, 200, 200)                           # Ecke x=0, y=0
    return frame


def main():
    c = cube.Cube(UDP_IP, UDP_PORT)
    frame = make_frame()
    print("Flächennummern werden angezeigt – Strg+C beendet.")
    while True:
        c.sendFrame(frame)
        time.sleep(0.2)


if __name__ == "__main__":
    main()
