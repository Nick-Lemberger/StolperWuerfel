#!/usr/bin/env python3
"""Game of Life auf der Oberfläche des LED-Würfels – vektorisierte Version.

Weltlayout intern: world[face, x, y] (uint8, 0/1).
Die Kantenübergänge zwischen den Flächen sind exakt dieselben wie im
Original (getCell); sie werden einmalig in eine Index-Tabelle übersetzt.
"""
import time
import numpy as np
import cube

# ---------------------------------------------------------------- Regeln
# Wild growth
ruleS = [3, 2, 7]   # Überleben
ruleB = [3]         # Geburt

world_size = 64
start_density = 0.20
cutoff_pop = 5000

# Optional: Bildrate begrenzen (None = so schnell wie möglich)
target_fps = None

# ---------------------------------------------------------------- Farben
# Preset "Slow Winter"
# cIncr = [4, 6, 10]; cDecr = [1, 1, 1]; cLim = [255, 255, 255]

# Preset "Sakura Kiss"
cIncr = np.array([27, 26, 48], dtype=np.int16)
cDecr = np.array([2, 5, 3], dtype=np.int16)
cLim = np.array([181, 150, 152], dtype=np.int16)

UDP_IP = '192.168.178.50'
UDP_PORT = 2000

rng = np.random.default_rng()


# ---------------------------------------------------------------- Topologie
def resolve(face, x, y):
    """Wie getCell im Original, liefert aber (face, x, y) statt des Zellwerts."""
    n = world_size
    if 0 <= x < n and 0 <= y < n:
        return face, x, y
    if face in (0, 1, 2, 3):
        if x < 0:
            return resolve((face + 1) % 4, n - 1, y)
        if x >= n:
            return resolve((face - 1) % 4, 0, y)
        if y < 0:
            if face == 0:
                return resolve(4, n - 1, n - 1 - x)
            elif face == 1:
                return resolve(4, x, n - 1)
            elif face == 2:
                return resolve(4, 0, x)
            else:
                return resolve(4, n - 1 - x, 0)
        if y >= n:
            if face == 0:
                return resolve(5, n - 1, x)
            elif face == 1:
                return resolve(5, x, 0)
            elif face == 2:
                return resolve(5, 0, n - 1 - x)
            else:
                return resolve(5, n - 1 - x, n - 1)
    if face == 4:
        if x >= n:
            return resolve(0, n - 1 - y, 0)
        if x < 0:
            return resolve(2, y, 0)
        if y >= n:
            return resolve(1, x, 0)
        if y < 0:
            return resolve(3, n - 1 - x, 0)
    if face == 5:
        if x >= n:
            return resolve(0, y, n - 1)
        if x < 0:
            return resolve(2, n - 1 - y, n - 1)
        if y >= n:
            return resolve(3, n - 1 - x, n - 1)
        if y < 0:
            return resolve(1, x, n - 1)
    raise ValueError(f"Unaufgelöste Koordinate {(face, x, y)}")


def build_pad_index():
    """Flache Indizes für eine um 1 Zelle gepolsterte Welt (6, 66, 66)."""
    n = world_size
    idx = np.empty((6, n + 2, n + 2), dtype=np.intp)
    for f in range(6):
        for px in range(n + 2):
            for py in range(n + 2):
                ff, xx, yy = resolve(f, px - 1, py - 1)
                idx[f, px, py] = (ff * n + xx) * n + yy
    return idx


PAD_IDX = build_pad_index()

# Lookup-Tabellen: Nachbaranzahl (0..8) -> neuer Zustand
SURVIVE = np.zeros(9, dtype=np.uint8)
SURVIVE[[k for k in ruleS if 0 <= k <= 8]] = 1
BIRTH = np.zeros(9, dtype=np.uint8)
BIRTH[[k for k in ruleB if 0 <= k <= 8]] = 1


def step(world):
    """Eine Generation, komplett vektorisiert."""
    p = world.ravel()[PAD_IDX]                       # (6, 66, 66) mit Nachbarflächen am Rand
    nb = (p[:, :-2, :-2] + p[:, :-2, 1:-1] + p[:, :-2, 2:] +
          p[:, 1:-1, :-2] +                  p[:, 1:-1, 2:] +
          p[:, 2:, :-2] + p[:, 2:, 1:-1] + p[:, 2:, 2:])
    return np.where(world == 1, SURVIVE[nb], BIRTH[nb])


def seed_world():
    return (rng.random((6, world_size, world_size)) < start_density).astype(np.uint8)


def randomColors():
    global cIncr, cDecr, cLim
    cIncr = rng.integers(2, 51, 3).astype(np.int16)
    cDecr = rng.integers(1, 6, 3).astype(np.int16)
    cLim = rng.integers(100, 256, 3).astype(np.int16)


# ---------------------------------------------------------------- Darstellung
# Farbpuffer [face, x, y, rgb] als int16, damit Addition/Subtraktion nicht überläuft
color_buf = np.zeros((6, world_size, world_size, 3), dtype=np.int16)
frame = np.zeros((6, world_size, world_size, 3), dtype=np.uint8)


def sendToCube(c, world):
    # lebende Zellen: +cIncr-cDecr, tote Zellen: -cDecr (wie im Original)
    color_buf[:] += world[..., None].astype(np.int16) * cIncr - cDecr
    np.clip(color_buf, 0, cLim, out=color_buf)       # cLim <= 255 deckt die 8-Bit-Sättigung mit ab
    frame[:] = color_buf
    c.sendFrame(frame)


# ---------------------------------------------------------------- Hauptschleife
def main():
    c = cube.Cube(UDP_IP, UDP_PORT)
    world = seed_world()
    # randomColors()
    print(cIncr, cDecr, cLim)

    frame_time = 1.0 / target_fps if target_fps else 0.0
    t_stat, frames = time.perf_counter(), 0
    while True:
        t0 = time.perf_counter()
        if np.count_nonzero(world) > cutoff_pop:
            world = seed_world()
            # randomColors()

        sendToCube(c, world)
        world = step(world)

        frames += 1
        now = time.perf_counter()
        if now - t_stat >= 5.0:
            print(f"{frames / (now - t_stat):.1f} FPS")
            t_stat, frames = now, 0
        if frame_time:
            rest = frame_time - (time.perf_counter() - t0)
            if rest > 0:
                time.sleep(rest)


if __name__ == "__main__":
    main()
