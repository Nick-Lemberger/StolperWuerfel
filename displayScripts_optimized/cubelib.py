#!/usr/bin/env python3
"""Gemeinsame, vektorisierte Bausteine für die Würfel-Animationen.

Layouts:
  Welt / Zähler:  [face, x, y]          (6, 64, 64)
  Frame:          [face, x, y, rgb]     (6, 64, 64, 3) uint8  -> cube.Cube.sendFrame()
  Vektorfeld:     [face, x, y, (vx,vy)] (6, 64, 64, 2)

Die Kantenübergänge entsprechen exakt getCell() bzw. wrapCoords() der
Originalskripte.
"""
import time
import numpy as np

N = 64          # Kantenlänge eines Panels
FACES = 6

rng = np.random.default_rng()


# ====================================================================== Topologie
def resolve(face, x, y):
    """Wie getCell() im Original, liefert (face, x, y) statt des Zellwerts."""
    n = N
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


def _build_pad_index():
    """Flache Indizes einer um 1 Zelle gepolsterten Welt (6, 66, 66)."""
    idx = np.empty((FACES, N + 2, N + 2), dtype=np.intp)
    for f in range(FACES):
        for px in range(N + 2):
            for py in range(N + 2):
                ff, xx, yy = resolve(f, px - 1, py - 1)
                idx[f, px, py] = (ff * N + xx) * N + yy
    return idx


PAD_IDX = _build_pad_index()


# ====================================================================== Game of Life
def make_rule(ruleS, ruleB):
    """Lookup-Tabellen (Überleben, Geburt) für Nachbaranzahl 0..8."""
    S = np.zeros(9, dtype=np.uint8)
    B = np.zeros(9, dtype=np.uint8)
    S[[k for k in ruleS if 0 <= k <= 8]] = 1
    B[[k for k in ruleB if 0 <= k <= 8]] = 1
    return S, B


def life_step(world, rule):
    """Eine Generation auf der Würfeloberfläche. Gibt ein neues Array zurück."""
    S, B = rule
    p = world.ravel()[PAD_IDX]
    nb = (p[:, :-2, :-2] + p[:, :-2, 1:-1] + p[:, :-2, 2:] +
          p[:, 1:-1, :-2] +                  p[:, 1:-1, 2:] +
          p[:, 2:, :-2] + p[:, 2:, 1:-1] + p[:, 2:, 2:])
    return np.where(world == 1, S[nb], B[nb])


def empty_world():
    return np.zeros((FACES, N, N), dtype=np.uint8)


def seed(world, density):
    """Setzt jede Zelle mit Wahrscheinlichkeit `density` lebendig (in-place, bestehende bleiben)."""
    world |= (rng.random(world.shape) < density).astype(np.uint8)
    return world


# ====================================================================== Farben
class ColorFade:
    """Farbakkumulation wie sendToCube() im Original.

    Pro Frame:  buf += count * cIncr - cDecr,  danach auf [0, cLim] begrenzt.
    """

    def __init__(self, cIncr, cDecr, cLim):
        self.buf = np.zeros((FACES, N, N, 3), dtype=np.int32)
        self.frame = np.zeros((FACES, N, N, 3), dtype=np.uint8)
        self.set(cIncr, cDecr, cLim)

    def set(self, cIncr, cDecr, cLim):
        self.incr = np.asarray(cIncr, dtype=np.int32)
        self.decr = np.asarray(cDecr, dtype=np.int32)
        # Obergrenze 255 entspricht der uint8-Sättigung im Original
        self.lim = np.minimum(np.asarray(cLim, dtype=np.int32), 255)

    def update(self, counts):
        """counts: (6, 64, 64) – lebende Zellen bzw. Partikel pro Pixel. Gibt den uint8-Frame zurück."""
        self.buf += counts[..., None].astype(np.int32) * self.incr - self.decr
        np.clip(self.buf, 0, self.lim, out=self.buf)
        self.frame[:] = self.buf
        return self.frame


def _randint(a, b):
    """Wie random.randint: a <= n <= b."""
    return int(rng.integers(a, b + 1))


def random_colors_simple(print_out=True):
    """randomColors() aus cubeGameOfLife.py."""
    cIncr = [_randint(2, 50) for _ in range(3)]
    cDecr = [_randint(1, 5) for _ in range(3)]
    cLim = [_randint(100, 255) for _ in range(3)]
    if print_out:
        print("cIncr cDecr cLim:", cIncr, cDecr, cLim)
    return cIncr, cDecr, cLim


def random_colors_lightpower(max_lightPower=500, print_out=True):
    """randomColors() aus blobbs/bloomBoom/rndGoL (Summe der cLim begrenzt)."""
    cIncr = [_randint(2, 50) for _ in range(3)]
    cDecr = [_randint(1, 10) for _ in range(3)]
    power = max_lightPower
    r = _randint(25, 200)
    power -= r
    g = _randint(25, 200) if power > 200 else _randint(25, power)
    power -= g
    b = _randint(25, 200) if power > 200 else _randint(25, power)
    cLim = [r, g, b]
    if print_out:
        print("cIncr cDecr cLim:", cIncr, cDecr, cLim)
    return cIncr, cDecr, cLim


# ====================================================================== Partikel
def _sel(v, m):
    return v[m] if isinstance(v, np.ndarray) else v


def wrap_particles(face, x, y, vx, vy, max_iter=8):
    """Vektorisierte Version von wrapCoords(): Position über die Würfelkanten
    führen und die Geschwindigkeit entsprechend drehen. Arbeitet in-place."""
    n, m1 = N, N - 1
    for _ in range(max_iter):
        out = (x < 0) | (x >= n) | (y < 0) | (y >= n)
        if not out.any():
            return face, x, y, vx, vy
        idx = np.nonzero(out)[0]
        f, X, Y, VX, VY = face[idx], x[idx], y[idx], vx[idx], vy[idx]
        nf, nx, ny, nvx, nvy = f.copy(), X.copy(), Y.copy(), VX.copy(), VY.copy()
        todo = np.ones(idx.size, dtype=bool)

        def rule(cond, F, XX, YY, VVX, VVY):
            m = todo & cond
            if m.any():
                nf[m] = _sel(F, m)
                nx[m] = _sel(XX, m)
                ny[m] = _sel(YY, m)
                nvx[m] = _sel(VVX, m)
                nvy[m] = _sel(VVY, m)
                todo[m] = False

        # Reihenfolge exakt wie in wrapCoords()
        ring = f < 4
        rule(ring & (X < 0), (f + 1) % 4, m1, Y, VX, VY)
        rule(ring & (X >= n), (f - 1) % 4, 0, Y, VX, VY)
        c = ring & (Y < 0)
        rule(c & (f == 0), 4, m1, m1 - X, VY, -VX)
        rule(c & (f == 1), 4, X, m1, VX, VY)
        rule(c & (f == 2), 4, 0, X, -VY, VX)
        rule(c & (f == 3), 4, m1 - X, 0, -VX, -VY)
        c = ring & (Y >= n)
        rule(c & (f == 0), 5, m1, X, -VY, VX)
        rule(c & (f == 1), 5, X, 0, VX, VY)
        rule(c & (f == 2), 5, 0, m1 - X, VY, -VX)
        rule(c & (f == 3), 5, m1 - X, m1, -VX, -VY)
        c = f == 4
        rule(c & (X >= n), 0, m1 - Y, 0, -VY, VX)
        rule(c & (X < 0), 2, Y, 0, VY, -VX)
        rule(c & (Y >= n), 1, X, 0, VX, VY)
        rule(c & (Y < 0), 3, m1 - X, 0, -VX, -VY)
        c = f == 5
        rule(c & (X >= n), 0, Y, m1, VY, -VX)
        rule(c & (X < 0), 2, m1 - Y, m1, -VY, VX)
        rule(c & (Y >= n), 3, m1 - X, m1, -VX, -VY)
        rule(c & (Y < 0), 1, X, m1, VX, VY)

        face[idx], x[idx], y[idx], vx[idx], vy[idx] = nf, nx, ny, nvx, nvy
    raise RuntimeError("wrap_particles: Koordinaten nach max_iter nicht aufgelöst")


def pixel_index(face, x, y):
    """Ganzzahlige Pixelindizes (wie int(x), int(y) im Original; x, y >= 0)."""
    return face, x.astype(np.intp), y.astype(np.intp)


def particle_counts(face, ix, iy):
    """Anzahl Partikel pro Pixel als (6, 64, 64)-Array."""
    flat = (face * N + ix) * N + iy
    return np.bincount(flat, minlength=FACES * N * N).reshape(FACES, N, N)


def wall_axes():
    """Hilfsgitter: I[x] und J[y] als broadcastfähige Arrays (für Feldformeln)."""
    a = np.arange(N, dtype=float)
    return a[None, :, None], a[None, None, :]


# ====================================================================== Timing
class FrameLimiter:
    """Begrenzt die Bildrate auf target_fps (None = unbegrenzt) und meldet die FPS."""

    def __init__(self, target_fps=None, report_every=5.0):
        self.frame_time = 1.0 / target_fps if target_fps else 0.0
        self.report_every = report_every
        self._t_last = time.perf_counter()
        self._t_stat = self._t_last
        self._frames = 0

    def tick(self):
        now = time.perf_counter()
        if self.frame_time:
            rest = self.frame_time - (now - self._t_last)
            if rest > 0:
                time.sleep(rest)
                now = time.perf_counter()
        self._t_last = now
        self._frames += 1
        if self.report_every and now - self._t_stat >= self.report_every:
            print(f"{self._frames / (now - self._t_stat):.1f} FPS")
            self._t_stat, self._frames = now, 0
