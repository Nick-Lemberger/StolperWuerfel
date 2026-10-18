#!/usr/bin/env python3
"""Lavalampe auf dem LED-Würfel.

Wachsblobs schweben im Würfelvolumen: Am Boden werden sie aufgeheizt, steigen
auf, kühlen nach oben hin ab und sinken wieder in die Wachsschicht am Boden.
Die Blobs sind 3D-Metaballs; ihr Feld wird direkt auf den sechs Panelflächen
ausgewertet, dadurch fließen Blobs nahtlos über Kanten und Ecken und
verschmelzen/trennen sich wie echtes Wachs. Das Farbschema wechselt langsam
und zufällig.
"""
import numpy as np
import cube
import cubelib as cl

# ====================================================================== Ausrichtung
# Flächen, die oben bzw. unten sind – müssen gegenüberliegen.
# Gegenüberliegende Paare: (0, 2), (1, 3), (4, 5)
top_face = 4
bottom_face = 5

# ====================================================================== Wachs / Physik
n_blobs = 9                     # Anzahl Wachsblobs
blob_radius = (6.0, 11.0)       # Radiusbereich in Pixeln
radius_breathe = 0.12           # langsame relative Größenschwankung der Blobs
buoyancy = 0.014                # Auftrieb pro Temperatureinheit (px/Frame²)
drag = 0.965                    # erhaltene Geschwindigkeit pro Frame (Zähigkeit)
max_speed = 0.35                # px/Frame
heat_rate = 0.012               # Aufheizrate in der Heizzone am Boden
heat_depth = 8.0                # Reichweite der Heizzone über dem Boden (px)
cool_rate = 0.010               # Abkühlrate in der Kühlzone unter der Decke
cool_depth = 10.0               # Reichweite der Kühlzone unter der Decke (px)
cool_base = 0.01                # schwache Abkühlung überall (Anteil von cool_rate)
wander = 0.004                  # seitliche Zufallsbewegung
wall_depth = 4.0                # bevorzugter Abstand der Blobs zur Seitenwand (px)
wall_pull = 0.002               # Stärke, mit der Blobs an die "Glaswand" driften
blob_repel = 0.004              # schwache Abstoßung zwischen Blobs

pool_level = 6.0                # Höhe der Wachsschicht am Boden (px)
pool_wave = 1.5                 # Wellenamplitude ihrer Oberfläche (px)
pool_soft = 2.5                 # Weichheit des Übergangs Wachsschicht -> Blobs

# ====================================================================== Darstellung
edge_softness = 0.08            # Kantenweichheit der Blobs (Anti-Aliasing)
halo = 0.25                     # Schein um das Wachs
bottom_glow = 0.6               # Lampenschein von unten (0..1)
glow_height = 22.0              # Abklinghöhe des Lampenscheins (px)
brightness = 0.8                # 0..1, globale Helligkeit

# ====================================================================== Farbschema
color_hold = 900                # Frames, die ein Farbschema stehen bleibt
color_blend = 600               # Frames für den Übergang zum nächsten

target_fps = 30                 # None = so schnell wie möglich

UDP_IP = '192.168.178.50'
UDP_PORT = 2000

rng = cl.rng
N = cl.N
S = float(N)                    # Kantenlänge des Volumens

# ====================================================================== Geometrie
# Einbettung der Flächen in 3D: P = O + x * ex + y * ey (passend zu cubelib.resolve)
FACE_O = np.array([[64, 0, 64], [0, 0, 64], [0, 64, 64],
                   [64, 64, 64], [0, 64, 64], [0, 0, 0]], dtype=float)
FACE_EX = np.array([[0, 1, 0], [1, 0, 0], [0, -1, 0],
                    [-1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float)
FACE_EY = np.array([[0, 0, -1], [0, 0, -1], [0, 0, -1],
                    [0, 0, -1], [0, -1, 0], [0, 1, 0]], dtype=float)
CENTER = np.full(3, S / 2)


def face_center(f):
    return FACE_O[f] + (S / 2) * (FACE_EX[f] + FACE_EY[f])


def orientation(top, bottom):
    """Liefert (Achse, Vorzeichen) der Hochrichtung."""
    ct, cb = face_center(top), face_center(bottom)
    if top == bottom or not np.allclose(ct + cb, 2 * CENTER):
        raise ValueError(f"top_face={top} und bottom_face={bottom} liegen nicht gegenüber. "
                         "Gültige Paare: (0, 2), (1, 3), (4, 5) in beliebiger Reihenfolge.")
    up = ct - CENTER
    axis = int(np.argmax(np.abs(up)))
    return axis, (1.0 if up[axis] > 0 else -1.0)


UP_AXIS, UP_SIGN = orientation(top_face, bottom_face)
H_AXES = [k for k in range(3) if k != UP_AXIS]       # waagrechte Achsen


def height(p):
    """Höhe über dem Boden (0..64) für 3D-Punkte p[..., 3]."""
    return p[..., UP_AXIS] if UP_SIGN > 0 else S - p[..., UP_AXIS]


# Pixelmittelpunkte aller Panels in 3D, Reihenfolge [face, x, y] wie im Frame
_f, _x, _y = np.meshgrid(np.arange(cl.FACES), np.arange(N) + 0.5, np.arange(N) + 0.5, indexing='ij')
PIX = (FACE_O[_f] + _x[..., None] * FACE_EX[_f] + _y[..., None] * FACE_EY[_f]).reshape(-1, 3)
PIX_H = height(PIX)
PIX_T = np.ascontiguousarray(PIX.T, dtype=np.float32)          # (3, Pixel) für schnelles Rendern
GLOW = (bottom_glow * np.exp(-PIX_H / glow_height)).astype(np.float32)   # statischer Lampenschein

# Nur Pixel nahe dem Boden können zur Wachsschicht gehören
POOL_IDX = np.nonzero(PIX_H < pool_level + pool_wave * 1.5 + 8 * pool_soft)[0]
POOL_H = PIX_H[POOL_IDX].astype(np.float32)
POOL_A = (PIX[POOL_IDX, H_AXES[0]] - S / 2).astype(np.float32)  # waagrechte Koordinaten
POOL_B = (PIX[POOL_IDX, H_AXES[1]] - S / 2).astype(np.float32)


# ====================================================================== Farben
def hsv_to_rgb(hsv):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    k = (np.array([5.0, 3.0, 1.0]) + h[..., None] * 6.0) % 6.0
    return v[..., None] - v[..., None] * s[..., None] * np.clip(np.minimum(k, 4.0 - k), 0.0, 1.0)


def random_scheme():
    """Zufälliges, stimmiges Schema in HSV:
    [Hintergrund dunkel, Hintergrund Lampenschein, Wachs kühl, Wachs heiß]."""
    h = rng.random()
    off = rng.choice([0.5, 0.42, 0.58, 0.33, 0.67])          # Komplementär-/Triadenabstand
    hb = (h + off + rng.uniform(-0.04, 0.04)) % 1.0
    return np.array([
        [hb, rng.uniform(0.70, 0.95), rng.uniform(0.05, 0.09)],
        [hb, rng.uniform(0.60, 0.90), rng.uniform(0.22, 0.32)],
        [h, rng.uniform(0.85, 1.00), rng.uniform(0.60, 0.75)],
        [(h + rng.uniform(-0.07, 0.07)) % 1.0, rng.uniform(0.45, 0.70), 1.0],
    ])


class ColorScheme:
    """Hält ein Schema `color_hold` Frames und blendet dann weich (in HSV) zum nächsten."""

    def __init__(self):
        self.start = random_scheme()
        self.target = self.start.copy()
        self.hsv = self.start.copy()
        self.t = 0
        self._announce(self.hsv)

    @staticmethod
    def _announce(s):
        print(f"Farbschema: Wachs {s[2, 0] * 360:.0f}°, Hintergrund {s[0, 0] * 360:.0f}°")

    def step(self):
        self.t += 1
        period = color_hold + color_blend
        if self.t % period == 0:
            self.start = self.hsv.copy()
            self.target = random_scheme()
            self._announce(self.target)
        phase = self.t % period
        if phase <= color_blend and self.t >= period:
            s = phase / color_blend
            s = s * s * (3 - 2 * s)
            dh = (self.target[:, 0] - self.start[:, 0] + 0.5) % 1.0 - 0.5   # kürzester Weg im Farbkreis
            self.hsv[:, 0] = (self.start[:, 0] + s * dh) % 1.0
            self.hsv[:, 1:] = (1 - s) * self.start[:, 1:] + s * self.target[:, 1:]
        return hsv_to_rgb(self.hsv)


# ====================================================================== Wachs
class Wax:
    def __init__(self, n):
        self.n = n
        self.R0 = rng.uniform(*blob_radius, n)
        self.phase = rng.random(n) * 2 * np.pi
        self.pos = rng.uniform(wall_depth, S - wall_depth, (n, 3))
        h = rng.uniform(2, S - 2, n)
        self.pos[:, UP_AXIS] = h if UP_SIGN > 0 else S - h
        self.vel = np.zeros((n, 3))
        self.T = rng.random(n)
        self.heat_gain = rng.uniform(0.7, 1.3, n)       # individuelle "Wärmekapazität"
        self.up = np.zeros(3)
        self.up[UP_AXIS] = UP_SIGN
        self.t = 0

    @property
    def R(self):
        return self.R0 * (1 + radius_breathe * np.sin(0.004 * self.t + self.phase))

    def step(self):
        self.t += 1
        h = height(self.pos)
        R = self.R

        # Temperatur: Heizen nur am Boden, Abkühlen vor allem unter der Decke.
        # Dazwischen behält ein Blob seine Temperatur -> er steigt ganz auf bzw. sinkt ganz ab.
        heat = np.exp(-(h / heat_depth) ** 2)              # scharf begrenzte Zonen,
        cool = np.exp(-((S - h) / cool_depth) ** 2)        # sonst schweben Blobs in der Mitte
        self.T += heat_rate * self.heat_gain * heat * (1 - self.T)
        self.T -= cool_rate / self.heat_gain * (cool + cool_base) * self.T
        np.clip(self.T, 0, 1, out=self.T)

        acc = np.outer(buoyancy * (self.T - 0.5), self.up)

        # Zur Seitenwand driften (sichtbar an der "Glaswand"), nahe der Decke schwächer
        q = self.pos[:, H_AXES] - S / 2
        dom = np.argmax(np.abs(q), axis=1)
        qd = q[np.arange(self.n), dom]
        target = np.sign(qd) * (S / 2 - wall_depth - 0.3 * R)
        w = 1.0 - 0.7 * np.exp(-(S - h) / 10.0)
        f = wall_pull * w * (target - qd)
        for k in range(2):
            acc[:, H_AXES[k]] += np.where(dom == k, f, 0.0)

        # Seitliches Umherwandern
        for k in H_AXES:
            acc[:, k] += wander * rng.standard_normal(self.n)

        # Schwache gegenseitige Abstoßung
        d = self.pos[:, None, :] - self.pos[None, :, :]
        dist = np.sqrt((d ** 2).sum(-1)) + np.eye(self.n) * 1e3
        overlap = np.clip(0.7 * (R[:, None] + R[None, :]) - dist, 0, None)
        acc += blob_repel * ((overlap / dist)[..., None] * d).sum(1)

        self.vel = drag * self.vel + acc
        sp = np.linalg.norm(self.vel, axis=1)
        fast = sp > max_speed
        self.vel[fast] *= (max_speed / sp[fast])[:, None]
        self.pos += self.vel

        # Wände: waagrecht im Volumen halten, Boden/Decke abhängig von der Blobgröße
        for k in H_AXES:
            lo, hi = 0.5, S - 0.5
            hit = (self.pos[:, k] < lo) | (self.pos[:, k] > hi)
            self.pos[:, k] = np.clip(self.pos[:, k], lo, hi)
            self.vel[hit, k] = 0
        h = np.clip(height(self.pos), 1.0, S - 0.35 * R)
        newp = h if UP_SIGN > 0 else S - h
        hit = newp != self.pos[:, UP_AXIS]
        self.pos[:, UP_AXIS] = newp
        self.vel[hit, UP_AXIS] = 0


# ====================================================================== Darstellung
def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


class Renderer:
    def __init__(self):
        self.frame = np.zeros((cl.FACES, N, N, 3), dtype=np.uint8)
        self.t = 0

    def render(self, wax, colors):
        self.t += 1
        t = np.float32(self.t)
        f32 = np.float32
        bg_dark, bg_glow, wax_cold, wax_hot = (c.astype(np.float32)[:, None] for c in colors)

        # Metaball-Feld der Blobs auf allen Panelpixeln: contrib[blob, pixel]
        C = wax.pos.astype(np.float32)
        d2 = (PIX_T[0][None, :] - C[:, 0, None]) ** 2
        d2 += (PIX_T[1][None, :] - C[:, 1, None]) ** 2
        d2 += (PIX_T[2][None, :] - C[:, 2, None]) ** 2
        d2 += f32(1.0)
        contrib = (wax.R ** 2).astype(np.float32)[:, None] / d2
        F = contrib.sum(0)
        heat = wax.T.astype(np.float32) @ contrib            # Summe T_i * Beitrag_i

        # Wachsschicht am Boden: wogende Oberfläche, langsam wandernde Konvektionsmuster
        A, B = POOL_A, POOL_B
        level = f32(pool_level) + f32(pool_wave) * (
            np.sin(f32(0.13) * A + f32(0.011) * t) * np.cos(f32(0.09) * B - f32(0.007) * t)
            + f32(0.5) * np.sin(f32(0.21) * (A + B) - f32(0.016) * t))
        pool = np.exp(np.minimum((level - POOL_H) / f32(pool_soft), f32(8.0)))
        pool_temp = f32(0.72) + f32(0.14) * (
            np.sin(f32(0.17) * A - f32(0.006) * t + f32(1.3) * np.sin(f32(0.11) * B + f32(0.004) * t))
            + np.cos(f32(0.15) * B + f32(0.005) * t))
        F[POOL_IDX] += pool
        heat[POOL_IDX] += pool * pool_temp
        temp = heat / F                                      # Wachstemperatur am Pixel

        m = smoothstep(f32(1 - edge_softness), f32(1 + edge_softness), F)
        inner = np.clip((F - f32(1.0)) * f32(1 / 3), f32(0), f32(1))
        shade = f32(0.78) + f32(0.22) * inner                # Kern etwas heller -> Plastizität
        wcol = (wax_cold + (wax_hot - wax_cold) * temp) * shade
        bg = bg_dark + (bg_glow - bg_dark) * GLOW
        glow = f32(halo) * smoothstep(f32(0.3), f32(1.0), F) * (f32(1) - m)

        col = bg + (wcol - bg) * m + glow * wcol             # (3, Pixel)
        col *= f32(255 * brightness)
        np.clip(col, 0, 255, out=col)
        self.frame.reshape(-1, 3)[:] = col.T
        return self.frame


# ====================================================================== Hauptschleife
def main():
    c = cube.Cube(UDP_IP, UDP_PORT)
    limiter = cl.FrameLimiter(target_fps)
    print(f"Oben: Fläche {top_face}, unten: Fläche {bottom_face}")
    wax = Wax(n_blobs)
    colors = ColorScheme()
    renderer = Renderer()
    while True:
        wax.step()
        c.sendFrame(renderer.render(wax, colors.step()))
        limiter.tick()


if __name__ == "__main__":
    main()
