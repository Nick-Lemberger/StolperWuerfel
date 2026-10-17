#!/usr/bin/env python3
"""Aurora – leuchtende Partikel in einer wirbelnden Curl-Strömung.

Ein glattes Zufallspotential psi auf der Würfeloberfläche wird um 90° gedrehter
Gradient zum Geschwindigkeitsfeld (divergenzfrei, d. h. echte Wirbel ohne
Quellen und Senken). psi überblendet ständig zwischen zwei Zufallsfeldern,
die Wirbel wandern, verschmelzen und zerfallen. Die Partikel ziehen
Leuchtschweife; ihre Farbe ergibt sich aus dem Potential am Ort, sodass
zusammengehörige Strömungsbänder dieselbe Farbe tragen.
"""
import numpy as np
import cube
import cubelib as cl

# ---------------------------------------------------------------- Simulation
n_particles = 3000
max_speed = 0.7                 # Pixel pro Frame (95. Perzentil des Feldes)
inertia = 0.85                  # 0 = folgt dem Feld sofort, ->1 = träge Bahnen
field_scale = 70                # Glättungsschritte: größer = größere Wirbel
morph_frames = 600              # Frames für eine Feldüberblendung (Viertelperiode)
respawn_rate = 0.002            # Anteil der Partikel, die pro Frame neu erscheinen

# ---------------------------------------------------------------- Darstellung
trail_fade = 0.90               # Nachleuchten pro Frame (0..1)
particle_gain = 0.55            # Helligkeitsbeitrag eines Partikels
brightness = 0.8                # 0..1, globale Helligkeit
color_drift = 0.00025           # langsames Wandern der Farbskala pro Frame

# Zyklischer Farbverlauf (Anfang = Ende): Nordlicht
PALETTE = [(0.00, (20, 60, 200)),
           (0.22, (0, 170, 190)),
           (0.42, (40, 230, 120)),
           (0.62, (170, 240, 90)),
           (0.80, (220, 70, 200)),
           (1.00, (20, 60, 200))]

target_fps = 30                 # None = so schnell wie möglich

UDP_IP = '192.168.178.50'
UDP_PORT = 2000
rng = cl.rng
N = cl.N


# ---------------------------------------------------------------- Hilfsfunktionen
def make_lut(stops, size=512):
    pos = np.array([s[0] for s in stops])
    col = np.array([s[1] for s in stops], dtype=float) / 255.0
    t = np.linspace(0, 1, size, endpoint=False)
    return np.stack([np.interp(t, pos, col[:, k]) for k in range(3)], axis=-1)


def pad(field):
    return field.ravel()[cl.PAD_IDX]


def cube_blur(field):
    """3x3-Mittelwert über die Würfeloberfläche (kantenübergreifend)."""
    p = pad(field)
    return (p[:, :-2, :-2] + p[:, :-2, 1:-1] + p[:, :-2, 2:] +
            p[:, 1:-1, :-2] + p[:, 1:-1, 1:-1] + p[:, 1:-1, 2:] +
            p[:, 2:, :-2] + p[:, 2:, 1:-1] + p[:, 2:, 2:]) * (1.0 / 9.0)


def random_potential():
    """Glattes Zufallsfeld auf dem Würfel, normiert auf Standardabweichung 1."""
    psi = rng.standard_normal((cl.FACES, N, N))
    for _ in range(field_scale):
        psi = cube_blur(psi)
    psi -= psi.mean()
    return psi / psi.std()


def curl(psi):
    """Um 90° gedrehter Gradient -> divergenzfreies Feld (6, 64, 64, 2).
    Alle Kantenübergänge des Würfels sind Drehungen, daher ist die Drehung
    auf allen Flächen konsistent."""
    p = pad(psi)
    gx = 0.5 * (p[:, 2:, 1:-1] - p[:, :-2, 1:-1])
    gy = 0.5 * (p[:, 1:-1, 2:] - p[:, 1:-1, :-2])
    return np.stack([gy, -gx], axis=-1)


class Field:
    """psi(t) = A cos(phi) + B sin(phi); bei phi = 90° wird B zu A und B neu gewürfelt."""

    def __init__(self):
        self.A, self.B = random_potential(), random_potential()
        self._prepare()
        self.phi = 0.0

    def _prepare(self):
        self.uA, self.uB = curl(self.A), curl(self.B)
        speed = np.hypot(self.uA[..., 0], self.uA[..., 1])
        self.gain = max_speed / max(np.percentile(speed, 95), 1e-9)

    def advance(self):
        self.phi += 0.5 * np.pi / morph_frames
        if self.phi >= 0.5 * np.pi:
            self.phi -= 0.5 * np.pi
            self.A, self.B = self.B, random_potential()
            self._prepare()
        c, s = np.cos(self.phi), np.sin(self.phi)
        return (c * self.uA + s * self.uB) * self.gain, c * self.A + s * self.B


class Particles:
    def __init__(self, n):
        self.face = rng.integers(0, cl.FACES, n)
        self.x = rng.random(n) * N
        self.y = rng.random(n) * N
        self.vx = np.zeros(n)
        self.vy = np.zeros(n)
        self.hue = rng.random(n) * 0.08        # kleine individuelle Farbstreuung

    def step(self, u):
        self.x += self.vx
        self.y += self.vy
        cl.wrap_particles(self.face, self.x, self.y, self.vx, self.vy)
        _, ix, iy = cl.pixel_index(self.face, self.x, self.y)
        self.vx = inertia * self.vx + (1 - inertia) * u[self.face, ix, iy, 0]
        self.vy = inertia * self.vy + (1 - inertia) * u[self.face, ix, iy, 1]

        # Ein paar Partikel neu verteilen, damit sich keine Klumpen bilden
        k = rng.random(self.x.size) < respawn_rate
        if k.any():
            m = int(k.sum())
            self.face[k] = rng.integers(0, cl.FACES, m)
            self.x[k] = rng.random(m) * N
            self.y[k] = rng.random(m) * N
            self.vx[k] = 0.0
            self.vy[k] = 0.0
        return ix, iy


def main():
    c = cube.Cube(UDP_IP, UDP_PORT)
    limiter = cl.FrameLimiter(target_fps)
    lut = make_lut(PALETTE)
    field = Field()
    parts = Particles(n_particles)

    glow = np.zeros((cl.FACES * N * N, 3))
    frame = np.zeros((cl.FACES, N, N, 3), dtype=np.uint8)
    offset = 0.0
    while True:
        u, psi = field.advance()
        ix, iy = parts.step(u)

        # Farbe aus dem Potential am Partikelort (tanh staucht auf 0..1), zyklisch
        val = 0.5 + 0.5 * np.tanh(0.8 * psi[parts.face, ix, iy])
        cidx = ((val + parts.hue + offset) % 1.0 * len(lut)).astype(np.intp)
        col = lut[cidx] * particle_gain

        flat = (parts.face * N + ix) * N + iy
        glow *= trail_fade
        for k in range(3):
            glow[:, k] += np.bincount(flat, weights=col[:, k], minlength=glow.shape[0])

        # Weiche Sättigung, damit dichte Stellen nicht hart ausbrennen
        out = 1.0 - np.exp(-1.6 * glow)
        frame[:] = (out * (255 * brightness)).reshape(cl.FACES, N, N, 3)
        c.sendFrame(frame)

        offset = (offset + color_drift) % 1.0
        limiter.tick()


if __name__ == "__main__":
    main()
