#!/usr/bin/env python3
"""Physarum – Schleimpilz-Simulation auf der Würfeloberfläche.

Tausende Agenten laufen über den Würfel, hinterlassen eine Duftspur und
steuern zur stärksten Spur vor sich. Die Spur verdunstet und verläuft
(auch über die Würfelkanten hinweg). So bilden sich adrige Netzwerke, die
sich durch langsam "atmende" Parameter, gelegentliche Ausbrüche (Blooms)
und überblendende Farbpaletten ständig umbauen.
"""
import numpy as np
import cube
import cubelib as cl

# ---------------------------------------------------------------- Simulation
n_agents = 9000
speed = 1.0                     # Pixel pro Frame
sensor_dist = 6.0               # Abstand der Sensoren vor dem Agenten
sensor_angle_deg = (30, 60)     # Sensorwinkel pendelt zwischen diesen Werten
turn_angle_deg = (25, 45)       # Drehwinkel pendelt zwischen diesen Werten
breathe_period = 1200           # Frames für eine volle "Atem"-Periode
jitter_deg = 6                  # zufälliges Zittern der Laufrichtung
deposit = 1.0                   # Spur pro Agent und Frame
decay = 0.90                    # Verdunstung pro Frame
diffuse = 0.35                  # Anteil der Spur, der pro Frame verläuft
trail_cap = 5.0                 # Sättigung der Spur – verhindert, dass ein Strang alles schluckt

bloom_every = 900               # alle n Frames ein Ausbruch ...
bloom_fraction = 0.35           # ... mit diesem Anteil der Agenten

# ---------------------------------------------------------------- Darstellung
brightness = 0.75               # 0..1, globale Helligkeit
trail_share = 0.5               # Anteil der Palette für die Spur (Rest: wandernde Agenten)
glow_threshold = 1.0            # Agentendichte, ab der Lichtpunkte aufleuchten ...
glow_range = 3.5                # ... und über diesen Bereich bis zur vollen Helligkeit
palette_every = 2000            # Palettenwechsel alle n Frames
palette_fade = 240              # Überblenddauer in Frames

# Farbverläufe: (Position 0..1, (R, G, B))
PALETTES = [
    ("Glut",    [(0.00, (0, 0, 0)), (0.18, (25, 4, 45)), (0.42, (120, 18, 95)),
                 (0.66, (235, 85, 45)), (0.86, (255, 175, 70)), (1.00, (255, 235, 175))]),
    ("Lagune",  [(0.00, (0, 0, 0)), (0.20, (0, 14, 45)), (0.45, (0, 80, 120)),
                 (0.70, (30, 185, 165)), (0.88, (160, 240, 195)), (1.00, (235, 255, 240))]),
    ("Flieder", [(0.00, (0, 0, 0)), (0.20, (15, 8, 50)), (0.45, (70, 40, 150)),
                 (0.68, (175, 95, 210)), (0.88, (250, 175, 215)), (1.00, (255, 235, 245))]),
]

target_fps = 30                 # None = so schnell wie möglich

UDP_IP = '192.168.178.50'
UDP_PORT = 2000
rng = cl.rng
N = cl.N


# ---------------------------------------------------------------- Hilfsfunktionen
def make_lut(stops, size=256):
    """Farbverlauf als (size, 3)-Tabelle."""
    pos = np.array([s[0] for s in stops])
    col = np.array([s[1] for s in stops], dtype=float)
    t = np.linspace(0, 1, size)
    return np.stack([np.interp(t, pos, col[:, k]) for k in range(3)], axis=-1)


def cube_blur(field):
    """3x3-Mittelwert über die Würfeloberfläche (kantenübergreifend)."""
    p = field.ravel()[cl.PAD_IDX]
    return (p[:, :-2, :-2] + p[:, :-2, 1:-1] + p[:, :-2, 2:] +
            p[:, 1:-1, :-2] + p[:, 1:-1, 1:-1] + p[:, 1:-1, 2:] +
            p[:, 2:, :-2] + p[:, 2:, 1:-1] + p[:, 2:, 2:]) * (1.0 / 9.0)


def rotate(vx, vy, ang):
    c, s = np.cos(ang), np.sin(ang)
    return vx * c - vy * s, vx * s + vy * c


class Agents:
    def __init__(self, n):
        self.face = rng.integers(0, cl.FACES, n)
        self.x = rng.random(n) * N
        self.y = rng.random(n) * N
        a = rng.random(n) * 2 * np.pi
        self.vx, self.vy = np.cos(a), np.sin(a)

    def sense(self, trail, sa):
        """Spurstärke an den drei Sensoren (vorne, links, rechts) – in einem Durchgang."""
        n = self.x.size
        lx, ly = rotate(self.vx, self.vy, sa)
        rx, ry = rotate(self.vx, self.vy, -sa)
        dx = np.concatenate([self.vx, lx, rx])
        dy = np.concatenate([self.vy, ly, ry])
        f = np.tile(self.face, 3)
        sx = np.tile(self.x, 3) + sensor_dist * dx
        sy = np.tile(self.y, 3) + sensor_dist * dy
        cl.wrap_particles(f, sx, sy, dx, dy)
        v = trail[f, sx.astype(np.intp), sy.astype(np.intp)]
        return v[:n], v[n:2 * n], v[2 * n:]

    def step(self, trail, sa, ra):
        n = self.x.size
        F, L, R = self.sense(trail, sa)

        turn = np.zeros(n)
        both = (F < L) & (F < R)                       # Gabelung: zufällig
        turn[both] = np.where(rng.random(both.sum()) < 0.5, ra, -ra)
        rest = ~both & ~((F > L) & (F > R))            # geradeaus, wenn vorne am stärksten
        turn[rest & (L > R)] = ra
        turn[rest & (R > L)] = -ra
        turn += np.radians(jitter_deg) * (rng.random(n) * 2 - 1)

        self.vx, self.vy = rotate(self.vx, self.vy, turn)
        self.x += speed * self.vx
        self.y += speed * self.vy
        cl.wrap_particles(self.face, self.x, self.y, self.vx, self.vy)

    def bloom(self, fraction):
        """Ein Teil der Agenten bricht strahlenförmig von einem Zufallspunkt aus."""
        n = self.x.size
        idx = rng.choice(n, int(n * fraction), replace=False)
        self.face[idx] = rng.integers(0, cl.FACES)
        self.x[idx] = rng.uniform(16, 48)
        self.y[idx] = rng.uniform(16, 48)
        a = rng.random(idx.size) * 2 * np.pi
        self.vx[idx], self.vy[idx] = np.cos(a), np.sin(a)


class Renderer:
    def __init__(self):
        self.luts = [make_lut(stops) * brightness for _, stops in PALETTES]
        self.cur, self.nxt, self.fade_t = 0, None, 0
        self.lut8 = self.luts[0].astype(np.uint8)

    def next_palette(self):
        self.nxt = (self.cur + 1) % len(self.luts)
        self.fade_t = 0
        print("Palette ->", PALETTES[self.nxt][0])

    def render(self, trail, density):
        # Spur füllt den dunklen Teil der Palette, die (geglättete) Agentendichte
        # setzt helle, entlang der Adern wandernde Lichtpunkte darauf.
        t = trail_share * (trail / trail_cap) ** 0.7
        t += (1 - trail_share) * np.clip((density - glow_threshold) / glow_range, 0.0, 1.0)
        idx = (np.minimum(t, 1.0) * 255).astype(np.intp)

        if self.nxt is not None:
            a = self.fade_t / palette_fade
            self.lut8 = ((1 - a) * self.luts[self.cur] + a * self.luts[self.nxt]).astype(np.uint8)
            self.fade_t += 1
            if self.fade_t > palette_fade:
                self.cur, self.nxt = self.nxt, None
        return self.lut8[idx]


def main():
    c = cube.Cube(UDP_IP, UDP_PORT)
    limiter = cl.FrameLimiter(target_fps)
    agents = Agents(n_agents)
    renderer = Renderer()
    trail = np.zeros((cl.FACES, N, N))

    sa_lo, sa_hi = np.radians(sensor_angle_deg)
    ra_lo, ra_hi = np.radians(turn_angle_deg)
    step = 0
    while True:
        # Parameter "atmen" lassen – das Netzwerk verdichtet und lockert sich
        b = 0.5 - 0.5 * np.cos(2 * np.pi * step / breathe_period)
        sa = sa_lo + (sa_hi - sa_lo) * b
        ra = ra_lo + (ra_hi - ra_lo) * (1 - b)

        agents.step(trail, sa, ra)
        _, ix, iy = cl.pixel_index(agents.face, agents.x, agents.y)
        counts = cl.particle_counts(agents.face, ix, iy)
        trail += deposit * counts
        trail = decay * ((1 - diffuse) * trail + diffuse * cube_blur(trail))
        np.minimum(trail, trail_cap, out=trail)

        c.sendFrame(renderer.render(trail, cube_blur(counts.astype(float))))

        step += 1
        if step % bloom_every == 0:
            agents.bloom(bloom_fraction)
        if step % palette_every == 0:
            renderer.next_palette()
        limiter.tick()


if __name__ == "__main__":
    main()
