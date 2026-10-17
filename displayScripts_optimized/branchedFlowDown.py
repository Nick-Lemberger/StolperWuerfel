#!/usr/bin/env python3
"""Partikel starten oben in der Mitte und fließen verzweigt die Seiten hinab."""
import numpy as np
from scipy.ndimage import gaussian_filter
import cube
import cubelib as cl

# Red tracers
# cIncr = [12, 5, 5]; cDecr = [1, 2, 2]; cLim = [255, 200, 200]
# Green tracers
# cIncr = [0, 5, 0]; cDecr = [5, 1, 5]; cLim = [200, 255, 120]
# Fireflies
cIncr = [20, 10, 10]
cDecr = [1, 4, 1]
cLim = [255, 150, 120]

lifeTime = 1000          # Lebensdauer eines Partikels in Frames
launch_per_frame = 2     # neue Partikel pro Frame
maxSpeed = 0.1           # Pixel pro Frame (konstant)
landscape_every = 1000   # neue Landschaft alle n Frames
sigma = 2                # Glättung des Vektorfelds

target_fps = None        # z. B. 30; None = so schnell wie möglich

UDP_IP = '192.168.178.50'
UDP_PORT = 2000
rng = cl.rng


def random_landscape():
    """Vektorfeld [face, x, y, (vx, vy)] wie randomLandscape() im Original.
    Oben: Streuung in alle Richtungen; Ring: Drift nach unten; Unterseite: 0."""
    v = np.zeros((cl.FACES, cl.N, cl.N, 2))
    v[4] = 0.01 * (rng.random((cl.N, cl.N, 2)) * 2 - 1)
    v[0:4, :, :, 0] = 0.05 * (rng.random((4, cl.N, cl.N)) * 2 - 1)
    v[0:4, :, :, 1] = 0.01
    return gaussian_filter(v, sigma=(0, sigma, sigma, 0))


class Swarm:
    def __init__(self):
        self.face = np.empty(0, dtype=np.intp)
        self.x = np.empty(0)
        self.y = np.empty(0)
        self.vx = np.empty(0)
        self.vy = np.empty(0)
        self.tLeft = np.empty(0, dtype=np.int64)

    def launch(self, k):
        angle = 2 * np.pi * (2 * rng.random(k) - 1)
        self.face = np.concatenate([self.face, np.full(k, 4, dtype=np.intp)])
        self.x = np.concatenate([self.x, np.full(k, 31.0)])
        self.y = np.concatenate([self.y, np.full(k, 31.0)])
        self.vx = np.concatenate([self.vx, maxSpeed * np.cos(angle)])
        self.vy = np.concatenate([self.vy, maxSpeed * np.sin(angle)])
        self.tLeft = np.concatenate([self.tLeft, np.full(k, lifeTime, dtype=np.int64)])

    def advance(self, vField, dt=1):
        # Abgelaufene Partikel entfernen
        alive = self.tLeft >= 1
        if not alive.all():
            for name in ("face", "x", "y", "vx", "vy", "tLeft"):
                setattr(self, name, getattr(self, name)[alive])
        self.tLeft -= 1

        face, x, y, vx, vy = self.face, self.x, self.y, self.vx, self.vy
        x += vx * dt
        y += vy * dt
        cl.wrap_particles(face, x, y, vx, vy)
        _, ix, iy = cl.pixel_index(face, x, y)
        vx += vField[face, ix, iy, 0]
        vy += vField[face, ix, iy, 1]
        # Auf konstante Geschwindigkeit normieren
        sp = np.sqrt(vx ** 2 + vy ** 2)
        nz = sp > 0
        vx[nz] = maxSpeed * vx[nz] / sp[nz]
        vy[nz] = maxSpeed * vy[nz] / sp[nz]
        return ix, iy


def main():
    c = cube.Cube(UDP_IP, UDP_PORT)
    colors = cl.ColorFade(cIncr, cDecr, cLim)
    limiter = cl.FrameLimiter(target_fps)

    swarm = Swarm()
    vField = random_landscape()
    step = 1
    while True:
        swarm.launch(launch_per_frame)
        ix, iy = swarm.advance(vField)
        c.sendFrame(colors.update(cl.particle_counts(swarm.face, ix, iy)))
        if step % landscape_every == 0:
            vField = random_landscape()
        step += 1
        limiter.tick()


if __name__ == "__main__":
    main()
