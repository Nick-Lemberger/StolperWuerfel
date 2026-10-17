#!/usr/bin/env python3
"""Partikel, die einem geglätteten Zufalls-Vektorfeld über den Würfel folgen."""
import numpy as np
from scipy.ndimage import gaussian_filter
import cube
import cubelib as cl

# Red tracers
# cIncr = [12, 10, 10]; cDecr = [1, 2, 2]; cLim = [255, 200, 200]
# Green tracers
cIncr = [0, 12, 0]
cDecr = [5, 1, 5]
cLim = [200, 255, 120]
# Fireflies
# cIncr = [40, 25, 40]; cDecr = [5, 4, 1]; cLim = [255, 150, 120]

pNum = 1000          # Anzahl Partikel
maxSpeed = 0.5       # Pixel pro Frame
field_every = 500    # neues Vektorfeld alle n Frames
sigma = 10           # Glättung des Vektorfelds

target_fps = None    # z. B. 30; None = so schnell wie möglich

UDP_IP = '192.168.178.50'
UDP_PORT = 2000
rng = cl.rng


def change_fields():
    """Vektorfeld [face, x, y, (vx, vy)] wie changeFields() im Original."""
    v = 0.2 * (rng.random((cl.FACES, cl.N, cl.N, 2)) * 2 - 1)
    v = gaussian_filter(v, sigma=(0, sigma, sigma, 0))   # 2D-Glättung je Fläche und Komponente
    I, J = cl.wall_axes()
    v[..., 0] += 0.001 * (J - 32)
    v[..., 1] += 0.001 * (I - 32)
    return v


def init_particles():
    angle = 2 * np.pi * rng.random(pNum)
    vx = maxSpeed * np.cos(angle)
    vy = maxSpeed * np.sin(angle)
    x = rng.integers(0, cl.N, pNum).astype(float)
    y = rng.integers(0, cl.N, pNum).astype(float)
    face = rng.integers(0, cl.FACES, pNum)
    return face, x, y, vx, vy


def advance(face, x, y, vx, vy, vField, dt=1):
    x += vx * dt
    y += vy * dt
    cl.wrap_particles(face, x, y, vx, vy)
    _, ix, iy = cl.pixel_index(face, x, y)
    vx += vField[face, ix, iy, 0]
    vy += vField[face, ix, iy, 1]
    # Geschwindigkeit begrenzen
    sp = np.sqrt(vx ** 2 + vy ** 2)
    fast = sp > maxSpeed
    vx[fast] = maxSpeed * vx[fast] / sp[fast]
    vy[fast] = maxSpeed * vy[fast] / sp[fast]
    return ix, iy


def main():
    c = cube.Cube(UDP_IP, UDP_PORT)
    colors = cl.ColorFade(cIncr, cDecr, cLim)
    limiter = cl.FrameLimiter(target_fps)

    face, x, y, vx, vy = init_particles()
    vField = change_fields()
    step = 1
    while True:
        ix, iy = advance(face, x, y, vx, vy, vField)
        c.sendFrame(colors.update(cl.particle_counts(face, ix, iy)))
        if step % field_every == 0:
            vField = change_fields()
        step += 1
        limiter.tick()


if __name__ == "__main__":
    main()
