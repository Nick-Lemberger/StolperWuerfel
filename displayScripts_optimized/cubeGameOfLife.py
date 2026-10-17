#!/usr/bin/env python3
"""Game of Life auf dem Würfel – Neustart bei Überbevölkerung, zufällige Farben."""
import numpy as np
import cube
import cubelib as cl

# Game-of-Life-Regeln: Überleben / Geburt
# Original rules
ruleS = [3, 2, 7]
ruleB = [3]
# The Blobb
# ruleS = [3, 5]
# ruleB = [3, 4]

start_density = 0.20
cutoff_pop = 5000

# Preset "Slow Winter" (wird beim Start durch Zufallsfarben ersetzt)
cIncr = [4, 6, 10]
cDecr = [1, 1, 1]
cLim = [255, 255, 255]

target_fps = None   # z. B. 30; None = so schnell wie möglich

UDP_IP = '192.168.178.50'
UDP_PORT = 2000


def main():
    c = cube.Cube(UDP_IP, UDP_PORT)
    rule = cl.make_rule(ruleS, ruleB)
    colors = cl.ColorFade(cIncr, cDecr, cLim)
    colors.set(*cl.random_colors_simple())
    limiter = cl.FrameLimiter(target_fps)

    old = cl.empty_world()
    new = cl.seed(cl.empty_world(), start_density)
    while True:
        # Neu säen bei Überbevölkerung
        if np.count_nonzero(old) > cutoff_pop:
            new = cl.seed(cl.empty_world(), start_density)
            colors.set(*cl.random_colors_simple())

        old = new
        c.sendFrame(colors.update(old))
        new = cl.life_step(old, rule)
        limiter.tick()


if __name__ == "__main__":
    main()
