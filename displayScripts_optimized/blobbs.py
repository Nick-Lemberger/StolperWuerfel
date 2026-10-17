#!/usr/bin/env python3
"""Game of Life "Devouring Blobbs" auf dem Würfel."""
import numpy as np
import cube
import cubelib as cl

# Devouring Blobbs: Überleben / Geburt
ruleS = [3, 5]
ruleB = [3, 4]

start_density = 0.18   # Wahrscheinlichkeit pro Zelle beim Säen
max_pop = 5000         # Neu säen oberhalb ...
min_pop = 1000         # ... bzw. unterhalb dieser Population
popcheck = 50          # alle n Generationen prüfen
max_lightPower = 500   # maximale Summe der cLim

# Preset "Sakura Kiss" (wird beim Start durch Zufallsfarben ersetzt)
cIncr = [27, 26, 48]
cDecr = [2, 5, 3]
cLim = [181, 150, 152]

target_fps = None   # z. B. 30; None = so schnell wie möglich

UDP_IP = '192.168.178.50'
UDP_PORT = 2000


def main():
    c = cube.Cube(UDP_IP, UDP_PORT)
    rule = cl.make_rule(ruleS, ruleB)
    colors = cl.ColorFade(cIncr, cDecr, cLim)
    limiter = cl.FrameLimiter(target_fps)

    old = cl.empty_world()
    new = cl.seed(cl.empty_world(), start_density)
    colors.set(*cl.random_colors_lightpower(max_lightPower))

    gen = 1
    while True:
        # Neu säen bei Über- oder Unterbevölkerung
        if gen % popcheck == 0:
            pop = np.count_nonzero(old)
            if pop > max_pop or pop < min_pop:
                new[:] = 0
                cl.seed(new, start_density)
                colors.set(*cl.random_colors_lightpower(max_lightPower))

        # Etwas Chaos einstreuen
        if (gen + 25) % 50 == 0:
            cl.seed(new, start_density / 1.5)

        old = new
        c.sendFrame(colors.update(old))
        gen += 1
        new = cl.life_step(old, rule)
        limiter.tick()


if __name__ == "__main__":
    main()
