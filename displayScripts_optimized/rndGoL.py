#!/usr/bin/env python3
"""Game of Life auf dem Würfel mit zufällig wechselnden Regelsätzen."""
import numpy as np
import cube
import cubelib as cl

# Regelsatz: [Überleben, Geburt, Startdichte]
bloomBoom = [[3, 4], [3], 0.5]
wildGrowth = [[3, 2, 7], [3], 0.18]
blobbs = [[3, 5], [3, 4], 0.18]
gol = [[2, 3], [3], 0.32]

ruleList = [gol, bloomBoom, wildGrowth, blobbs]
ruleListNames = ["gol", "bloomBoom", "wildGrowth", "blobbs"]

max_pop = 6000         # Neu säen oberhalb ...
min_pop = 100          # ... bzw. unterhalb dieser Population
popcheck = 100         # alle n Generationen prüfen
max_lightPower = 500   # maximale Summe der cLim

p_sprinkle = 0.05      # Wahrscheinlichkeit pro Frame für Nachsäen
p_rule_change = 0.01   # Wahrscheinlichkeit pro Frame für Regelwechsel

# Preset "Sakura Kiss" (wird beim Start durch Zufallsfarben ersetzt)
cIncr = [27, 26, 48]
cDecr = [2, 5, 3]
cLim = [181, 150, 152]

target_fps = None   # z. B. 30; None = so schnell wie möglich

UDP_IP = '192.168.178.50'
UDP_PORT = 2000


def random_rule():
    """Zufälligen Regelsatz wählen -> (Lookup-Tabellen, Startdichte)."""
    k = int(cl.rng.integers(len(ruleList)))
    ruleS, ruleB, density = ruleList[k]
    print("Rule: " + ruleListNames[k])
    return cl.make_rule(ruleS, ruleB), density


def main():
    c = cube.Cube(UDP_IP, UDP_PORT)
    colors = cl.ColorFade(cIncr, cDecr, cLim)
    limiter = cl.FrameLimiter(target_fps)
    rng = cl.rng

    rule, start_density = random_rule()
    colors.set(*cl.random_colors_lightpower(max_lightPower))
    old = cl.empty_world()
    new = cl.seed(cl.empty_world(), start_density)

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
        if rng.random() < p_sprinkle:
            cl.seed(new, start_density / 2)

        # Zufälliger Regelwechsel
        if rng.random() < p_rule_change:
            rule, start_density = random_rule()

        old = new
        c.sendFrame(colors.update(old))
        gen += 1
        new = cl.life_step(old, rule)
        limiter.tick()


if __name__ == "__main__":
    main()
