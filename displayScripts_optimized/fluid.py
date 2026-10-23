#!/usr/bin/env python3
"""Flüssigkeit im Würfel, die der gemessenen Schwerkraft und Bewegung folgt.

Physik: partikelbasierte Flüssigkeit nach "Double Density Relaxation"
(Clavet, Beaudoin, Poulin 2005). Jeder Partikel ist ein kleines Flüssigkeits-
paket; ein Dichte- und ein Nah-Druckterm halten das Volumen konstant und lassen
die Flüssigkeit zu weichen, zusammenhängenden Tropfen verschmelzen. Das
Verfahren ist auch mit großen Zeitschritten stabil – ideal für den Pi.

Antrieb (aus cubesensor.read(), alles im Würfelsystem):
  - Schwerkraft entgegen der gemessenen Hochrichtung `up`
  - Bewegung des Würfels: Scheinkraft entgegen `lin_acc` -> Schwappen
  - Drehung des Würfels (Gyro): die Flüssigkeit bleibt im Raum stehen,
    während sich die Wände um sie herum drehen

Darstellung: wie bei der Lavalampe als Metaballs ("Softbälle") – das Feld wird
direkt auf den Panelpixeln ausgewertet, die Flüssigkeit erscheint dort, wo sie
die Würfelwand berührt. Schnell bewegte Flüssigkeit schäumt hell auf.

Ohne laufenden Sensordienst startet ein Demo-Modus mit langsam kippender
Schwerkraft und gelegentlichem Schütteln.

Benötigt gcc (der C-Kern wird beim ersten Start einmalig kompiliert).
"""
import ctypes
import hashlib
import math
import os
import platform
import shutil
import subprocess
import time

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")   # OpenMP-Threads zwischen Frames schlafen lassen

import numpy as np
import cube
import cubelib as cl

try:
    import cubesensor
except ImportError:
    cubesensor = None

# ====================================================================== Einstellungen
# ---------------------------------------------------------------- Flüssigkeit
n_particles = 700               # Anzahl Flüssigkeitspakete (weniger = schneller)
fill = 0.30                     # Füllgrad des Würfels (0..1)
substeps = 5                    # Physikschritte pro Frame (mehr = stabiler bei Schütteln)

gravity = 1200.0                # Beschleunigung bei 1 g in px/s². Echtes Wasser in einem
                                # ~19 cm Würfel entspräche ~3300 – kleiner wirkt träger/zäher
motion_gain = 1.5               # Verstärkung der Bewegungsbeschleunigung (Schwappen)
rotation_gain = 1.0             # 1 = Flüssigkeit bleibt bei Drehung im Raum stehen, 0 = aus
gyro_deadband = 3.0             # °/s, kleinere Drehraten ignorieren (Sensorrauschen)
max_accel_g = 4.0               # Begrenzung der Antriebsbeschleunigung (in g)

stiffness = 0.4                 # Druck: Volumenerhaltung (höher = weniger kompressibel)
near_stiffness = 0.6            # Nah-Druck: hält Pakete auseinander, erzeugt Tropfenbildung
viscosity = 0.15                # lineare Viskosität (höher = honigartiger)
viscosity_quad = 0.02           # quadratische Viskosität (dämpft schnelle Stöße)
wall_friction = 0.05            # Reibung an den Wänden (0..1)
wall_stiffness = 0.7            # Gegendruck der Wände (0.5 = weich/klebt eher, 1 = hart/zittert eher)

# ---------------------------------------------------------------- Darstellung
palette = "Ozean"               # "Ozean", "Lava", "Gift", "Magenta" oder eigener Eintrag in PALETTES
blob_radius = 8.0               # Reichweite eines Pakets auf der Wand in px (größer = glatter)
threshold = 0.45                # Feldwert, ab dem ein Pixel als Flüssigkeit gilt
edge_softness = 0.25            # Kantenweichheit (relativ zu threshold)
foam_speed = (1.2, 3.0)         # Geschwindigkeit [px/Schritt] für Beginn/volle Schaumfarbe
depth_fade = 18.0               # px unter der Oberfläche bis zur vollen Tiefenfarbe
edge_highlight = 0.35           # Aufhellung von Wasserlinie und Tropfenrändern (0..1)
glow = 0.25                     # Schein um die Flüssigkeit
shimmer = 0.15                  # wandernde Lichtreflexe in der Flüssigkeit (0 = aus)
brightness = 0.8                # 0..1

# Farben je Palette: Hintergrund, Rand/Oberfläche, Tiefe, Schaum.
# Jede Palette hat einen Kanal, der überall <= 50 bleibt (gesättigte LED-Farben).
PALETTES = {
    "Ozean":   dict(bg=(0, 2, 10),  rim=(0, 140, 255), deep=(0, 25, 140), foam=(0, 255, 220)),
    "Lava":    dict(bg=(10, 1, 0),  rim=(255, 70, 0),  deep=(150, 5, 0),  foam=(255, 200, 0)),
    "Gift":    dict(bg=(1, 8, 0),   rim=(70, 255, 0),  deep=(0, 100, 15), foam=(210, 255, 0)),
    "Magenta": dict(bg=(8, 0, 10),  rim=(255, 0, 190), deep=(90, 0, 130), foam=(255, 40, 255)),
}

# ---------------------------------------------------------------- Demo ohne Sensor
demo_tilt_deg = 70              # maximale Neigung der Schwerkraft im Demo-Modus
demo_shake_every = 9.0          # s zwischen zwei Schüttel-Stößen im Demo-Modus

target_fps = 30                 # Bildrate (die Physik ist darauf abgestimmt)

UDP_IP = '192.168.178.50'
UDP_PORT = 2000

N = cl.N
S = float(N)

# ====================================================================== Würfelgeometrie
# Einbettung der Flächen in 3D: P = O + x * ex + y * ey (passend zu cubelib)
FACE_O = np.array([[64, 0, 64], [0, 0, 64], [0, 64, 64],
                   [64, 64, 64], [0, 64, 64], [0, 0, 0]], dtype=np.float64)
FACE_EX = np.array([[0, 1, 0], [1, 0, 0], [0, -1, 0],
                    [-1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=np.float64)
FACE_EY = np.array([[0, 0, -1], [0, 0, -1], [0, 0, -1],
                    [0, 0, -1], [0, -1, 0], [0, 1, 0]], dtype=np.float64)
CENTER = np.full(3, S / 2)

# ====================================================================== C-Kern
_C_SRC = r"""
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ---------- Nachbarliste: feste Plätze je Partikel ----------
   Zellgröße h/2, Partikel per Counting Sort nach Zellen geordnet; die 5 Zellen in
   z-Richtung liegen dadurch zusammenhängend im Speicher (25 Bereiche statt 125 Listen). */
#define MAXN 96
typedef struct { int64_t *cnt, *idx; } nlist_t;
typedef struct { int64_t *start, *order, *cid; double *Xs; } grid_t;

static void build_list(int64_t n, const double *X, double h, int64_t nc, grid_t *G, nlist_t *L)
{
    const double h2 = h * h, inv_cs = 2.0 / h;
    const int64_t ncell = nc * nc * nc;
    for (int64_t c = 0; c <= ncell; c++) G->start[c] = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t q[3];
        for (int k = 0; k < 3; k++) {
            double v = X[3*i+k] * inv_cs;
            if (!(v >= 0.0)) v = 0.0;
            if (v > (double)(nc - 1)) v = (double)(nc - 1);
            q[k] = (int64_t)v;
        }
        G->cid[i] = (q[0] * nc + q[1]) * nc + q[2];
        G->start[G->cid[i] + 1]++;
    }
    for (int64_t c = 0; c < ncell; c++) G->start[c + 1] += G->start[c];
    for (int64_t i = 0; i < n; i++) {                 /* stabil einsortieren */
        int64_t p = G->start[G->cid[i]]++;
        G->order[p] = i;
    }
    for (int64_t c = ncell; c > 0; c--) G->start[c] = G->start[c - 1];
    G->start[0] = 0;
    for (int64_t p = 0; p < n; p++) {
        int64_t i = G->order[p];
        G->Xs[3*p] = X[3*i]; G->Xs[3*p+1] = X[3*i+1]; G->Xs[3*p+2] = X[3*i+2];
    }

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i++) {
        int64_t c = 0, *out = L->idx + i * MAXN;
        const double xi = X[3*i], yi = X[3*i+1], zi = X[3*i+2];
        const int64_t ci = G->cid[i];
        const int64_t cx = ci / (nc * nc), cy = (ci / nc) % nc, cz = ci % nc;
        const int64_t z0 = cz > 1 ? cz - 2 : 0, z1 = cz + 2 < nc ? cz + 2 : nc - 1;
        for (int64_t ax = (cx > 1 ? cx - 2 : 0); ax <= (cx + 2 < nc ? cx + 2 : nc - 1); ax++)
        for (int64_t ay = (cy > 1 ? cy - 2 : 0); ay <= (cy + 2 < nc ? cy + 2 : nc - 1); ay++) {
            int64_t base = (ax * nc + ay) * nc;
            for (int64_t p = G->start[base + z0]; p < G->start[base + z1 + 1]; p++) {
                double dx = G->Xs[3*p] - xi, dy = G->Xs[3*p+1] - yi, dz = G->Xs[3*p+2] - zi;
                if (dx*dx + dy*dy + dz*dz < h2) {
                    int64_t j = G->order[p];
                    if (j != i && c < MAXN) out[c++] = j;
                }
            }
        }
        L->cnt[i] = c;
    }
}

/* Wandtabellen: wt[0..3][m] für Abstand d = 0..h (Dichte, Nah-Dichte, Druck-, Nahdruck-Schub)
   = Beitrag gedachter Flüssigkeit hinter der Wand (Halbraum-Integrale, in Python berechnet) */
static inline double wall_lookup(const double *t, int64_t m, double d, double h)
{
    if (d >= h) return 0.0;
    if (d < 0.0) d = 0.0;
    double u = d / h * (double)(m - 1);
    int64_t a = (int64_t)u;
    if (a >= m - 1) return t[m - 1];
    double f = u - (double)a;
    return t[a] * (1.0 - f) + t[a + 1] * f;
}

int fl_step(int64_t n, double *X, double *V, const double *g, int64_t nsub,
            double h, double rho0, double k, double knear, double sigma, double beta,
            double lo, double hi, double fric, double vmax, const double *wt, int64_t m,
            double wk)
{
    const int64_t nc = (int64_t)(64.0 / (0.5 * h)) + 1;
    const int64_t nn = n > 0 ? n : 1;
    double  *Xp   = malloc(sizeof(double) * 3 * nn);
    nlist_t L = { malloc(sizeof(int64_t) * nn), malloc(sizeof(int64_t) * MAXN * nn) };
    grid_t G = { malloc(sizeof(int64_t) * (nc * nc * nc + 1)), malloc(sizeof(int64_t) * nn),
                 malloc(sizeof(int64_t) * nn), malloc(sizeof(double) * 3 * nn) };
    int rc = 0;
    if (!Xp || !L.cnt || !L.idx || !G.start || !G.order || !G.cid || !G.Xs) { rc = -1; goto done; }
    const double h2 = h * h, inv_h = 1.0 / h;

    build_list(n, X, h, nc, &G, &L);

    for (int64_t s = 0; s < nsub; s++) {
        /* 1) äußere Kraft */
        for (int64_t i = 0; i < n; i++)
            for (int c = 0; c < 3; c++) V[3 * i + c] += g[c];

        /* 2) Viskosität (Impulse zwischen sich annähernden Paketen), jedes Paar einmal */
        for (int64_t i = 0; i < n; i++) {
            for (int64_t e = i * MAXN; e < i * MAXN + L.cnt[i]; e++) {
                int64_t j = L.idx[e];
                if (j <= i) continue;
                double dx = X[3*j] - X[3*i], dy = X[3*j+1] - X[3*i+1], dz = X[3*j+2] - X[3*i+2];
                double r2 = dx*dx + dy*dy + dz*dz;
                if (r2 >= h2 || r2 < 1e-12) continue;
                double r = sqrt(r2), q = r * inv_h;
                dx /= r; dy /= r; dz /= r;
                double u = (V[3*i] - V[3*j]) * dx + (V[3*i+1] - V[3*j+1]) * dy + (V[3*i+2] - V[3*j+2]) * dz;
                if (u > 0.0) {
                    double I = 0.5 * (1.0 - q) * (sigma * u + beta * u * u);
                    V[3*i] -= I * dx; V[3*i+1] -= I * dy; V[3*i+2] -= I * dz;
                    V[3*j] += I * dx; V[3*j+1] += I * dy; V[3*j+2] += I * dz;
                }
            }
        }

        /* 3) Vorhersage der Positionen */
        for (int64_t i = 0; i < n; i++) {
            double vx = V[3*i], vy = V[3*i+1], vz = V[3*i+2];
            double sp = sqrt(vx*vx + vy*vy + vz*vz);
            if (sp > vmax) { double f = vmax / sp; vx *= f; vy *= f; vz *= f; }
            for (int c = 0; c < 3; c++) Xp[3*i+c] = X[3*i+c];
            X[3*i] += vx; X[3*i+1] += vy; X[3*i+2] += vz;
        }

        /* 4) Double Density Relaxation (Nachbarliste an den neuen Positionen) */
        build_list(n, X, h, nc, &G, &L);
        for (int64_t i = 0; i < n; i++) {
            double rho = 0.0, rhon = 0.0;
            for (int64_t e = i * MAXN; e < i * MAXN + L.cnt[i]; e++) {
                int64_t j = L.idx[e];
                double dx = X[3*j] - X[3*i], dy = X[3*j+1] - X[3*i+1], dz = X[3*j+2] - X[3*i+2];
                double r2 = dx*dx + dy*dy + dz*dz;
                if (r2 >= h2) continue;
                double w = 1.0 - sqrt(r2) * inv_h;
                rho += w * w; rhon += w * w * w;
            }
            /* Wände: Abstand zu den 6 Ebenen x,y,z = 0 bzw. 64 */
            double dw[6];
            for (int c = 0; c < 3; c++) { dw[2*c] = X[3*i+c]; dw[2*c+1] = 64.0 - X[3*i+c]; }
            for (int q = 0; q < 6; q++) {
                if (dw[q] >= h) continue;
                rho  += wall_lookup(wt,         m, dw[q], h);
                rhon += wall_lookup(wt + m,     m, dw[q], h);
            }
            double P = k * (rho - rho0), Pn = knear * rhon;
            double ddx = 0.0, ddy = 0.0, ddz = 0.0;
            for (int q = 0; q < 6; q++) {
                if (dw[q] >= h) continue;
                double push = wk * (P * wall_lookup(wt + 2 * m, m, dw[q], h)
                                   + Pn * wall_lookup(wt + 3 * m, m, dw[q], h));
                double sgn = (q % 2 == 0) ? 1.0 : -1.0;      /* weg von der Wand */
                if (q / 2 == 0) ddx += sgn * push; else if (q / 2 == 1) ddy += sgn * push; else ddz += sgn * push;
            }
            for (int64_t e = i * MAXN; e < i * MAXN + L.cnt[i]; e++) {
                int64_t j = L.idx[e];
                double dx = X[3*j] - X[3*i], dy = X[3*j+1] - X[3*i+1], dz = X[3*j+2] - X[3*i+2];
                double r2 = dx*dx + dy*dy + dz*dz;
                if (r2 >= h2) continue;
                double r = sqrt(r2);
                if (r < 1e-9) { dx = 1e-3 * (double)((i * 7 + j * 13) % 5 - 2); dy = 1e-3; dz = 0.0; r = sqrt(dx*dx + dy*dy); }
                double w = 1.0 - r * inv_h;
                double D = 0.5 * (P * w + Pn * w * w) / r;
                X[3*j] += D * dx; X[3*j+1] += D * dy; X[3*j+2] += D * dz;
                ddx -= D * dx; ddy -= D * dy; ddz -= D * dz;
            }
            X[3*i] += ddx; X[3*i+1] += ddy; X[3*i+2] += ddz;
        }

        /* 5) Wände und neue Geschwindigkeit */
        for (int64_t i = 0; i < n; i++) {
            int hit = 0;
            for (int c = 0; c < 3; c++) {
                double x = X[3*i+c];
                if (!(x == x)) x = 32.0;                  /* NaN-Schutz */
                if (x < lo) { x = lo; hit = 1; }
                if (x > hi) { x = hi; hit = 1; }
                X[3*i+c] = x;
            }
            for (int c = 0; c < 3; c++) V[3*i+c] = X[3*i+c] - Xp[3*i+c];
            if (hit && fric > 0.0)
                for (int c = 0; c < 3; c++) {
                    double x = X[3*i+c];
                    if (x > lo && x < hi) V[3*i+c] *= (1.0 - fric);   /* tangential bremsen */
                }
        }
    }
done:
    free(Xp); free(L.cnt); free(L.idx); free(G.start); free(G.order); free(G.cid); free(G.Xs);
    return rc;
}

/* ---------- Metaball-Feld direkt auf den Panelpixeln ----------
   field/speed/depth: 6*64*64, werden hier genullt.
   Für jedes Paket und jede Fläche nur das Fenster der Pixel innerhalb R. */
void fl_render(int64_t n, const double *X, const double *V, double R,
               const double *FO, const double *FEX, const double *FEY,
               float *field, float *speed, float *depth)
{
    memset(field, 0, sizeof(float) * 6 * 64 * 64);
    memset(speed, 0, sizeof(float) * 6 * 64 * 64);
    memset(depth, 0, sizeof(float) * 6 * 64 * 64);
    const double R2 = R * R, invR2 = 1.0 / R2;
    for (int f = 0; f < 6; f++) {
        const double *O = FO + 3 * f, *ex = FEX + 3 * f, *ey = FEY + 3 * f;
        double nx = ex[1] * ey[2] - ex[2] * ey[1];
        double ny = ex[2] * ey[0] - ex[0] * ey[2];
        double nz = ex[0] * ey[1] - ex[1] * ey[0];
        for (int64_t i = 0; i < n; i++) {
            double px = X[3*i] - O[0], py = X[3*i+1] - O[1], pz = X[3*i+2] - O[2];
            double d = fabs(px * nx + py * ny + pz * nz);
            if (d >= R) continue;
            double u = px * ex[0] + py * ex[1] + pz * ex[2];
            double v = px * ey[0] + py * ey[1] + pz * ey[2];
            double s = sqrt(R2 - d * d);
            int64_t x0 = (int64_t)floor(u - 0.5 - s), x1 = (int64_t)ceil(u - 0.5 + s);
            int64_t y0 = (int64_t)floor(v - 0.5 - s), y1 = (int64_t)ceil(v - 0.5 + s);
            if (x0 < 0) x0 = 0;
            if (y0 < 0) y0 = 0;
            if (x1 > 63) x1 = 63;
            if (y1 > 63) y1 = 63;
            if (x0 > x1 || y0 > y1) continue;
            double vs = sqrt(V[3*i]*V[3*i] + V[3*i+1]*V[3*i+1] + V[3*i+2]*V[3*i+2]);
            double base = d * d;
            for (int64_t x = x0; x <= x1; x++) {
                double du = u - (x + 0.5);
                double bx = base + du * du;
                if (bx >= R2) continue;
                float *frow = field + (f * 64 + x) * 64;
                float *srow = speed + (f * 64 + x) * 64;
                float *drow = depth + (f * 64 + x) * 64;
                for (int64_t y = y0; y <= y1; y++) {
                    double dv = v - (y + 0.5);
                    double r2 = bx + dv * dv;
                    if (r2 >= R2) continue;
                    double t = 1.0 - r2 * invR2;
                    float w = (float)(t * t * t);
                    frow[y] += w; srow[y] += w * (float)vs; drow[y] += w * (float)d;
                }
            }
        }
    }
}
"""


def _load_kernel():
    """Kompiliert den C-Kern einmalig (Ergebnis in __pycache__/) und lädt ihn per ctypes."""
    cc = shutil.which("cc") or shutil.which("gcc")
    build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "__pycache__")
    tag = hashlib.sha1((_C_SRC + platform.machine()).encode()).hexdigest()[:12]
    so = os.path.join(build_dir, f"fluid_kernel_{tag}.so")
    if not os.path.exists(so):
        if cc is None:
            raise RuntimeError("kein C-Compiler gefunden – 'sudo apt install gcc'")
        os.makedirs(build_dir, exist_ok=True)
        src = so[:-3] + ".c"
        with open(src, "w") as fh:
            fh.write(_C_SRC)
        tmp = f"{so}.{os.getpid()}.tmp"
        err = ""
        for flags in (["-O3", "-march=native", "-fopenmp"], ["-O3", "-fopenmp"], ["-O3"]):
            res = subprocess.run([cc, *flags, "-shared", "-fPIC", "-o", tmp, src, "-lm"],
                                 capture_output=True, text=True)
            if res.returncode == 0:
                os.replace(tmp, so)
                break
            err = res.stderr
        else:
            raise RuntimeError("C-Kern ließ sich nicht kompilieren:\n" + err)

    lib = ctypes.CDLL(so)
    d = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    f = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
    i64, dbl = ctypes.c_int64, ctypes.c_double
    lib.fl_step.argtypes = [i64, d, d, d, i64, dbl, dbl, dbl, dbl, dbl, dbl, dbl, dbl, dbl, dbl, d, i64, dbl]
    lib.fl_step.restype = ctypes.c_int
    lib.fl_render.argtypes = [i64, d, d, dbl, d, d, d, f, f, f]
    lib.fl_render.restype = None
    return lib


# ====================================================================== Flüssigkeit
def rest_density(h, spacing):
    """Ruhedichte eines kubischen Gitters mit Abstand `spacing` (Kernel (1 - r/h)²)."""
    m = int(math.ceil(h / spacing))
    r = np.arange(-m, m + 1) * spacing
    X, Y, Z = np.meshgrid(r, r, r, indexing="ij")
    d = np.sqrt(X ** 2 + Y ** 2 + Z ** 2).ravel()
    d = d[(d > 0) & (d < h)]
    return float(((1 - d / h) ** 2).sum())


_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")      # NumPy 1.x (Bookworm) und 2.x


def wall_tables(h, rho0, m=129):
    """Beitrag einer mit Ruhedichte gefüllten Halbraum-"Geisterflüssigkeit" hinter einer Wand
    im Abstand d (0..h): Dichte, Nah-Dichte und Schub der beiden Druckterme senkrecht zur Wand.
    Normiert so, dass der volle Raum genau rho0 ergibt (wie das Gitter im Inneren)."""
    r = np.linspace(0.0, h, 4001)
    w = 1.0 - r / h
    full = _trapz(w ** 2 * 4 * np.pi * r ** 2, r)
    c = rho0 / full                                   # = Teilchendichte, an das Gitter angepasst
    out = np.zeros((4, m))
    for a, d in enumerate(np.linspace(0.0, h, m)):
        sel = r >= d
        rr, ww = r[sel], w[sel]
        if rr.size < 2:
            continue
        cap = 2 * np.pi * (1 - d / np.maximum(rr, 1e-12)) * rr ** 2   # Kugelkappe hinter der Wand
        lat = np.pi * (rr ** 2 - d ** 2)                             # Integral von cos(θ) über die Kappe
        out[0, a] = c * _trapz(ww ** 2 * cap, rr)
        out[1, a] = c * _trapz(ww ** 3 * cap, rr)
        out[2, a] = c * _trapz(ww * lat, rr)
        out[3, a] = c * _trapz(ww ** 2 * lat, rr)
    return np.ascontiguousarray(out)


class Fluid:
    def __init__(self, lib):
        self.lib = lib
        self.spacing = (fill * S ** 3 / n_particles) ** (1 / 3)
        self.h = 2.0 * self.spacing                     # Wechselwirkungsradius
        self.rho0 = rest_density(self.h, self.spacing)
        self.wt = wall_tables(self.h, self.rho0)
        # Startblock am Boden (Fläche 5), leicht verrauscht
        nx = int(round(S / self.spacing)) - 1
        layers = int(math.ceil(n_particles / (nx * nx)))
        g = np.stack(np.meshgrid(np.arange(nx), np.arange(nx), np.arange(layers), indexing="ij"),
                     -1).reshape(-1, 3)[:n_particles]
        X = (g + 0.75) * self.spacing
        X += cl.rng.uniform(-0.1, 0.1, X.shape) * self.spacing
        self.X = np.ascontiguousarray(np.clip(X, 0.5, S - 0.5))
        self.V = np.zeros_like(self.X)
        self.vmax = 0.5 * self.h

    def rotate(self, omega, dt):
        """Würfel dreht sich mit omega (rad/s, Würfelsystem): Flüssigkeit im Würfelsystem
        entgegengesetzt um die Würfelmitte drehen."""
        w = np.asarray(omega, dtype=float)
        ang = np.linalg.norm(w) * dt
        if ang < 1e-9:
            return
        k = w / np.linalg.norm(w)
        c, s = math.cos(-ang), math.sin(-ang)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        Rm = np.eye(3) + s * K + (1 - c) * (K @ K)
        self.X[:] = (self.X - CENTER) @ Rm.T + CENTER
        self.V[:] = self.V @ Rm.T
        np.clip(self.X, 0.5, S - 0.5, out=self.X)

    def step(self, accel_px_s2, fps):
        dt = 1.0 / (fps * substeps)
        g = np.ascontiguousarray(np.asarray(accel_px_s2, dtype=np.float64) * dt * dt)
        res = self.lib.fl_step(self.X.shape[0], self.X, self.V, g, substeps,
                               self.h, self.rho0, stiffness, near_stiffness,
                               viscosity, viscosity_quad, 0.5, S - 0.5, wall_friction, self.vmax,
                               self.wt, self.wt.shape[1], wall_stiffness)
        if res != 0:
            raise MemoryError("Fluid-Kern: Speicher konnte nicht reserviert werden")


# ====================================================================== Antrieb
class Drive:
    """Liefert pro Frame (Beschleunigung in px/s², Drehrate in rad/s) – vom Sensor
    oder im Demo-Modus synthetisch."""

    def __init__(self):
        self.t0 = time.monotonic()
        self.mode = None

    def get(self):
        s = cubesensor.read() if cubesensor is not None else None
        if s is not None:
            self._announce("Sensor")
            up = np.array(s["up"], dtype=float)
            lin = np.array(s["lin_acc"], dtype=float)
            a = -(up + motion_gain * lin)
            gyro = np.array(s["gyro"], dtype=float)
            if np.linalg.norm(gyro) < gyro_deadband:
                gyro[:] = 0.0
            omega = np.radians(gyro) * rotation_gain
        else:
            self._announce("Demo")
            t = time.monotonic() - self.t0
            tilt = math.radians(demo_tilt_deg) * math.sin(0.23 * t) * math.sin(0.07 * t + 1.0)
            az = 0.31 * t
            up = np.array([math.sin(tilt) * math.cos(az), math.sin(tilt) * math.sin(az), math.cos(tilt)])
            a = -up
            ph = t % demo_shake_every
            if ph < 0.6:                                   # kurzer seitlicher Stoß
                a += 1.8 * math.sin(2 * math.pi * ph / 0.3) * np.array([math.cos(az), math.sin(az), 0.0])
            omega = np.zeros(3)
        n = np.linalg.norm(a)
        if n > max_accel_g:
            a *= max_accel_g / n
        return a * gravity, omega

    def _announce(self, mode):
        if mode != self.mode:
            self.mode = mode
            print("Antrieb:", "Lagesensor" if mode == "Sensor" else
                  "Demo-Modus (kein Sensordienst aktiv – 'cube@cubesensor' starten)", flush=True)


# ====================================================================== Darstellung
def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


class Renderer:
    def __init__(self, lib):
        self.lib = lib
        p = PALETTES[palette]
        c = {k: np.array(v, dtype=np.float32)[:, None] / 255.0 for k, v in p.items()}
        self.bg, self.rim, self.deep, self.foam = c["bg"], c["rim"], c["deep"], c["foam"]
        size = cl.FACES * N * N
        self.field = np.zeros(size, dtype=np.float32)
        self.speed = np.zeros(size, dtype=np.float32)
        self.depth = np.zeros(size, dtype=np.float32)
        self.frame = np.zeros((cl.FACES, N, N, 3), dtype=np.uint8)
        f, x, y = np.meshgrid(np.arange(cl.FACES), np.arange(N) + 0.5, np.arange(N) + 0.5, indexing="ij")
        pix = FACE_O[f] + x[..., None] * FACE_EX[f] + y[..., None] * FACE_EY[f]
        self.pix = np.ascontiguousarray((pix.reshape(-1, 3) - CENTER).T, dtype=np.float32)  # (3, Pixel)
        # großflächiges, langsam wanderndes Lichtmuster (Wellenlängen ~30–50 px)
        self.px = (x.ravel() * 0.15 + y.ravel() * 0.09 + f.ravel() * 1.7).astype(np.float32)
        self.py = (x.ravel() * 0.07 - y.ravel() * 0.16 + f.ravel() * 2.9).astype(np.float32)
        self.up = np.array([0.0, 0.0, 1.0])
        self.level = None
        self.t = 0

    def render(self, fluid, accel):
        """accel: aktuelle Antriebsbeschleunigung (zeigt "nach unten")."""
        self.t += 1
        f32 = np.float32
        # geglättete Hochrichtung und Pegel der Flüssigkeit entlang dieser Richtung
        n = np.linalg.norm(accel)
        if n > 1e-6:
            self.up += 0.25 * (-np.asarray(accel) / n - self.up)
            self.up /= np.linalg.norm(self.up)
        hp = (fluid.X - CENTER) @ self.up
        level = float(np.percentile(hp, 97))
        self.level = level if self.level is None else self.level + 0.3 * (level - self.level)

        self.lib.fl_render(fluid.X.shape[0], fluid.X, fluid.V, float(blob_radius),
                           FACE_O, FACE_EX, FACE_EY, self.field, self.speed, self.depth)
        F = self.field
        spd = self.speed / np.maximum(F, f32(1e-6))     # mittlere Geschwindigkeit am Pixel
        th = f32(threshold)
        m = smoothstep(th * f32(1 - edge_softness), th * f32(1 + edge_softness), F)
        foam = smoothstep(f32(foam_speed[0]), f32(foam_speed[1]), spd)

        # Tiefe unter der Oberfläche: oben hell (rim), nach unten dunkler (deep)
        hpix = self.up.astype(np.float32) @ self.pix
        depth = np.clip((f32(self.level) - hpix) / f32(depth_fade), f32(0), f32(1))
        liquid = self.rim + (self.deep - self.rim) * depth
        # Wasserlinie / Tropfenränder: knapp über der Schwelle aufhellen
        edge = f32(1) - smoothstep(th, th * f32(2.2), F)
        liquid = liquid + (self.foam - liquid) * (f32(edge_highlight) * edge)
        if shimmer:
            t = f32(self.t)
            liquid = liquid * (f32(1.0) + f32(shimmer) * np.sin(self.px + f32(0.09) * t)
                               * np.sin(self.py - f32(0.07) * t))
        liquid = liquid + (self.foam - liquid) * foam
        halo = f32(glow) * smoothstep(th * f32(0.3), th, F) * (f32(1) - m)
        col = self.bg + (liquid - self.bg) * m + halo * self.rim
        col *= f32(255 * brightness)
        np.clip(col, 0, 255, out=col)
        self.frame.reshape(-1, 3)[:] = col.T
        return self.frame


# ====================================================================== Hauptschleife
def main():
    if palette not in PALETTES:
        raise ValueError(f"Unbekannte Palette '{palette}'. Verfügbar: {', '.join(PALETTES)}")
    lib = _load_kernel()
    c = cube.Cube(UDP_IP, UDP_PORT)
    limiter = cl.FrameLimiter(target_fps)
    fluid = Fluid(lib)
    drive = Drive()
    renderer = Renderer(lib)
    fps = float(target_fps or 30)
    print(f"{n_particles} Pakete, Abstand {fluid.spacing:.2f} px, h = {fluid.h:.2f} px, "
          f"Palette {palette}", flush=True)

    normals = {0: (1, 0, 0), 1: (0, -1, 0), 2: (-1, 0, 0), 3: (0, 1, 0), 4: (0, 0, 1), 5: (0, 0, -1)}
    next_status = time.monotonic() + 2.0
    while True:
        accel, omega = drive.get()
        if time.monotonic() > next_status:          # Diagnose: woher kommt die Schwerkraft, wohin zeigt sie?
            next_status = time.monotonic() + 5.0
            down = max(normals, key=lambda f: float(np.dot(normals[f], accel)))
            print(f"Antrieb: {'Sensor' if drive.mode == 'Sensor' else 'DEMO (kein Sensor)'} | "
                  f"Schwerkraft zeigt zu Fläche {down}", flush=True)
        if rotation_gain:
            fluid.rotate(omega, 1.0 / fps)
        fluid.step(accel, fps)
        c.sendFrame(renderer.render(fluid, accel))
        limiter.tick()


if __name__ == "__main__":
    main()
