#!/usr/bin/env python3
"""Particle Life auf der Würfeloberfläche.

Mehrere Partikelspezies ziehen sich gegenseitig an oder stoßen sich ab –
festgelegt durch eine (asymmetrische) Kopplungsmatrix. Daraus entstehen
Zellen, Ketten, Jäger-Beute-Schwärme und rotierende Strukturen. Die
Kopplungsmatrix verändert sich zufällig über die Zeit, sodass sich die
"Lebensformen" laufend umbilden.

Kraftgesetz (r = Abstand / r_max):
    r < beta        : repel_force * (r / beta - 1)              (Abstoßung, < 0)
    beta <= r < 1   : K[a, b] * (1 - |2r - 1 - beta| / (1 - beta))  (Dreiecksprofil)
    r >= 1          : 0

Die Partikel werden für die Abstandsberechnung auf einen 3D-Würfel
abgebildet, dadurch wirken Kräfte auch über Kanten und Ecken hinweg.
"""
import importlib.util
import json
import os
import platform
import subprocess
import sys

# OpenMP-Threads zwischen den Frames schlafen lassen statt aktiv zu warten
# (spart auf dem Pi CPU-Last und Wärme; muss vor dem Laden des C-Kerns gesetzt sein)
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import numpy as np
import cube
import cubelib as cl

# ====================================================================== Einstellungen
# ---------------------------------------------------------------- Partikel
n_species = 6                   # Anzahl Spezies (1 .. len(species_colors) bzw. beliebig)
n_particles = 1600              # Gesamtzahl Partikel
species_counts = None           # optional Liste je Spezies, z. B. [400, 300, 300, 200, 200, 100]

# ---------------------------------------------------------------- Physik
r_max = 14.0                    # Wechselwirkungsradius in Pixeln
beta = 0.3                      # Anteil von r_max mit Abstoßung (Kernradius)
repel_force = 3.0               # Stärke der Kernabstoßung (höher = weniger Verklumpung)
force_multiplier = 0.01         # globaler Kraftfaktor (Pixel/Frame² pro Krafteinheit)
friction = 0.85                 # Geschwindigkeit, die pro Frame erhalten bleibt (0..1)
max_speed = 1.5                 # Pixel pro Frame (Stabilitätsgrenze)

# ---------------------------------------------------------------- Kopplungsmatrix
# K[a, b]: wie stark Spezies a von Spezies b angezogen (+) bzw. abgestoßen (-) wird.
# None = zufällig, sonst Liste von Listen (n_species x n_species) mit Werten -1..1.
coupling_matrix = None
matrix_period = 700             # alle n Frames ein neues Ziel für die Matrix
matrix_blend = 300              # Überblenddauer zum neuen Ziel (Frames)
matrix_change_fraction = 0.4    # Anteil der Einträge, die sich pro Wechsel ändern
matrix_drift = 0.002            # zusätzliches leichtes Zufallsrauschen pro Frame

# ---------------------------------------------------------------- Farben
# Farbmatrix: eine RGB-Zeile pro Spezies. Bei mehr Spezies werden Farben ergänzt.
species_colors = [
    (255, 95, 80),     # Koralle
    (255, 185, 40),    # Bernstein
    (140, 230, 70),    # Limette
    (30, 210, 175),    # Türkis
    (60, 140, 255),    # Himmelblau
    (165, 95, 255),    # Violett
    (245, 70, 185),    # Magenta
    (240, 235, 215),   # Elfenbein
]
trail_fade = 0.55               # Nachleuchten pro Frame (0 = keins)
halo = 0.15                     # weicher Schein um Partikel (0 = aus)
particle_gain = 0.9             # Helligkeitsbeitrag eines Partikels
brightness = 0.8                # 0..1, globale Helligkeit

target_fps = 30                 # None = so schnell wie möglich

# ---------------------------------------------------------------- Rechenkern
# "auto"     : beim ersten Start wird in einem Unterprozess getestet, welcher Kern auf
#              diesem System stabil läuft (C -> Numba parallel -> Numba seriell -> NumPy).
#              Das Ergebnis steht in .particleLife_kernel.json (löschen = neu testen).
# "c"        : kleiner C-Kern, einmalig mit gcc kompiliert, OpenMP auf allen CPU-Kernen
# "parallel" : Numba auf allen CPU-Kernen       "serial": Numba auf einem CPU-Kern
# "numpy"    : ohne Compiler (langsam)
kernel_mode = "auto"

UDP_IP = '192.168.178.50'
UDP_PORT = 2000

rng = cl.rng
N = cl.N

# ====================================================================== Würfelgeometrie
# Einbettung jeder Fläche in 3D: P = O + x * ex + y * ey  (passend zu cubelib.resolve)
FACE_O = np.array([[64, 0, 64], [0, 0, 64], [0, 64, 64],
                   [64, 64, 64], [0, 64, 64], [0, 0, 0]], dtype=float)
FACE_EX = np.array([[0, 1, 0], [1, 0, 0], [0, -1, 0],
                    [-1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float)
FACE_EY = np.array([[0, 0, -1], [0, 0, -1], [0, 0, -1],
                    [0, 0, -1], [0, -1, 0], [0, 1, 0]], dtype=float)


def to_3d(face, x, y):
    return FACE_O[face] + x[:, None] * FACE_EX[face] + y[:, None] * FACE_EY[face]


# ====================================================================== Kraftkern
def _forces_numpy(P, species, K, r_max, beta, repel_force):
    """Kräfte über k-d-Baum und vektorisierte Paarliste (ohne Compiler, langsam)."""
    from scipy.spatial import cKDTree
    n = P.shape[0]
    pairs = cKDTree(P).query_pairs(r_max, output_type='ndarray')
    if pairs.size == 0:
        return np.zeros_like(P)
    i, j = pairs[:, 0], pairs[:, 1]
    d = P[j] - P[i]                                   # zeigt von i nach j
    dist = np.maximum(np.sqrt(np.einsum('ij,ij->i', d, d)), 1e-6)
    r = dist / r_max

    # Geometrie einmal pro Paar, Kopplung in beide Richtungen (Matrix ist asymmetrisch)
    core = r < beta
    tri = 1.0 - np.abs(2.0 * r - 1.0 - beta) / (1.0 - beta)
    rep = repel_force * (r / beta - 1.0)
    si, sj = species[i], species[j]
    f_ij = np.where(core, rep, K[si, sj] * tri)        # Kraft auf i, Richtung j
    f_ji = np.where(core, rep, K[sj, si] * tri)        # Kraft auf j, Richtung i

    coef = np.concatenate([f_ij, -f_ji]) / np.concatenate([dist, dist])
    vec = np.concatenate([d, d]) * coef[:, None]
    idx = (np.concatenate([i, j])[:, None] * 3 + np.arange(3)).ravel()
    return np.bincount(idx, weights=vec.ravel(), minlength=3 * n).reshape(n, 3)


# ---------------------------------------------------------------- C-Kern (gcc + OpenMP)
_C_SRC = r"""
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

int pl_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/* Kraft auf jedes Partikel, Zellenliste mit Zellgröße r_max.
   P: n x 3 (Positionen), species: n, K: ns x ns (zeilenweise), F: n x 3 (Ausgabe) */
int pl_forces(int64_t n, const double *P, const int64_t *species, int64_t ns,
              const double *K, double r_max, double beta, double repel, double *F)
{
    const int64_t nc = (int64_t)(64.0 / r_max) + 1;
    const int64_t ncell = nc * nc * nc;
    int64_t *head = (int64_t *)malloc(sizeof(int64_t) * ncell);
    int64_t *nxt  = (int64_t *)malloc(sizeof(int64_t) * (n > 0 ? n : 1));
    int64_t *cell = (int64_t *)malloc(sizeof(int64_t) * 3 * (n > 0 ? n : 1));
    if (!head || !nxt || !cell) { free(head); free(nxt); free(cell); return -1; }

    for (int64_t c = 0; c < ncell; c++) head[c] = -1;
    for (int64_t i = 0; i < n; i++) {
        for (int k = 0; k < 3; k++) {
            double v = P[3 * i + k] / r_max;
            if (!(v >= 0.0)) v = 0.0;                 /* auch NaN abfangen */
            if (v > (double)(nc - 1)) v = (double)(nc - 1);
            cell[3 * i + k] = (int64_t)v;
        }
        int64_t ci = (cell[3 * i] * nc + cell[3 * i + 1]) * nc + cell[3 * i + 2];
        nxt[i] = head[ci];
        head[ci] = i;
    }

    const double r2max = r_max * r_max;
    const double inv_1mb = 1.0 / (1.0 - beta);

    #pragma omp parallel for schedule(dynamic, 32)
    for (int64_t i = 0; i < n; i++) {
        const int64_t si = species[i];
        const double px = P[3 * i], py = P[3 * i + 1], pz = P[3 * i + 2];
        const int64_t cx = cell[3 * i], cy = cell[3 * i + 1], cz = cell[3 * i + 2];
        double fx = 0.0, fy = 0.0, fz = 0.0;
        for (int64_t ax = (cx > 0 ? cx - 1 : 0); ax <= (cx + 1 < nc ? cx + 1 : nc - 1); ax++)
        for (int64_t ay = (cy > 0 ? cy - 1 : 0); ay <= (cy + 1 < nc ? cy + 1 : nc - 1); ay++)
        for (int64_t az = (cz > 0 ? cz - 1 : 0); az <= (cz + 1 < nc ? cz + 1 : nc - 1); az++) {
            for (int64_t j = head[(ax * nc + ay) * nc + az]; j != -1; j = nxt[j]) {
                if (j == i) continue;
                const double dx = P[3 * j] - px, dy = P[3 * j + 1] - py, dz = P[3 * j + 2] - pz;
                const double d2 = dx * dx + dy * dy + dz * dz;
                if (d2 >= r2max) continue;
                double dist = sqrt(d2);
                if (dist < 1e-6) dist = 1e-6;
                const double r = dist / r_max;
                double f;
                if (r < beta) f = repel * (r / beta - 1.0);
                else          f = K[si * ns + species[j]] * (1.0 - fabs(2.0 * r - 1.0 - beta) * inv_1mb);
                f /= dist;
                fx += f * dx; fy += f * dy; fz += f * dz;
            }
        }
        F[3 * i] = fx; F[3 * i + 1] = fy; F[3 * i + 2] = fz;
    }
    free(head); free(nxt); free(cell);
    return 0;
}
"""

C_THREADS = None


def _load_c_kernel():
    """Kompiliert den C-Kern einmalig (Ergebnis in __pycache__/) und lädt ihn per ctypes."""
    global C_THREADS
    import ctypes
    import hashlib
    import shutil
    cc = shutil.which("cc") or shutil.which("gcc")
    build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "__pycache__")
    tag = hashlib.sha1((_C_SRC + platform.machine()).encode()).hexdigest()[:12]
    so = os.path.join(build_dir, f"particleLife_kernel_{tag}.so")
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
    dptr = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    iptr = np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS")
    fn = lib.pl_forces
    fn.argtypes = [ctypes.c_int64, dptr, iptr, ctypes.c_int64, dptr,
                   ctypes.c_double, ctypes.c_double, ctypes.c_double, dptr]
    fn.restype = ctypes.c_int
    lib.pl_threads.restype = ctypes.c_int
    C_THREADS = lib.pl_threads()

    def forces_c(P, species, K, r_max, beta, repel_force):
        P = np.ascontiguousarray(P, dtype=np.float64)
        species = np.ascontiguousarray(species, dtype=np.int64)
        K = np.ascontiguousarray(K, dtype=np.float64)
        F = np.empty_like(P)
        if fn(P.shape[0], P, species, K.shape[0], K, r_max, beta, repel_force, F) != 0:
            raise MemoryError("C-Kern: Speicher konnte nicht reserviert werden")
        return F

    return forces_c


# ---------------------------------------------------------------- Numba-Kerne (optional)
def _load_numba_kernels():
    """Numba wird nur importiert, wenn ein Numba-Kern gewählt ist."""
    from numba import njit, prange

    @njit(cache=True)
    def _build_cells(P, r_max):
        n = P.shape[0]
        nc = int(64.0 / r_max) + 1
        head = np.full(nc * nc * nc, -1, dtype=np.int64)
        nxt = np.empty(n, dtype=np.int64)
        cell = np.empty((n, 3), dtype=np.int64)
        for i in range(n):
            for k in range(3):
                c = int(P[i, k] / r_max)
                cell[i, k] = min(max(c, 0), nc - 1)
            ci = (cell[i, 0] * nc + cell[i, 1]) * nc + cell[i, 2]
            nxt[i] = head[ci]
            head[ci] = i
        return head, nxt, cell, nc

    @njit(cache=True)
    def _force_on(i, P, species, K, r_max, beta, repel_force, head, nxt, cell, nc):
        r2max = r_max * r_max
        inv_1mb = 1.0 / (1.0 - beta)
        si = species[i]
        fx = 0.0
        fy = 0.0
        fz = 0.0
        for ax in range(max(cell[i, 0] - 1, 0), min(cell[i, 0] + 2, nc)):
            for ay in range(max(cell[i, 1] - 1, 0), min(cell[i, 1] + 2, nc)):
                for az in range(max(cell[i, 2] - 1, 0), min(cell[i, 2] + 2, nc)):
                    j = head[(ax * nc + ay) * nc + az]
                    while j != -1:
                        if j != i:
                            dx = P[j, 0] - P[i, 0]
                            dy = P[j, 1] - P[i, 1]
                            dz = P[j, 2] - P[i, 2]
                            d2 = dx * dx + dy * dy + dz * dz
                            if d2 < r2max:
                                dist = max(np.sqrt(d2), 1e-6)
                                r = dist / r_max
                                if r < beta:
                                    f = repel_force * (r / beta - 1.0)
                                else:
                                    f = K[si, species[j]] * (1.0 - abs(2.0 * r - 1.0 - beta) * inv_1mb)
                                f /= dist
                                fx += f * dx
                                fy += f * dy
                                fz += f * dz
                        j = nxt[j]
        return fx, fy, fz

    @njit(cache=True)
    def _forces_serial(P, species, K, r_max, beta, repel_force):
        head, nxt, cell, nc = _build_cells(P, r_max)
        n = P.shape[0]
        F = np.empty((n, 3))
        for i in range(n):
            fx, fy, fz = _force_on(i, P, species, K, r_max, beta, repel_force, head, nxt, cell, nc)
            F[i, 0] = fx
            F[i, 1] = fy
            F[i, 2] = fz
        return F

    @njit(cache=True, parallel=True)
    def _forces_parallel(P, species, K, r_max, beta, repel_force):
        head, nxt, cell, nc = _build_cells(P, r_max)
        n = P.shape[0]
        F = np.empty((n, 3))
        for i in prange(n):
            fx, fy, fz = _force_on(i, P, species, K, r_max, beta, repel_force, head, nxt, cell, nc)
            F[i, 0] = fx
            F[i, 1] = fy
            F[i, 2] = fz
        return F

    return {"parallel": _forces_parallel, "serial": _forces_serial}


# ---------------------------------------------------------------- Auswahl
KERNEL_LABEL = {"c": "C/OpenMP", "parallel": "Numba, parallel", "serial": "Numba, seriell",
                "numpy": "NumPy (langsam)"}
KERNEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".particleLife_kernel.json")
compute_forces = _forces_numpy        # wird in main() durch den gewählten Kern ersetzt


def get_kernel(mode):
    if mode == "c":
        return _load_c_kernel()
    if mode in ("parallel", "serial"):
        return _load_numba_kernels()[mode]
    return _forces_numpy


def _numba_version():
    if importlib.util.find_spec("numba") is None:
        return None
    try:
        from importlib.metadata import version
        return version("numba")
    except Exception:
        return "?"


def _system_key():
    """Kennung von System und Problemgröße – ändert sich etwas davon, wird neu getestet."""
    return (f"v2, python {platform.python_version()}, {platform.machine()}, "
            f"numba {_numba_version()}, n={n_particles}, r_max={r_max}")


def _kernel_selftest(mode):
    """Läuft im Unterprozess: echte Simulationsschritte mit `mode`, Ergebnis gegen NumPy prüfen."""
    global compute_forces
    compute_forces = get_kernel(mode)
    parts = Particles()
    coupling = Coupling(n_species, verbose=False)
    for _ in range(100):
        parts.step(coupling.step())
    P = to_3d(parts.face, parts.x, parts.y)
    args = (P, parts.species, np.ascontiguousarray(coupling.K), float(r_max), float(beta), float(repel_force))
    return np.allclose(compute_forces(*args), _forces_numpy(*args), rtol=1e-6, atol=1e-9)


def choose_kernel():
    if kernel_mode in KERNEL_LABEL and kernel_mode != "auto":
        return kernel_mode

    key = _system_key()
    try:
        with open(KERNEL_FILE) as fh:
            saved = json.load(fh)
        if saved.get("key") == key and saved.get("mode") in KERNEL_LABEL:
            return saved["mode"]
    except (OSError, ValueError):
        pass

    candidates = ["c"] + (["parallel", "serial"] if _numba_version() else [])
    chosen = "numpy"
    for mode in candidates:
        print(f"Teste Kern '{KERNEL_LABEL[mode]}' (einmalig) ...", flush=True)
        try:
            res = subprocess.run([sys.executable, os.path.abspath(__file__), "--kernel-test", mode],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=900)
            code = res.returncode
            last = (res.stderr.strip().splitlines() or [""])[-1]
        except subprocess.TimeoutExpired:
            code, last = "Timeout", ""
        if code == 0:
            print("  -> ok")
            chosen = mode
            break
        reason = "Segfault" if code == -11 else f"Rückgabewert {code}"
        print(f"  -> fehlgeschlagen ({reason}){': ' + last if last else ''}")
    try:
        with open(KERNEL_FILE, "w") as fh:
            json.dump({"key": key, "mode": chosen}, fh)
    except OSError:
        pass
    return chosen


# ====================================================================== Kopplungsmatrix
class Coupling:
    """Kopplungsmatrix, die in Abständen weich zu einem teilweise neuen Zufallsziel überblendet."""

    def __init__(self, n, verbose=True):
        self.n = n
        self.verbose = verbose
        if coupling_matrix is not None:
            self.K = np.array(coupling_matrix, dtype=float)
            assert self.K.shape == (n, n), "coupling_matrix muss n_species x n_species sein"
        else:
            self.K = rng.uniform(-1, 1, (n, n))
        self.start = self.K.copy()
        self.target = self.K.copy()
        self.t = 0
        self._print("Start")

    def _print(self, title):
        if not self.verbose:
            return
        print(f"Kopplungsmatrix ({title}):")
        print(np.array2string(self.K if title == "Start" else self.target,
                              precision=2, suppress_small=True))

    def step(self):
        self.t += 1
        if self.t % matrix_period == 0:
            self.start = self.K.copy()
            self.target = self.K.copy()
            change = rng.random((self.n, self.n)) < matrix_change_fraction
            if not change.any():
                change[rng.integers(self.n), rng.integers(self.n)] = True
            self.target[change] = rng.uniform(-1, 1, change.sum())
            self._print("neues Ziel")
        phase = self.t % matrix_period
        if phase <= matrix_blend:
            s = phase / matrix_blend
            s = s * s * (3 - 2 * s)                  # weicher Übergang
            self.K = (1 - s) * self.start + s * self.target
        if matrix_drift:
            self.K += matrix_drift * rng.standard_normal(self.K.shape)
            np.clip(self.K, -1, 1, out=self.K)
        return self.K


# ====================================================================== Partikel
class Particles:
    def __init__(self):
        if species_counts is not None:
            counts = list(species_counts)
            assert len(counts) == n_species, "species_counts braucht n_species Einträge"
        else:
            base = n_particles // n_species
            counts = [base + (1 if k < n_particles % n_species else 0) for k in range(n_species)]
        self.species = np.repeat(np.arange(n_species), counts)
        n = self.species.size
        self.face = rng.integers(0, cl.FACES, n)
        self.x = rng.random(n) * N
        self.y = rng.random(n) * N
        self.vx = np.zeros(n)
        self.vy = np.zeros(n)

    def forces(self, K):
        """Kraft auf jedes Partikel als 3D-Vektor."""
        P = to_3d(self.face, self.x, self.y)
        return compute_forces(P, self.species, np.ascontiguousarray(K, dtype=np.float64),
                              float(r_max), float(beta), float(repel_force))

    def step(self, K):
        F = self.forces(K)
        # Kraft in die lokale Flächenebene projizieren
        fx = (F * FACE_EX[self.face]).sum(axis=1)
        fy = (F * FACE_EY[self.face]).sum(axis=1)
        self.vx = friction * self.vx + force_multiplier * fx
        self.vy = friction * self.vy + force_multiplier * fy
        sp = np.hypot(self.vx, self.vy)
        fast = sp > max_speed
        self.vx[fast] *= max_speed / sp[fast]
        self.vy[fast] *= max_speed / sp[fast]
        self.x += self.vx
        self.y += self.vy
        cl.wrap_particles(self.face, self.x, self.y, self.vx, self.vy)


# ====================================================================== Darstellung
def color_table(n):
    """Farbmatrix (n, 3) in 0..1; fehlende Farben werden gleichmäßig im Farbkreis ergänzt."""
    cols = [np.array(c, dtype=float) / 255.0 for c in species_colors[:n]]
    for k in range(len(cols), n):
        h = (k * 0.618034) % 1.0
        rgb = np.clip(np.abs((h * 6 + np.array([0, 4, 2])) % 6 - 3) - 1, 0, 1)
        cols.append(0.25 + 0.75 * rgb)
    return np.array(cols)


class Renderer:
    def __init__(self, species):
        self.size = cl.FACES * N * N
        col = color_table(n_species)[species] * particle_gain           # (n, 3)
        # Pixel selbst + 8 Nachbarpixel (kantenübergreifend) mit Halo-Gewichten
        offs = [(0, 0, 1.0)]
        if halo:
            offs += [(dx, dy, halo / 6.0 * (1.0 if dx == 0 or dy == 0 else 0.5))
                     for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
        self.nb = np.stack([cl.PAD_IDX[:, 1 + dx:1 + dx + N, 1 + dy:1 + dy + N].ravel()
                            for dx, dy, _ in offs], axis=1)                # (Pixel, k)
        w = np.array([o[2] for o in offs])
        self.weights = (col[:, None, :] * w[None, :, None])               # (n, k, 3)
        self.chan = np.arange(3) * self.size
        self.glow = np.zeros(3 * self.size)
        self.frame = np.zeros((cl.FACES, N, N, 3), dtype=np.uint8)

    def render(self, face, x, y):
        flat = (face * N + x.astype(np.intp)) * N + y.astype(np.intp)
        idx = (self.nb[flat][:, :, None] + self.chan).ravel()             # (n, k, 3)
        self.glow *= trail_fade
        self.glow += np.bincount(idx, weights=self.weights.ravel(), minlength=3 * self.size)
        out = 1.0 - np.exp(-1.5 * self.glow)
        self.frame[:] = out.reshape(3, cl.FACES, N, N).transpose(1, 2, 3, 0) * (255 * brightness)
        return self.frame


# ====================================================================== Hauptschleife
def main():
    global compute_forces
    mode = choose_kernel()
    compute_forces = get_kernel(mode)
    extra = f", {C_THREADS} Threads" if mode == "c" else ""
    print(f"Kraftkern: {KERNEL_LABEL[mode]}{extra}")

    c = cube.Cube(UDP_IP, UDP_PORT)
    limiter = cl.FrameLimiter(target_fps)
    coupling = Coupling(n_species)
    parts = Particles()
    renderer = Renderer(parts.species)

    while True:
        K = coupling.step()
        parts.step(K)
        c.sendFrame(renderer.render(parts.face, parts.x, parts.y))
        limiter.tick()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--kernel-test":
        sys.exit(0 if _kernel_selftest(sys.argv[2]) else 1)
    main()
