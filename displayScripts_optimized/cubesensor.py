#!/usr/bin/env python3
"""MPU6050 (HW-123 / GY-521) am Raspberry Pi: Lage und Beschleunigung des Würfels.

Zwei Rollen in einer Datei:

1) Dienst (läuft dauerhaft, z. B. als systemd-Unit cube@cubesensor):
       python3 cubesensor.py              # misst und schreibt SENSOR_FILE
       python3 cubesensor.py --show       # misst und zeigt Live-Werte (zum Einrichten)
       python3 cubesensor.py --calibrate  # Kalibrierung: Würfel ruhig mit der Unterseite
                                          # (Fläche 5) nach unten hinstellen

2) Bibliothek für Animationsskripte (beliebig viele gleichzeitig):
       import cubesensor
       s = cubesensor.read()              # dict oder None, wenn keine frischen Daten
       if s: up = s["up_face"]; shake = s["shake"]

Die Werte werden atomar (temporäre Datei + Umbenennen) als eine JSON-Zeile in
/dev/shm geschrieben (RAM, schont die SD-Karte). Leser sehen daher nie halbe Daten.

Koordinaten: alle Vektoren im Würfelsystem der Animationen
    +x = Fläche 0, -x = Fläche 2, -y = Fläche 1, +y = Fläche 3, +z = Fläche 4, -z = Fläche 5
Kalibrierung (--calibrate): Die Einbaulage des Sensors muss nicht bekannt sein. Die bei der
Kalibrierung gemessene Schwerkraftrichtung wird als "Fläche 5 unten" festgelegt; die Lage der
Seitenflächen 0–3 ergibt sich daraus beliebig, aber fest und konsistent (echte Drehung, daher
stimmt auch der Drehsinn des Gyros). Zusätzlich wird der Gyro-Nullpunkt bestimmt.
Gespeichert wird in cubesensor_calib.json neben diesem Skript; beim Start wird die Datei
automatisch geladen, ein laufender Dienst übernimmt eine neue Kalibrierung ohne Neustart.
"""
import json
import math
import os
import sys
import time

# ====================================================================== Einstellungen
I2C_BUS = 1                     # Pi: Bus 1 (GPIO2 = SDA, GPIO3 = SCL)
I2C_ADDR = 0x68                 # 0x69, wenn AD0 auf 3,3 V liegt
POLL_HZ = 100                   # Abtast- und Schreibrate
SENSOR_FILE = "/dev/shm/cube_sensor.json"

ACCEL_RANGE_G = 2               # 2, 4, 8 oder 16
GYRO_RANGE_DPS = 500            # 250, 500, 1000 oder 2000
DLPF = 3                        # interner Tiefpass: 0=260 Hz, 1=184, 2=94, 3=44, 4=21, 5=10, 6=5 Hz

GRAVITY_TAU = 0.4               # s, Zeitkonstante der Schwerkraftschätzung (Tiefpass)
SHAKE_TAU = 0.15                # s, Glättung des Schüttel-Werts

# Einbaulage – wird nur verwendet, solange keine Kalibrierung vorhanden ist.
# Welche Würfelachse entspricht der Sensor-Achse x, y, z? z. B. ("+y", "-x", "+z")
AXIS_MAP = ("+x", "+y", "+z")

CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cubesensor_calib.json")
CALIB_DOWN_FACE = 5             # Fläche, die bei der Kalibrierung nach unten zeigt
CALIB_SECONDS = 3.0             # Messdauer der Kalibrierung
CALIB_MAX_MOTION = 0.02         # g; stärkere Schwankung -> Würfel bewegt, Abbruch
STALE_AFTER = 0.5               # s; ältere Daten liefert read() als None

# ====================================================================== Würfelgeometrie
# Nach außen zeigende Normalen der Flächen im Würfelsystem (passend zu cubelib)
FACE_NORMALS = {0: (1, 0, 0), 1: (0, -1, 0), 2: (-1, 0, 0),
                3: (0, 1, 0), 4: (0, 0, 1), 5: (0, 0, -1)}


def _axis_matrix(axis_map):
    """AXIS_MAP -> 3x3-Matrix M mit v_cube = M @ v_sensor."""
    idx = {"x": 0, "y": 1, "z": 2}
    M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    used = set()
    for s_axis, spec in enumerate(axis_map):
        sign = -1 if spec[0] == "-" else 1
        c_axis = idx[spec[-1].lower()]
        if c_axis in used:
            raise ValueError(f"AXIS_MAP {axis_map}: Würfelachse doppelt vergeben")
        used.add(c_axis)
        M[c_axis][s_axis] = sign
    det = (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
           - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
           + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
    if det < 0:
        raise ValueError(f"AXIS_MAP {axis_map} ist gespiegelt – so kann der Sensor nicht eingebaut sein "
                         "(der Drehsinn des Gyros wäre falsch). Bei genau einer Achse das Vorzeichen umdrehen.")
    return M


def _rot(M, v):
    return [M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2],
            M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2],
            M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2]]


def _norm(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _align(u, t):
    """Kleinste Drehung (3x3), die den Einheitsvektor u auf t dreht (Rodrigues)."""
    c = sum(a * b for a, b in zip(u, t))
    v = [u[1] * t[2] - u[2] * t[1], u[2] * t[0] - u[0] * t[2], u[0] * t[1] - u[1] * t[0]]
    s = _norm(v)
    if s < 1e-9:
        if c > 0:
            return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        # entgegengesetzt: 180° um eine zu u senkrechte Achse
        a = [1, 0, 0] if abs(u[0]) < 0.9 else [0, 1, 0]
        k = [u[1] * a[2] - u[2] * a[1], u[2] * a[0] - u[0] * a[2], u[0] * a[1] - u[1] * a[0]]
        kn = _norm(k)
        k = [x / kn for x in k]
        return [[2 * k[i] * k[j] - (1 if i == j else 0) for j in range(3)] for i in range(3)]
    k = [x / s for x in v]
    K = [[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]]
    K2 = _matmul(K, K)
    return [[(1 if i == j else 0) + s * K[i][j] + (1 - c) * K2[i][j] for j in range(3)] for i in range(3)]


def _axis_aligned_base(u):
    """Achsparallele Drehung P (vorzeichenbehaftete Permutation, det = +1), die die
    Sensorachse, die am stärksten nach oben zeigt, auf Würfel-+z legt."""
    k = max(range(3), key=lambda i: abs(u[i]))
    s = 1 if u[k] > 0 else -1
    a, b = (k + 1) % 3, (k + 2) % 3
    P = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    P[0][a] = 1
    P[1][b] = 1
    P[2][k] = s
    det = (P[0][0] * (P[1][1] * P[2][2] - P[1][2] * P[2][1])
           - P[0][1] * (P[1][0] * P[2][2] - P[1][2] * P[2][0])
           + P[0][2] * (P[1][0] * P[2][1] - P[1][1] * P[2][0]))
    if det < 0:
        P[1][b] = -1
    return P, k, s


def up_face_of(up):
    """Fläche, deren Normale am besten mit der Hochrichtung übereinstimmt."""
    return max(FACE_NORMALS, key=lambda f: sum(a * b for a, b in zip(FACE_NORMALS[f], up)))


# ====================================================================== Lesen (für Animationen)
_cache = {"mtime": None, "data": None}


def read(max_age=STALE_AFTER):
    """Aktuelle Sensordaten als dict, oder None, wenn der Dienst nicht läuft bzw.
    die Daten älter als max_age Sekunden sind. Billig genug für jeden Frame
    (die Datei wird nur neu geparst, wenn sie sich geändert hat).

    Felder:
      t        Zeitstempel (time.time())
      acc      gemessene Beschleunigung [g] (in Ruhe: 1 g nach oben)
      up       Hochrichtung (Einheitsvektor, geglättet)
      down     Schwerkraftrichtung = -up
      lin_acc  Bewegungsbeschleunigung ohne Schwerkraft [g]
      shake    geglätteter Betrag von lin_acc [g] (Ruhe ~0,01)
      gyro     Drehrate [°/s]
      up_face  Fläche, die gerade oben ist (0..5)
      down_face Fläche, die gerade unten ist
      tilt     Winkel zwischen up und der Normalen von up_face [°] (0 = flach)
      temp     Sensortemperatur [°C]
    """
    try:
        mtime = os.stat(SENSOR_FILE).st_mtime
        if mtime != _cache["mtime"]:
            with open(SENSOR_FILE) as fh:
                _cache["data"] = json.loads(fh.read())
            _cache["mtime"] = mtime
    except (OSError, ValueError):
        return None
    d = _cache["data"]
    if d is None or time.time() - d.get("t", 0) > max_age:
        return None
    return d


# ====================================================================== MPU6050
class MPU6050:
    PWR_MGMT_1, SMPLRT_DIV, CONFIG = 0x6B, 0x19, 0x1A
    GYRO_CONFIG, ACCEL_CONFIG, ACCEL_XOUT_H, WHO_AM_I = 0x1B, 0x1C, 0x3B, 0x75

    def __init__(self, bus_no=I2C_BUS, addr=I2C_ADDR, reset=True):
        try:
            from smbus2 import SMBus
        except ImportError:
            from smbus import SMBus          # python3-smbus aus i2c-tools
        self.bus = SMBus(bus_no)
        self.addr = addr
        self.acc_scale = 16384.0 / (ACCEL_RANGE_G / 2)
        self.gyro_scale = 131.0 / (GYRO_RANGE_DPS / 250)
        self.init(reset)

    def init(self, reset=True):
        who = self.bus.read_byte_data(self.addr, self.WHO_AM_I)
        if who != 0x68:
            print(f"Hinweis: WHO_AM_I = 0x{who:02x} (erwartet 0x68) – evtl. Klon, mache weiter.")
        w = self.bus.write_byte_data
        if reset:                                           # (nicht bei laufendem Dienst)
            w(self.addr, self.PWR_MGMT_1, 0x80)             # Reset
            time.sleep(0.1)
        w(self.addr, self.PWR_MGMT_1, 0x01)                 # aufwecken, Takt = Gyro-PLL
        time.sleep(0.05)
        w(self.addr, self.CONFIG, DLPF & 0x07)
        w(self.addr, self.SMPLRT_DIV, 0)
        w(self.addr, self.GYRO_CONFIG, {250: 0, 500: 1, 1000: 2, 2000: 3}[GYRO_RANGE_DPS] << 3)
        w(self.addr, self.ACCEL_CONFIG, {2: 0, 4: 1, 8: 2, 16: 3}[ACCEL_RANGE_G] << 3)

    def read_raw(self):
        """(acc[g], gyro[°/s], temp[°C]) im Sensorsystem."""
        b = self.bus.read_i2c_block_data(self.addr, self.ACCEL_XOUT_H, 14)
        v = [((b[i] << 8) | b[i + 1]) for i in range(0, 14, 2)]
        v = [x - 65536 if x > 32767 else x for x in v]
        acc = [v[0] / self.acc_scale, v[1] / self.acc_scale, v[2] / self.acc_scale]
        temp = v[3] / 340.0 + 36.53
        gyro = [v[4] / self.gyro_scale, v[5] / self.gyro_scale, v[6] / self.gyro_scale]
        return acc, gyro, temp


# ====================================================================== Dienst
def load_calibration():
    """Liefert (Matrix Sensor -> Würfel, Gyro-Nullpunkt im Sensorsystem, Info-Text).
    Ohne gültige Kalibrierung: AXIS_MAP und Nullpunkt 0."""
    M = _axis_matrix(AXIS_MAP)
    try:
        with open(CALIB_FILE) as fh:
            c = json.load(fh)
    except FileNotFoundError:
        return M, [0.0, 0.0, 0.0], "keine Kalibrierung gefunden – AXIS_MAP wird verwendet"
    except (OSError, ValueError) as e:
        return M, [0.0, 0.0, 0.0], f"Kalibrierdatei unlesbar ({e}) – AXIS_MAP wird verwendet"
    bias = [float(x) for x in c.get("gyro_bias", [0.0, 0.0, 0.0])]
    R = c.get("matrix")
    if R is None:                                   # ältere Datei: nur Gyro-Nullpunkt
        return M, bias, "Kalibrierung ohne Lagekorrektur (bitte --calibrate erneut ausführen)"
    R = [[float(x) for x in row] for row in R]
    return R, bias, f"Kalibrierung vom {c.get('date', '?')} geladen (Fläche {c.get('down_face', 5)} = unten)"


def _write_atomic(path, text):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o644)
    except OSError:
        pass


def calibrate():
    print(f"Kalibrierung: Würfel mit Fläche {CALIB_DOWN_FACE} nach unten ruhig hinstellen.")
    print(f"Messe {CALIB_SECONDS:.0f} s ...", flush=True)
    mpu = MPU6050(reset=False)                      # läuft der Dienst, wird er nicht gestört
    time.sleep(0.1)
    accs, gyros = [], []
    t_end = time.monotonic() + CALIB_SECONDS
    while time.monotonic() < t_end:
        a, g, _ = mpu.read_raw()
        accs.append(a)
        gyros.append(g)
        time.sleep(1.0 / POLL_HZ)
    n = len(accs)
    a_mean = [sum(a[k] for a in accs) / n for k in range(3)]
    g_mean = [sum(g[k] for g in gyros) / n for k in range(3)]
    mags = [_norm(a) for a in accs]
    m_mean = sum(mags) / n
    m_std = math.sqrt(sum((m - m_mean) ** 2 for m in mags) / n)
    spread = max(_norm([a[k] - a_mean[k] for k in range(3)]) for a in accs)

    if m_std > CALIB_MAX_MOTION or spread > 5 * CALIB_MAX_MOTION:
        print(f"Abbruch: Würfel hat sich bewegt (Schwankung {m_std:.3f} g, max. Abweichung {spread:.3f} g). "
              "Ruhig hinstellen und erneut starten.")
        return 1
    if not 0.85 < m_mean < 1.15:
        print(f"Warnung: gemessene Schwerkraft {m_mean:.3f} g weicht deutlich von 1 g ab "
              "(Messbereich/Sensor prüfen). Kalibriere trotzdem.")

    # Gemessene Hochrichtung im Sensorsystem exakt auf "oben" (Gegenrichtung von CALIB_DOWN_FACE)
    # drehen – unabhängig von der Einbaulage. Die Drehung um die Senkrechte (welche Seite
    # Fläche 0..3 ist) lässt sich aus einer Messung nicht bestimmen; gewählt wird die Lage, in
    # der die Sensorachsen parallel zu den Würfelkanten liegen (übliche Montage), danach wird
    # nur noch die Restneigung korrigiert. "Unten" stimmt in jedem Fall exakt.
    un = _norm(a_mean)
    u = [x / un for x in a_mean]
    P, k, sgn = _axis_aligned_base(u)
    C = _align(_rot(P, u), [0.0, 0.0, 1.0])            # Restneigung
    target = [-x for x in FACE_NORMALS[CALIB_DOWN_FACE]]
    R = _matmul(_align([0.0, 0.0, 1.0], target), _matmul(C, P))
    resid = math.degrees(math.acos(max(-1.0, min(1.0, abs(u[k])))))

    calib = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "down_face": CALIB_DOWN_FACE,
        "matrix": [[round(x, 6) for x in row] for row in R],
        "gyro_bias": [round(x, 4) for x in g_mean],
        "gravity_g": round(m_mean, 4),
        "sensor_up": [round(x, 4) for x in u],
    }
    _write_atomic(CALIB_FILE, json.dumps(calib, indent=1))
    print(f"Gespeichert in {CALIB_FILE}")
    print(f"  Sensorachse {'+-'[sgn < 0]}{'xyz'[k]} zeigt nach oben (Restneigung {resid:.1f}° korrigiert)"
          f" -> Fläche {CALIB_DOWN_FACE} = unten")
    print(f"  Gyro-Nullpunkt [°/s]: {[round(x, 3) for x in g_mean]}")
    print(f"  gemessene Schwerkraft: {m_mean:.3f} g")
    return 0


def _calib_mtime():
    try:
        return os.stat(CALIB_FILE).st_mtime
    except OSError:
        return None


def run(show=False):
    M, bias, info = load_calibration()
    print(info, flush=True)
    calib_mtime = _calib_mtime()
    next_calib_check = time.monotonic() + 1.0
    dt = 1.0 / POLL_HZ
    a_g = math.exp(-dt / GRAVITY_TAU)
    a_s = math.exp(-dt / SHAKE_TAU)

    mpu = None
    grav = None
    shake = 0.0
    seq = 0
    errors = 0
    next_t = time.monotonic()
    last_show = 0.0
    while True:
        try:
            if mpu is None:
                mpu = MPU6050()
                print(f"MPU6050 an Bus {I2C_BUS}, Adresse 0x{I2C_ADDR:02x} – {POLL_HZ} Hz -> {SENSOR_FILE}",
                      flush=True)
            acc_s, gyro_s, temp = mpu.read_raw()
            errors = 0
        except OSError as e:                           # I2C-Fehler: neu initialisieren
            errors += 1
            if errors in (1, 10) or errors % 100 == 0:
                print(f"I2C-Fehler ({errors}x): {e}", flush=True)
            mpu = None
            time.sleep(min(1.0, 0.05 * errors))
            continue

        # neue Kalibrierung übernehmen, ohne den Dienst neu zu starten
        if time.monotonic() > next_calib_check:
            next_calib_check = time.monotonic() + 1.0
            mt = _calib_mtime()
            if mt != calib_mtime:
                calib_mtime = mt
                M, bias, info = load_calibration()
                grav = None
                print(info, flush=True)

        acc = _rot(M, acc_s)
        gyro = _rot(M, [g - b for g, b in zip(gyro_s, bias)])

        # Schwerkraftschätzung: Tiefpass der Beschleunigung (in Ruhe = 1 g nach oben)
        grav = acc[:] if grav is None else [a_g * g + (1 - a_g) * a for g, a in zip(grav, acc)]
        gn = _norm(grav) or 1.0
        up = [g / gn for g in grav]
        lin = [a - g for a, g in zip(acc, grav)]
        shake = a_s * shake + (1 - a_s) * _norm(lin)
        uf = up_face_of(up)
        n_up = FACE_NORMALS[uf]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, sum(a * b for a, b in zip(n_up, up))))))

        seq += 1
        data = {
            "t": time.time(), "seq": seq,
            "acc": [round(v, 4) for v in acc],
            "up": [round(v, 4) for v in up],
            "down": [round(-v, 4) for v in up],
            "lin_acc": [round(v, 4) for v in lin],
            "shake": round(shake, 4),
            "gyro": [round(v, 2) for v in gyro],
            "up_face": uf,
            "down_face": {0: 2, 2: 0, 1: 3, 3: 1, 4: 5, 5: 4}[uf],
            "tilt": round(tilt, 1),
            "temp": round(temp, 1),
        }
        _write_atomic(SENSOR_FILE, json.dumps(data))

        if show and time.monotonic() - last_show > 0.2:
            last_show = time.monotonic()
            print(f"oben: Fläche {uf} (Neigung {tilt:4.1f}°) | up {data['up']} | "
                  f"|acc| {_norm(acc):.3f} g | shake {shake:.3f} g | gyro {data['gyro']} | "
                  f"{temp:.1f} °C", flush=True)

        next_t += dt
        sleep = next_t - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)
        else:
            next_t = time.monotonic()                  # nicht nachholen, wenn zu langsam


if __name__ == "__main__":
    if "--calibrate" in sys.argv:
        sys.exit(calibrate())
    else:
        run(show="--show" in sys.argv)
