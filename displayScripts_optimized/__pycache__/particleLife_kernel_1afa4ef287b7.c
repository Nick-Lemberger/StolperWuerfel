
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
