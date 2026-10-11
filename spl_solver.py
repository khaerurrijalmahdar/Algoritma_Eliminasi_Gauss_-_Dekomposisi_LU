#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eliminasi Gauss dan Dekomposisi LU untuk penyelesaian A x = b.
Seluruh algoritma ditulis langsung dari rumusnya.

Empat algoritma:
  1. Eliminasi Gauss (pivoting parsial)
  2. Gauss-Jordan  (pivoting parsial)
  3. Dekomposisi LU Doolittle (tanpa pivoting)
  4. LU berpivot: PA = LU

"""
import argparse
import random
import time


# ----------------------------------------------------------------------
# Penghitung operasi: md = perkalian/pembagian, pm = penjumlahan/pengurangan
# ----------------------------------------------------------------------
class Cnt:
    def __init__(self):
        self.md = self.pm = 0

    @property
    def total(self):
        return self.md + self.pm


def fmt(v):
    v = 0.0 if abs(v) < 1e-12 else v
    return str(int(v)) if v == int(v) else ("%.4f" % v).rstrip("0").rstrip(".")


def tampil(M, judul=""):
    if judul:
        print("   " + judul)
    for r in M:
        print("     [" + "  ".join("%9s" % fmt(v) for v in r) + " ]")


def residu(A, x, b):
    return max(abs(sum(a * xi for a, xi in zip(row, x)) - bi) for row, bi in zip(A, b))


# ----------------------------------------------------------------------
# 1. Eliminasi Gauss dengan pivoting parsial
# ----------------------------------------------------------------------
def gauss(A, b, c=None, v=False):
    c = c or Cnt()
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    if v:
        tampil(M, "Matriks augmented [A | b]:")
    for k in range(n - 1):
        p = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[p][k]) < 1e-12:
            raise ZeroDivisionError("matriks singular pada kolom %d" % (k + 1))
        if p != k:
            M[p], M[k] = M[k], M[p]
            if v:
                print("   tukar R%d <-> R%d (pivot %s)" % (k + 1, p + 1, fmt(M[k][k])))
        for i in range(k + 1, n):
            m = M[i][k] / M[k][k]
            c.md += 1
            for j in range(k + 1, n + 1):
                M[i][j] -= m * M[k][j]
                c.md += 1
                c.pm += 1
            M[i][k] = 0.0
            if v:
                print("   R%d <- R%d - (%s)*R%d" % (i + 1, i + 1, fmt(m), k + 1))
        if v:
            tampil(M)
    if abs(M[n - 1][n - 1]) < 1e-12:
        raise ZeroDivisionError("matriks singular")
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n]
        for j in range(i + 1, n):
            s -= M[i][j] * x[j]
            c.md += 1
            c.pm += 1
        x[i] = s / M[i][i]
        c.md += 1
    return x


# ----------------------------------------------------------------------
# 2. Gauss-Jordan dengan pivoting parsial
# ----------------------------------------------------------------------
def gauss_jordan(A, b, c=None, v=False):
    c = c or Cnt()
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    if v:
        tampil(M, "Matriks augmented [A | b]:")
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[p][k]) < 1e-12:
            raise ZeroDivisionError("matriks singular pada kolom %d" % (k + 1))
        if p != k:
            M[p], M[k] = M[k], M[p]
            if v:
                print("   tukar R%d <-> R%d" % (k + 1, p + 1))
        for i in range(n):
            if i == k:
                continue
            m = M[i][k] / M[k][k]
            c.md += 1
            for j in range(k + 1, n + 1):
                M[i][j] -= m * M[k][j]
                c.md += 1
                c.pm += 1
            M[i][k] = 0.0
        if v:
            tampil(M, "setelah kolom %d dituntaskan:" % (k + 1))
    x = []
    for i in range(n):
        x.append(M[i][n] / M[i][i])
        c.md += 1
    return x


# ----------------------------------------------------------------------
# Substitusi (dipakai LU dan PA = LU)
# ----------------------------------------------------------------------
def substitusi(L, U, bb, c):
    n = len(bb)
    y = [0.0] * n
    for i in range(n):
        s = bb[i]
        for j in range(i):
            s -= L[i][j] * y[j]
            c.md += 1
            c.pm += 1
        y[i] = s
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= U[i][j] * x[j]
            c.md += 1
            c.pm += 1
        x[i] = s / U[i][i]
        c.md += 1
    return y, x


# ----------------------------------------------------------------------
# 3. Dekomposisi LU Doolittle (tanpa pivoting)
# ----------------------------------------------------------------------
def lu_faktor(A, c=None, v=False):
    c = c or Cnt()
    n = len(A)
    L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    U = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            s = 0.0
            for k in range(i):
                s += L[i][k] * U[k][j]
                c.md += 1
                c.pm += 1
            U[i][j] = A[i][j] - s
            c.pm += 1
        if U[i][i] == 0.0:
            raise ZeroDivisionError(
                "pivot nol pada U(%d,%d): LU tanpa pivoting gagal (gunakan PA = LU)" % (i + 1, i + 1))
        for j in range(i + 1, n):
            s = 0.0
            for k in range(i):
                s += L[j][k] * U[k][i]
                c.md += 1
                c.pm += 1
            L[j][i] = (A[j][i] - s) / U[i][i]
            c.pm += 1
            c.md += 1
    if v:
        tampil(L, "L =")
        tampil(U, "U =")
    return L, U


def lu(A, b, c=None, v=False):
    c = c or Cnt()
    L, U = lu_faktor(A, c, v)
    y, x = substitusi(L, U, b, c)
    if v:
        print("   y =", [fmt(t) for t in y])
    return x


# ----------------------------------------------------------------------
# 4. LU berpivot: P A = L U
# ----------------------------------------------------------------------
def plu_faktor(A, c=None, v=False):
    c = c or Cnt()
    n = len(A)
    U = [list(r) for r in A]
    L = [[0.0] * n for _ in range(n)]
    P = list(range(n))
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(U[i][k]))
        if abs(U[p][k]) < 1e-12:
            raise ZeroDivisionError("matriks singular pada kolom %d" % (k + 1))
        if p != k:
            U[p], U[k] = U[k], U[p]
            L[p], L[k] = L[k], L[p]
            P[p], P[k] = P[k], P[p]
            if v:
                print("   tukar baris %d <-> %d" % (k + 1, p + 1))
        for i in range(k + 1, n):
            m = U[i][k] / U[k][k]
            c.md += 1
            L[i][k] = m
            for j in range(k + 1, n):
                U[i][j] -= m * U[k][j]
                c.md += 1
                c.pm += 1
            U[i][k] = 0.0
    for i in range(n):
        L[i][i] = 1.0
    if v:
        print("   P (urutan baris A) =", [i + 1 for i in P])
        tampil(L, "L =")
        tampil(U, "U =")
    return L, U, P


def plu(A, b, c=None, v=False):
    c = c or Cnt()
    L, U, P = plu_faktor(A, c, v)
    y, x = substitusi(L, U, [b[i] for i in P], c)
    return x


# ----------------------------------------------------------------------
# Data studi kasus berdasarkan ukuran matriks
# ----------------------------------------------------------------------
KASUS = [
    ("Kasus 1 (3x3): arus rangkaian listrik", [[10, -2, -1], [-2, 8, -3], [-1, -3, 9]], [13, 17, -2], "(2, 3, 1)"),
    ("Kasus 2 (4x4): produksi furnitur", [[4, 1, 1, 0], [1, 5, 2, 1], [1, 2, 6, 1], [0, 1, 1, 4]], [75, 145, 145, 55], "(10, 20, 15, 5)"),
    ("Kasus 3 (5x5): harga buah", [[0, 2, 1, 1, 0], [1, 1, 1, 0, 2], [2, 1, 3, 1, 1], [1, 0, 1, 2, 1], [1, 2, 0, 1, 3]], [13, 11, 26, 18, 15], "(3, 2, 4, 5, 1)"),
]

METODE = [
    ("Gauss", gauss),
    ("Gauss-Jordan", gauss_jordan),
    ("LU Doolittle", lu),
    ("PA = LU", plu),
]


def matriks_uji(n, seed=1):
    """Matriks acak dominan diagonal (tidak ada pivot nol)."""
    rnd = random.Random(seed)
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i):
            A[i][j] = A[j][i] = rnd.random() - 0.5
        A[i][i] = float(n)
    return A


def ruas_kanan(A, q=1):
    return [q * sum(r) for r in A]


# ----------------------------------------------------------------------
# Rumus jumlah operasi (Bab III jurnal)
# ----------------------------------------------------------------------
def T_gauss(n):
    return 2 * n**3 / 3 + 3 * n**2 / 2 - 7 * n / 6


def T_gj(n):
    return n**3 + n**2 - n


def T_lu(n):
    return (4 * n**3 + 3 * n**2 - n) / 6.0


def T_plu(n):
    return (4 * n**3 - 3 * n**2 - n) / 6.0


def S_sub(n):
    return 2 * n**2 - n


def R_teori(n, k, T):
    return k * T_gauss(n) / (T(n) + k * S_sub(n))


# ----------------------------------------------------------------------
# Mode program
# ----------------------------------------------------------------------
def mode_kasus(v):
    for judul, A, b, harapan in KASUS:
        print("\n" + "=" * 68)
        print(judul, "   solusi diharapkan", harapan)
        print("=" * 68)
        for nama, f in METODE:
            print("\n--- %s ---" % nama)
            c = Cnt()
            try:
                x = f(A, b, c, v)
                print("   x = %s   residu = %.1e   operasi = %d" % ([round(t, 6) for t in x], residu(A, x, b), c.total))
            except ZeroDivisionError as e:
                print("   GAGAL:", e)


def ukur_ops(n):
    A = matriks_uji(n)
    b = ruas_kanan(A)
    h = {}
    c = Cnt(); gauss(A, b, c); h["gauss"] = c.total
    c = Cnt(); gauss_jordan(A, b, c); h["gj"] = c.total
    c = Cnt(); lu_faktor(A, c); h["lu_f"] = c.total
    c = Cnt(); plu_faktor(A, c); h["plu_f"] = c.total
    L, U = lu_faktor(A)
    c = Cnt(); substitusi(L, U, b, c); h["sub"] = c.total
    return h


def mode_ops():
    print("Verifikasi jumlah operasi (terukur vs rumus)\n")
    print("%5s | %-16s | %10s %10s" % ("n", "metode", "terukur", "rumus"))
    ok = True
    for n in (3, 10, 50, 100):
        h = ukur_ops(n)
        for nama, a, r in [("Gauss", h["gauss"], T_gauss(n)), ("Gauss-Jordan", h["gj"], T_gj(n)),
                           ("LU faktorisasi", h["lu_f"], T_lu(n)), ("PA=LU faktorisasi", h["plu_f"], T_plu(n)),
                           ("1 substitusi", h["sub"], S_sub(n))]:
            cocok = abs(a - r) < 1e-6
            ok &= cocok
            print("%5d | %-16s | %10.0f %10.0f  %s" % (n, nama, a, r, "OK" if cocok else "BEDA"))
    print("\nSemua cocok" if ok else "\nAda yang tidak cocok")


def waktu(f, ulang=5):
    t = []
    for _ in range(ulang):
        t0 = time.perf_counter()
        f()
        t.append(time.perf_counter() - t0)
    return sorted(t)[len(t) // 2] * 1000.0


def mode_bench():
    print("Pengukuran waktu (ms, median 5 ulangan)\n")
    print("A. Empat metode, satu ruas kanan")
    print("%5s | %9s %9s %9s %9s" % ("n", "Gauss", "G-Jordan", "LU", "PA=LU"))
    for n in (50, 100, 150):
        A = matriks_uji(n)
        b = ruas_kanan(A)
        print("%5d | %9.2f %9.2f %9.2f %9.2f" % (n, waktu(lambda: gauss(A, b)), waktu(lambda: gauss_jordan(A, b)),
                                                waktu(lambda: lu(A, b)), waktu(lambda: plu(A, b))))
    print("\nB. Banyak ruas kanan (k): Gauss berulang vs LU vs PA=LU")
    print("%5s %3s | %9s %9s %9s | Gauss/LU Gauss/PLU" % ("n", "k", "Gauss", "LU", "PA=LU"))
    for n in (100, 150):
        A = matriks_uji(n)
        for k in (1, 10, 20):
            bs = [ruas_kanan(A, q + 1) for q in range(k)]
            tg = waktu(lambda: [gauss(A, b) for b in bs])

            def f_lu():
                L, U = lu_faktor(A)
                return [substitusi(L, U, b, Cnt()) for b in bs]

            def f_plu():
                L, U, P = plu_faktor(A)
                return [substitusi(L, U, [b[i] for i in P], Cnt()) for b in bs]
            tl, tp = waktu(f_lu), waktu(f_plu)
            print("%5d %3d | %9.2f %9.2f %9.2f | %8.2f %9.2f" % (n, k, tg, tl, tp, tg / tl, tg / tp))


def mode_input():
    n = int(input("Ukuran n: "))
    print("Masukkan matriks A baris demi baris (pisahkan dengan spasi):")
    A = [[float(t) for t in input("  baris %d: " % (i + 1)).split()] for i in range(n)]
    b = [float(t) for t in input("Vektor b (n angka): ").split()]
    print("Metode: 1 Gauss, 2 Gauss-Jordan, 3 LU, 4 PA = LU")
    nama, f = METODE[int(input("Pilih: ")) - 1]
    c = Cnt()
    try:
        x = f(A, b, c, True)
        print("\nx =", [round(t, 8) for t in x], "  residu = %.1e" % residu(A, x, b), "  operasi =", c.total)
    except ZeroDivisionError as e:
        print("GAGAL:", e)


def menu():
    """Menu interaktif bila program dijalankan tanpa argumen (mis. tombol Run di editor)."""
    pilihan = {
        "1": ("Tiga studi kasus x empat metode", lambda: mode_kasus(False)),
        "2": ("Tiga studi kasus + langkah detail", lambda: mode_kasus(True)),
        "3": ("Verifikasi jumlah operasi terhadap rumus", mode_ops),
        "4": ("Pengukuran waktu (sekitar 1-2 menit)", mode_bench),
        "5": ("Masukkan matriks sendiri", mode_input),
    }
    while True:
        print("\n=== Eliminasi Gauss dan Dekomposisi LU ===")
        for k, (judul, _) in pilihan.items():
            print("  %s. %s" % (k, judul))
        print("  0. Keluar")
        p = input("Pilih menu: ").strip()
        if p == "0":
            break
        if p in pilihan:
            pilihan[p][1]()
        else:
            print("Pilihan tidak dikenal.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", choices=["kasus", "ops", "bench", "input"],
                    help="kosongkan untuk membuka menu interaktif")
    ap.add_argument("-v", "--verbose", action="store_true", help="tampilkan langkah detail")
    a = ap.parse_args()
    if a.mode is None:
        menu()
    else:
        {"kasus": lambda: mode_kasus(a.verbose), "ops": mode_ops,
         "bench": mode_bench, "input": mode_input}[a.mode]()
