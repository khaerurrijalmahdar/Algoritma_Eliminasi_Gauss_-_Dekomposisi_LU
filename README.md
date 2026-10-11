# Eliminasi Gauss dan Dekomposisi LU

Program Python satu file untuk menyelesaikan sistem persamaan linear **A x = b** dengan empat metode, ditulis langsung dari rumusnya tanpa NumPy atau library eksternal.

| No | Metode | Pivoting |
|----|--------|----------|
| 1 | Eliminasi Gauss | Parsial |
| 2 | Gauss-Jordan | Parsial |
| 3 | Dekomposisi LU Doolittle | Tanpa pivoting |
| 4 | LU berpivot (`PA = LU`) | Parsial |

Program juga menghitung jumlah operasi aritmetika dan mengukur waktu eksekusi, sehingga keempat metode bisa dibandingkan.

## Demo web

Studi kasus juga tersedia sebagai halaman web interaktif (`Studi Kasus Eliminasi Gauss dan Dekomposisi LU.html`), langsung bisa dibuka di browser tanpa instalasi:

**[Buka halaman studi kasus](https://khaerurrijalmahdar.github.io/Algoritma_Eliminasi_Gauss_-_Dekomposisi_LU/)**

## Isi repositori

| File | Fungsi |
|------|--------|
| `Eliminasi Gauss dan Dekomposisi LU.py` | Program Python (dijalankan lewat terminal) |
| `Studi Kasus Eliminasi Gauss dan Dekomposisi LU.html` | Halaman web studi kasus |
| `README.md` | Dokumentasi ini |

## Persyaratan

Python 3.6+ saja. Tidak ada dependensi yang perlu diinstal.

## Cara menjalankan

Unduh `Eliminasi Gauss dan Dekomposisi LU.py`, lalu jalankan:

## Contoh keluaran

```
====================================================================
Kasus 1 (3x3): arus rangkaian listrik    solusi diharapkan (2, 3, 1)
====================================================================

--- Gauss ---
   x = [2.0, 3.0, 1.0]   residu = 7.1e-15   operasi = 28

--- Gauss-Jordan ---
   x = [2.0, 3.0, 1.0]   residu = 3.6e-15   operasi = 33

--- LU Doolittle ---
   x = [2.0, 3.0, 1.0]   residu = 7.1e-15   operasi = 37

--- PA = LU ---
   x = [2.0, 3.0, 1.0]   residu = 7.1e-15   operasi = 28
```

## Studi kasus bawaan

| Kasus | Ukuran | Konteks | Solusi |
|-------|--------|---------|--------|
| 1 | 3x3 | Arus rangkaian listrik | (2, 3, 1) |
| 2 | 4x4 | Produksi furnitur | (10, 20, 15, 5) |
| 3 | 5x5 | Harga buah | (3, 2, 4, 5, 1) |

## Rumus jumlah operasi

Mode `ops` mencocokkan jumlah operasi terukur dengan rumus berikut (n = ukuran matriks):

| Komponen | Rumus |
|----------|-------|
| Eliminasi Gauss | `2n³/3 + 3n²/2 − 7n/6` |
| Gauss-Jordan | `n³ + n² − n` |
| Faktorisasi LU | `(4n³ + 3n² − n) / 6` |
| Faktorisasi `PA = LU` | `(4n³ − 3n² − n) / 6` |
| Satu kali substitusi | `2n² − n` |

## Catatan

- LU Doolittle tanpa pivoting gagal bila ada pivot nol; gunakan `PA = LU` untuk kasus itu.
- Matriks singular terdeteksi dan dilaporkan sebagai `ZeroDivisionError` dengan pesan yang jelas.
- LU menguntungkan bila matriks A yang sama dipakai untuk banyak vektor b, karena faktorisasi hanya dilakukan sekali.
