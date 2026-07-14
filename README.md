# Prediksi TMA — Bengawan Solo (Sebelas Maret Statistics Data Science 2026)

Pipeline prediksi Tinggi Muka Air (TMA) untuk 30 pos pemantauan di DAS
Bengawan Solo, dikemas dalam satu notebook: **`tma_pipeline_notebook.ipynb`**.

## Ringkasan pendekatan

1. Agregasi fitur eksogen (cuaca/tanah/iklim per jam) menjadi fitur rolling 6h/24h/72h/168h
2. Fitur hulu-hilir dari topologi sungai (HydroRIVERS) — TMA pos hulu sebagai prediktor pos hilir
3. Fitur lag & rolling TMA per pos, dengan outlier capping ringan
4. Model **LightGBM** (1 model global untuk 30 pos, `nama_pos` sebagai fitur kategorikal)
5. **Recursive/autoregressive forecasting** untuk 8 bulan ke depan, dengan **Tukey-fence clipping** untuk mencegah drift
6. Validasi yang jujur: simulasi recursive forecasting pada periode hold-out (bukan cuma one-step)

**Hasil:**
| Metode | RMSE recursive (validasi) |
|---|---|
| Tanpa clipping | 2.08 |
| Dengan clipping | 1.54 |
| Dengan clipping + fitur hulu-hilir | **1.47** |

Skor leaderboard Kaggle: **1.66**

## 1. Struktur folder yang dibutuhkan

```
sebelas-maret-statistics-data-science-2026/
├── data_pendukung/
│   ├── HydroRIVERS_v10_au_shp/
│   ├── data_lingkungan.csv
│   ├── HydroRIVERS_TechDoc_v10.pdf
│   └── koordinat_pos.csv
├── sample_submission.csv
├── test.csv
├── train.csv
├── tma_pipeline_notebook.ipynb
├── requirements.txt
└── README.md
```

> Dataset (`*.csv`, `data_pendukung/`, `output/`) sengaja di-gitignore dan
> tidak ikut di-push ke repo ini — download manual dari halaman kompetisi
> Kaggle dan taruh di lokasi di atas sebelum menjalankan notebook.

## 2. Setup environment

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

## 3. Menjalankan notebook

Buka `tma_pipeline_notebook.ipynb`, lalu di **Bagian 0 (Setup)** sesuaikan path:

```python
DATA_DIR = Path('.')            # folder dataset (berisi train.csv, test.csv, data_pendukung/)
OUTPUT_DIR = Path('./output')   # folder untuk simpan file perantara & hasil
```

Lalu **Run All**. Notebook akan jalan berurutan lewat semua tahap: EDA →
fitur eksogen → fitur hulu-hilir → lag features → training → validasi
recursive → training final → forecast → `submission.csv`.

**Estimasi waktu total:** ~15-20 menit (tahap paling lama: baca
`data_lingkungan.csv` 155MB, dan recursive forecasting 726 timestamp).

## 4. Struktur notebook

| Bagian | Isi |
|---|---|
| 0 | Setup path & konfigurasi |
| 1 | EDA singkat |
| 2 | Fitur eksogen (agregasi cuaca/tanah/iklim) |
| 3 | Fitur hulu-hilir (HydroRIVERS) |
| 4 | Grid waktu penuh + lag features + outlier capping |
| 5 | Training LightGBM + validasi one-step |
| 6 | **Validasi recursive yang jujur** — kenapa validasi one-step menyesatkan |
| 7 | Training final + forecast recursive + `submission.csv` |
| 8 | Verifikasi submission sebelum submit ke Kaggle |
| 9 | Kesimpulan & ide pengembangan lanjutan |

## 5. Kenapa ada bagian "validasi recursive"?

Evaluasi RMSE biasa (one-step, pakai lag dari data historis asli) **tidak**
merepresentasikan performa asli, karena forecast ke test (8 bulan ke depan)
dilakukan **recursive**: prediksi langkah `t` dipakai sebagai lag untuk
memprediksi langkah `t+1`, dst. Error bisa menumpuk ("drift"), terutama untuk
pos dengan variansi historis sangat kecil (near-konstan, misal `Lorog`).
Bagian 6 di notebook mensimulasikan proses recursive yang sama pada periode
yang nilai aslinya kita tahu, sehingga RMSE yang dilaporkan jujur — dan
otomatis menguji apakah **Tukey-fence clipping** (`[Q1 - k*IQR, Q3 + k*IQR]`
per pos) membantu redam drift tersebut.

## 6. Yang masih perlu diperbaiki

Pos dengan RMSE recursive tertinggi yang **tidak** terpengaruh clipping
maupun fitur hulu-hilir (kemungkinan butuh investigasi/fitur tambahan):
`Bojonegoro - Kali Kethek`, `Wonogiri Dam`, `Cepu`, `Jurug`.

Ide pengembangan lanjutan lainnya: hyperparameter tuning (Optuna), model
terpisah per pos vs global, arsitektur LSTM/GNN untuk memodelkan 30 pos
sebagai graf sungai, blending prediksi dengan persistence/exponential
smoothing untuk redam drift lebih halus daripada hard clipping.