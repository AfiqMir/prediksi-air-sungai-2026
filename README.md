# Prediksi TMA — Bengawan Solo (Sebelas Maret Statistics Data Science 2026)

Repositori eksperimen prediksi Tinggi Muka Air (TMA) untuk 30 pos pemantauan
di DAS Bengawan Solo. Target kompetisi adalah memprediksi TMA tiga kali sehari
untuk periode 19 September 2025 sampai 18 Mei 2026.

Proyek ini dimulai dari baseline LightGBM recursive, lalu dikembangkan menjadi
pipeline CatBoost direct/non-recursive dengan validasi rolling-origin, target
robust per pos, penanganan spike sensor, pemilihan jumlah tree berdasarkan RMSE
kompetisi, dan koreksi state terakhir yang meluruh terhadap horizon prediksi.

## Status saat ini

- Baseline LightGBM recursive memperoleh RMSE validasi 1,47 dan skor public
  leaderboard Kaggle **1,66**.
- Kandidat terbaik berdasarkan validasi lokal saat ini adalah **CatBoost RMSE +
  state anchor**, dengan mean RMSE empat fold **1,4466** dan pooled RMSE
  **1,6144**.
- File kandidat submission saat ini dihasilkan sebagai
  `output_catboost_experiments/submission_rmse_anchor.csv`.
- Skor leaderboard untuk kandidat CatBoost terbaru belum dicatat di repositori.
- Estimasi `API_TEST` terbaik yang tercatat tetap **1,5926** dari station-stack;
  eksperimen recency-weighted terbaru ditolak karena estimasinya memburuk menjadi
  **1,6144** walaupun OOF lokal membaik.

> Metrik LightGBM dan CatBoost berasal dari skema validasi yang berbeda, jadi
> angka keduanya tidak sepenuhnya apple-to-apple. Perbandingan utama untuk
> pemilihan model dilakukan antarvarian CatBoost pada empat fold yang sama.

## Evolusi pendekatan

### 1. Baseline LightGBM recursive

Pipeline awal berada di `tma_pipeline_notebook.ipynb` dan menggunakan:

1. agregasi fitur cuaca, tanah, dan iklim pada jendela 6/24/72/168 jam;
2. fitur topologi hulu-hilir dari HydroRIVERS;
3. fitur lag dan rolling TMA per pos;
4. satu LightGBM global dengan `nama_pos` sebagai fitur kategorikal;
5. recursive forecasting selama delapan bulan;
6. Tukey-fence clipping per pos untuk mengurangi drift.

Hasil validasi recursive pipeline awal:

| Varian | RMSE recursive |
|---|---:|
| Tanpa clipping | 2,08 |
| Dengan clipping | 1,54 |
| Dengan clipping + fitur hulu-hilir | **1,47** |

Validasi recursive diperlukan karena evaluasi one-step menggunakan lag TMA asli
dan dapat terlihat terlalu optimistis. Pada forecasting sebenarnya, prediksi
langkah sebelumnya menjadi input untuk langkah berikutnya sehingga error dapat
terakumulasi.

### 2. CatBoost direct/non-recursive

Eksperimen CatBoost dikembangkan di `tma_pipeline_v2_notebook.ipynb` dan
`notebook-catboost v2.ipynb`. Perubahan utamanya:

- prediksi direct dari fitur eksogen sehingga tidak ada recursive target drift;
- normalisasi robust target per pos menggunakan median dan skala berbasis IQR;
- CatBoost dengan Huber loss untuk mengurangi pengaruh outlier;
- isolated sensor spike dideteksi secara lokal dan diberi sample weight 0,05;
- fitur cuaca rolling, antecedent precipitation index, tanah, iklim, koordinat,
  atribut sungai, serta agregasi hulu;
- ensemble tiga seed: 17, 41, dan 83;
- validasi leakage-safe dengan empat rolling-origin fold.

Fold validasi yang digunakan:

| Fold | Train sampai | Periode validasi |
|---|---|---|
| `sep_2023` | 18 Sep 2023 | 19 Sep 2023–18 Mei 2024 |
| `may_2024` | 18 Mei 2024 | 19 Mei 2024–18 Jan 2025 |
| `sep_2024` | 18 Sep 2024 | 19 Sep 2024–18 Mei 2025 |
| `jan_2025` | 18 Jan 2025 | 19 Jan 2025–18 Sep 2025 |

Dua fold September–Mei paling dekat dengan musim dan panjang horizon test.

### 3. Pemilihan tree berbasis RMSE dan state anchor

`run_catboost_rmse_anchor_experiment.py` memakai feature engineering CatBoost
yang sama agar perubahan model dapat diukur secara terkontrol. Eksperimen ini:

1. membandingkan pemilihan tree berdasarkan normalized MAE, normalized RMSE,
   dan RMSE pada skala TMA asli;
2. memilih jumlah tree final dari median best iteration antar-fold;
3. menambahkan koreksi berdasarkan anomali TMA terakhir setiap pos terhadap
   baseline musimannya;
4. meluruhkan koreksi secara eksponensial tanpa memasukkan prediksi kembali ke
   model.

Konfigurasi anchor terpilih adalah `alpha=0.75` dan `tau_days=365`. Jumlah tree
final adalah 655, lalu tiga seed dirata-ratakan. Konfigurasi dipilih hanya jika
mean RMSE membaik, kedua fold September–Mei tidak memburuk, dan tidak ada stress
fold yang memburuk.

### 4. Eksperimen blend CatBoost–XGBoost

`train_blend.py` membandingkan CatBoost dan XGBoost pada OOF prediction. Pada
pooled OOF, bobot optimal terbatasi memilih **100% CatBoost dan 0% XGBoost**:

| Model | Pooled RMSE, dua fold |
|---|---:|
| CatBoost | **2,1115** |
| XGBoost | 2,2989 |
| Blend OOF | **2,1115** |

Blend yang dituning hanya pada fold terbaru memang memperbaiki fold
`sep_2024`, dari 1,2377 menjadi 1,2027, tetapi memburukkan `sep_2023` dan pooled
RMSE. Karena itu blend tersebut dianggap eksperimental dan tidak dipilih sebagai
submission utama.

### 5. Eksperimen direct LightGBM residual

`run_lightgbm_residual_experiment.py` memakai feature table dan empat fold yang
sama dengan CatBoost. Baseline musiman `nama_pos x bulan x jam` dihitung hanya
dari bagian train setiap fold, lalu LightGBM memprediksi residual secara direct.
Residual mentah mengungguli residual yang diskalakan, tetapi masih kalah dari
CatBoost-anchor sebagai model tunggal:

| Model | Mean RMSE 4 fold | Pooled RMSE |
|---|---:|---:|
| CatBoost RMSE + state anchor | **1,4466** | 1,6144 |
| LightGBM raw residual | 1,5550 | 1,7045 |
| LightGBM scaled residual | 1,5645 | 1,7145 |

Walaupun demikian, optimum pooled OOF memberi bobot 13,60% LightGBM dan 86,40%
CatBoost. Blend ini mencapai mean fold RMSE 1,4390 dan pooled RMSE 1,6121. Fold
`sep_2023` memburuk 0,0064, sehingga kandidat ini tetap lebih agresif daripada
aturan seleksi konservatif yang memilih 100% CatBoost.

Pada evaluator `API_TEST`, blend memperoleh RMSE API 0,8448 dan estimasi RMSE
kompetisi 1,5931, dibanding estimasi 1,6025 untuk CatBoost-anchor. Kalibrasi API
hanya berdasarkan tujuh submission dengan residual RMSE 0,01482, sehingga hasil
ini diperlakukan sebagai estimasi, bukan pengganti skor leaderboard.

### 6. Eksperimen station-aware anchor dan local trend

`run_station_aware_experiment.py` menguji anchor yang di-shrink per pos, Ridge
residual lokal dengan tren waktu, dan stacking CatBoost-LightGBM. Station anchor
memperbaiki seluruh fold dan menurunkan pooled OOF dari 1,6144 menjadi 1,6092.
Ridge lokal tidak kompetitif dan mendapat bobot akhir 0%. Stack OOF terpilih
adalah 81% station-anchor dan 19% LightGBM, dengan pooled RMSE 1,6083.

Pada `API_TEST`, stack tersebut memperoleh RMSE API 0,8439 dan estimasi kompetisi
**1,5926**. Bobot LightGBM per pos tampak lebih baik di OOF (1,6054), tetapi
memburuk menjadi estimasi 1,5966 di API_TEST sehingga ditolak sebagai
meta-overfit. Target estimasi 1,56 belum tercapai.

### 7. Eksperimen CatBoost recency-weighted

`run_catboost_recency_experiment.py` menguji apakah data terbaru perlu diberi
bobot lebih besar untuk menghadapi perubahan pola antarperiode. Empat expert
diuji pada rolling-origin fold yang sama: exponential half-life 180, 365, dan
730 hari, serta jendela keras 540 hari. Koreksi station-anchor dibuat identik
dengan kontrol agar perubahan yang diukur hanya berasal dari recency weighting.

Varian standalone terbaik adalah half-life 180 hari. Seleksi OOF konservatif
memilih blend 81% expert tersebut dan 19% station-anchor kontrol:

| Model | Mean RMSE 4 fold | Pooled RMSE | Estimasi API_TEST |
|---|---:|---:|---:|
| Station-anchor kontrol | 1,4395 | 1,6092 | 1,6053 |
| Half-life 180 hari | 1,4294 | 1,6065 | 1,6185 |
| Blend recency terpilih | **1,4288** | **1,6053** | 1,6144 |
| Station-stack sebelumnya | 1,4323 | 1,6083 | **1,5926** |

Blend recency memperbaiki tiga fold, tetapi memburukkan `sep_2023` sebesar
0,0077. Pada `API_TEST`, RMSE API-nya 0,8871 dan estimasi kompetisinya 1,6144,
lebih buruk daripada station-anchor maupun station-stack. Kandidat ini ditolak:
peningkatan OOF tidak berpindah ke periode test dan memberi bukti bahwa recency
weighting agresif tidak cukup stabil untuk submission utama.

## Ringkasan hasil CatBoost

| Model | Mean RMSE 4 fold | Pooled RMSE | Keputusan |
|---|---:|---:|---|
| CatBoost v2 | 1,4632 | 1,6391 | Kontrol konservatif |
| CatBoost, raw-RMSE tree selection | 1,4817 | 1,6414 | Tidak dipakai sendiri |
| **CatBoost RMSE + state anchor** | **1,4466** | **1,6144** | Kandidat utama |

Perbandingan per fold:

| Fold | CatBoost v2 | RMSE-selected | RMSE + anchor |
|---|---:|---:|---:|
| `sep_2023` | 2,6852 | 2,6545 | **2,6247** |
| `may_2024` | 0,7839 | 0,8591 | **0,7527** |
| `sep_2024` | **1,2239** | 1,2261 | 1,2244 |
| `jan_2025` | **1,1598** | 1,1873 | 1,1848 |

State anchor memperbaiki mean RMSE sekitar 1,13% dan pooled RMSE sekitar 1,50%
dibanding CatBoost v2. Peningkatannya tidak seragam di semua fold, sehingga skor
leaderboard tetap diperlukan sebagai konfirmasi eksternal.

## Struktur proyek

```text
sebelas-maret-statistics-data-science-2026/
├── data_pendukung/
│   ├── HydroRIVERS_v10_au_shp/
│   ├── data_lingkungan.csv
│   ├── HydroRIVERS_TechDoc_v10.pdf
│   └── koordinat_pos.csv
├── train.csv
├── test.csv
├── sample_submission.csv
├── tma_pipeline_notebook.ipynb
├── tma_pipeline_v2_notebook.ipynb
├── notebook-catboost v2.ipynb
├── compare_submission.ipynb
├── train_blend.py
├── run_catboost_rmse_anchor_experiment.py
├── requirements.txt
└── README.md
```

Dataset, feature cache, model binary, dan hasil submission sengaja tidak
disimpan di Git. Folder `output/`, `output_v2/`, `output_blend/`,
`output_catboost_experiments/`, serta `catboost_info/` sudah tercantum di
`.gitignore`.

## Setup environment

```bash
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows PowerShell
venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Letakkan `train.csv`, `test.csv`, `sample_submission.csv`, dan folder
`data_pendukung/` pada root proyek sebelum menjalankan pipeline.

## Menjalankan eksperimen

Baseline LightGBM dan notebook CatBoost dapat dijalankan dengan membuka notebook
terkait lalu memilih **Run All**.

Eksperimen blend:

```bash
python train_blend.py
```

Eksperimen CatBoost RMSE + anchor:

```bash
python run_catboost_rmse_anchor_experiment.py
```

Feature table dan model validasi akan di-cache di
`output_catboost_experiments/`. Anchor dapat dituning ulang dari OOF dan base
submission yang sudah ada tanpa melatih ulang model:

```bash
python run_catboost_rmse_anchor_experiment.py --retune-anchor-only
```

## Rencana eksperimen berikutnya

### Prioritas 1 — Direct LightGBM residual

Eksperimen berikutnya yang paling direkomendasikan:

1. hitung baseline musiman secara leakage-safe dari kombinasi
   `nama_pos × bulan × jam` pada train fold;
2. ubah target menjadi `TMA - baseline_musiman`;
3. latih LightGBM global direct pada feature table CatBoost yang sama;
4. kembalikan prediksi ke skala asli dengan menambahkan baseline;
5. cari bobot blend LightGBM–CatBoost hanya dari OOF prediction.

Pendekatan residual diharapkan membuat model fokus pada perubahan TMA, bukan
perbedaan elevasi dasar antarpos. LightGBM direct juga dapat memberi pola error
yang berbeda dari CatBoost tanpa risiko recursive drift.

### Prioritas 2 — Global/local station hybrid

Bandingkan satu model global dengan model per pos atau per kelompok pos. Model
lokal dipakai hanya ketika jumlah sampel historis memadai; model global menjadi
fallback. Bobot global/local ditentukan dari OOF, bukan dari test prediction.

### Prioritas 3 — State-space per pos

Gunakan structural time-series/Kalman filter untuk memodelkan local level, tren,
seasonality, dan beberapa fitur eksogen terpilih. Hasil state-space lebih cocok
sebagai komponen blend atau baseline residual daripada pengganti tunggal
CatBoost. Eksperimen ini relevan karena state anchor sudah menunjukkan bahwa
kondisi terakhir setiap pos mengandung sinyal.

### Prioritas 4 — ExtraTrees residual

ExtraTrees dapat dicoba pada target residual untuk memperoleh model dengan error
yang lebih beragam. Fokus utamanya bukan mengalahkan CatBoost secara standalone,
tetapi memberi peningkatan saat di-blend berdasarkan OOF.

### Eksperimen jangka lanjut

- Temporal Fusion Transformer, LSTM/GRU, atau model deep forecasting lain setelah
  baseline tree dan state-space matang;
- model graf sungai/GNN untuk merepresentasikan hubungan 30 pos secara eksplisit;
- hyperparameter tuning terkontrol setelah desain target dan validasi stabil.

Deep learning belum menjadi prioritas karena jumlah seri hanya 30, horizon test
panjang, dan risiko overfit serta biaya eksperimennya lebih tinggi.

## Aturan evaluasi eksperimen baru

Semua kandidat baru harus:

1. memakai empat rolling-origin fold yang sama;
2. melaporkan RMSE skala TMA asli per fold, mean fold, dan pooled RMSE;
3. memberi perhatian khusus pada dua fold September–Mei;
4. membentuk blend hanya dari OOF prediction dengan bobot terbatasi;
5. menolak kandidat yang hanya menang pada satu fold tetapi memburukkan pooled
   RMSE atau stress fold secara berarti;
6. memeriksa NaN, infinity, duplikasi ID, rentang prediksi per pos, dan kesesuaian
   urutan dengan `sample_submission.csv` sebelum submit;
7. mencatat skor public leaderboard tanpa menggunakannya untuk tuning berulang.

Pos yang tetap perlu dianalisis secara khusus berdasarkan pipeline awal adalah
`Bojonegoro - Kali Kethek`, `Wonogiri Dam`, `Cepu`, dan `Jurug`.
