# Studio 2 — Klasifikasi Sampah · Kelompok 5 / Kulkas LG 2 Pintu

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap            | NIM       | Kontribusi utama |
|----|-------------------------|-----------|------------------|
| 1  | Muhammad Fadhil Hawari  | 123140147 | Training dan validasi model |
| 2  | Muhammad Rafly Yahya Ramadhan    | 123140148 | Tuning model |
| 3  | Muhammad Royhan Alfitra | 123140146 | Arsitektur model dan dokumentasi |

## Ringkasan Pendekatan

Model yang dikirim untuk **Final** adalah *fine-tuning* **ConvNeXt-Tiny pretrained ImageNet** (torchvision, 27,8 juta parameter). Model awal pada Preview 1 dan 2 (CNN *from scratch* `DSCNet`) dijelaskan di bagian "Riwayat Eksperimen" dan sudah tidak dipakai.

- **Arsitektur**: `torchvision.models.convnext_tiny`, kepala klasifikasi diganti `Linear(768, 5)`. Pada inferensi, logits dirata-ratakan dari gambar asli dan gambar *flip* horizontal (*test-time augmentation*). `get_model()` hanya membangun arsitektur, bobot dimuat oleh `evaluate.py`.
- **Preprocessing**: `Resize(256)` → `CenterCrop(224)` → `ToTensor()` → normalisasi mean/std ImageNet. Transform yang sama dipakai saat validasi dan inferensi.
- **Pelatihan**:
  - Optimizer AdamW (lr 2e-4, weight decay 0,05), *warm-up* 1 epoch lalu *cosine decay*, 20 epoch, batch 32, *mixed precision* (AMP), *gradient clipping* 1,0, seed 13.
  - Loss `CrossEntropyLoss` dengan *label smoothing* 0,1.
  - Sampler berbobot ringan antar kelas (bobot ∝ 1/√frekuensi).
  - Augmentasi (hanya train): `RandomResizedCrop` (skala 0,5–1,0), *horizontal flip*, rotasi ±20°, `ColorJitter`, `TrivialAugmentWide`, `RandomErasing`. Augmentasi dipilih agar realistis untuk foto sampah (tanpa *vertical flip* dan rotasi 180° seperti pada model awal).
  - **Model final** dilatih pada **100% train set resmi (2.753 gambar)** dengan bobot **epoch terakhir** (bukan epoch terbaik di satu split). Alasannya: pemilihan epoch pada satu split validasi kecil hanyalah *noise* (lihat hasil validasi silang).
  - Bobot disimpan dalam fp16 (±55,7 MB) agar di bawah batas 100 MB GitHub; `load_state_dict` mengonversinya kembali ke fp32.
- **Data tambahan**: **tidak ada**. Seluruh pelatihan final memakai train set resmi dari folder Drive asisten. Pada percobaan awal (Preview 1–2) data dimuat dari Hugging Face `HCL13/sampah`, yang berjumlah 2.753 gambar dan 5 kelas, sama persis dengan jumlah train set resmi. Dataset itu tidak dipakai sebagai tambahan pada model final.
- **Validasi**: tidak mengandalkan satu split. Selain split lokal 85/15, dipakai *5-fold stratified cross-validation* pada seluruh 2.753 gambar (konfigurasi yang sama dengan model final, bobot epoch terakhir). Sebagai kalibrasi, resep `train_baseline.py` (ResNet-18, 8 epoch) dilatih pada split lokal yang sama.

## Hasil

### Model final (ConvNeXt-Tiny)

| Pengukuran | Macro-F1 | Catatan |
|------------|----------|---------|
| 5-fold CV, *out-of-fold* (n = 2.753) | **0,9623** (CI 95% bootstrap: 0,9549 – 0,9699) | Akurasi 0,9637. Estimasi utama. |
| Satu split lokal 85/15 (n = 413), epoch terbaik | 0,9740 | Optimistis: epoch dipilih dari validasi yang sama. |
| Baseline ResNet-18 pada split lokal yang sama | 0,9300 | Skor resmi baseline pada validation set tersembunyi: 0,9159 (selisih +0,014 → split lokal sedikit lebih mudah). |

Hasil *out-of-fold* per kelas:

| Kelas          | Precision | Recall | F1     |
|----------------|-----------|--------|--------|
| botol_plastik  | 0,9715    | 0,9771 | 0,9743 |
| gelas_plastik  | 0,9511    | 0,9712 | 0,9611 |
| kertas_kardus  | 0,9526    | 0,9370 | 0,9448 |
| saset_kemasan  | 0,9686    | 0,9614 | 0,9650 |
| styrofoam      | 0,9686    | 0,9642 | 0,9664 |

Model final dilatih pada seluruh data sehingga **tidak ada skor validasi bersih** untuknya. Skor `evaluate.py` pada gambar yang ikut training (1,0000) hanya dipakai sebagai uji bahwa model termuat dengan benar, bukan sebagai ukuran performa. Perkiraan kami untuk validation set tersembunyi adalah sekitar 0,95–0,97.

### Skor putaran evaluasi (diumumkan asisten)

| Putaran | Commit | Model | Macro-F1 (diumumkan asisten) |
|---------|--------|-------|------------------------------|
| Preview 1 (14 Okt) | `<ISI hash>` | DSCNet from scratch | <ISI dari Google Sheets> |
| Preview 2 (21 Okt) | `<ISI hash>` | DSCNet from scratch | <ISI dari Google Sheets> |
| Final (25 Okt)     | `<ISI hash>` | ConvNeXt-Tiny | <ISI setelah diumumkan> |

### Catatan eksperimen

- Beralih dari CNN *from scratch* (Macro-F1 lokal ±0,74) ke *backbone* pretrained adalah perubahan terbesar. Dengan ±2.900 gambar, model dari nol sulit menyamai fitur ImageNet.
- Augmentasi model awal (*vertical flip*, rotasi hingga 180° dengan `expand`, *salt & pepper*, *masking* besar) tidak realistis untuk foto sampah dan menghilangkan detail pembeda seperti leher botol. Augmentasi diganti dengan yang lebih wajar.
- Kurva validasi pada split tunggal berfluktuasi ±0,02 antar epoch. Karena itu pemilihan epoch dihindari pada model final dan digantikan dengan validasi silang.
- Pada validasi silang, 100 dari 2.753 gambar salah prediksi, dan 55 di antaranya diprediksi dengan keyakinan tinggi berbeda dari labelnya (p(label) < 0,10). Ini mengindikasikan adanya *label noise* atau gambar ambigu. Daftar kandidat dibuat di `suspects.csv`, namun **kami tidak membersihkan label** pada submission ini karena tidak ada waktu untuk memverifikasinya secara manual.
- Kesalahan tersebar ke semua kelas tanpa satu pasangan dominan. Kelas terlemah adalah `kertas_kardus` (F1 0,945).
- Hal yang belum dicoba: *ensemble*, ukuran input lebih besar, backbone yang lebih besar, pembersihan label.

## Riwayat Eksperimen (model awal, Preview 1–2)

- **Arsitektur**: CNN *from scratch* `DSCNet`, ±1,52 juta parameter, lima blok residual `SiDSCConv` (3→64→128→256→512→512) berisi *depthwise-separable convolution*, MaxPool 2×2, kepala `AdaptiveAvgPool2d(1)` → `Linear(512, 5)`. Input 128×128 tanpa normalisasi mean/std.
- **Pelatihan**: Adam (lr 2e-3), `CosineAnnealingLR`, 80 epoch maksimum dengan *early stopping* (patience 15, berhenti epoch 40, checkpoint terbaik epoch 25). Augmentasi offline 7 versi per gambar. Split validasi lokal 20% (551 gambar).
- **Hasil lokal Preview 1** (epoch 25): akurasi 0,7477 · Macro-F1 0,7410 · MCC 0,6816. Model *overfit* kuat (akurasi train ±99% vs validasi ±70–75%). F1 tertinggi `styrofoam` (0,83), terendah `gelas_plastik` (0,67).
- Notebook: `DL_CNN.ipynb`.

## Cara Reproduksi

Eksperimen dijalankan di Google Colab (GPU T4) dengan train set resmi di `MyDrive/train/<kelas>/*.jpg` dan `evaluate.py` di `MyDrive/evaluate.py`.

1. Buka `studio2_final.ipynb` di Colab, pilih runtime T4 GPU, lalu **Run all**. Notebook memuat data, melatih ConvNeXt-Tiny 20 epoch pada seluruh data, menulis `submission/model.py` dan `submission/model.pth`, lalu menjalankan `evaluate.py` sebagai uji sanity (±12–15 menit).
2. Untuk validasi silang dan kalibrasi baseline, jalankan `studio2_colab.ipynb` (bagian 10 dan 11, ±50 menit).
3. Uji submission:

```bash
python evaluate.py --submission ./submission --data ./my_val
```

`my_val/` berstruktur `<kelas>/*.jpg` dan dibuat dari data train oleh notebook. Karena train set tidak boleh diunggah ulang, folder tersebut tidak disertakan di repo.

Paket yang dibutuhkan: `torch`, `torchvision`, `scikit-learn`, `matplotlib`, `pillow`, `numpy`. Submission sendiri hanya memerlukan `torch`, `torchvision`, dan `pillow`, dan `submission/requirements.txt` dikosongkan.

## Struktur Repositori

```
submission/
  model.py              # get_model() dan get_transform()
  model.pth             # state_dict ConvNeXt-Tiny (fp16, ±55,7 MB), epoch terakhir, 100% data
  requirements.txt      # kosong (tidak ada paket tambahan)
evaluate.py             # salinan skrip evaluasi resmi
studio2_final.ipynb     # pelatihan model final
studio2_colab.ipynb     # eksperimen: split lokal, kalibrasi baseline, 5-fold CV
DL_CNN.ipynb            # model awal (DSCNet), Preview 1-2
```
