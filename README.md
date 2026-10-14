# Studio 2 — Klasifikasi Sampah · Kelompok 5 / Kulkas LG 2 Pintu

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap            | NIM       | Kontribusi utama |
|----|-------------------------|-----------|------------------|
| 1  | Muhammad Fadhil Hawari  | 123140147 | Training dan validasi model |
| 2  | Muhammad Rafly Yahya    | 123140148 | Tuning model |
| 3  | Muhammad Royhan Alfitra | 123140146 | Arsitektur model dan dokumentasi |

## Ringkasan Pendekatan

- **Arsitektur / backbone**: CNN *from scratch* (tanpa pretrained), `DSCNet` dengan ±1,52 juta parameter. Lima blok residual `SiDSCConv` (3→64→128→256→512→512). Tiap blok berisi dua pasang *depthwise-separable convolution* (depthwise 3×3 + pointwise 1×1, masing-masing dengan BatchNorm + ReLU) dan *skip connection* berupa proyeksi 1×1. Antar blok digunakan MaxPool 2×2. Kepala klasifikasi: `AdaptiveAvgPool2d(1)` → `Linear(512, 5)`.
- **Strategi pelatihan**:
  - Optimizer Adam, learning rate awal 2e-3, scheduler `CosineAnnealingLR` (T_max = 80, eta_min = 5e-6), loss `CrossEntropyLoss`.
  - Batch 64 per GPU (2× Tesla T4 via `DataParallel`, total 128), maksimum 80 epoch, *early stopping* dengan patience 15 pada Macro-F1 validasi (berhenti di epoch 40, checkpoint terbaik epoch 25). Seed 13.
  - Augmentasi offline 7 versi per gambar (hanya pada train): asli, horizontal flip, vertical flip, zoom 1,15–1,3×, rotasi acak 0–180° (expand), *salt & pepper noise*, dan *random masking*. Train 2.202 gambar menjadi 15.414 sampel.
- **Data tambahan**: <ISI — lihat catatan di bawah>. Data dimuat dari Hugging Face `HCL13/sampah` (2.753 gambar, 5 kelas) <ISI: apakah ini dataset yang sama dengan train set resmi, atau data tambahan? Sebutkan jumlah dan sumbernya>.
- **Ukuran input & preprocessing**: resize ke 128×128, `ToTensor()` (skala [0, 1]), tanpa normalisasi mean/std. Split validasi lokal: 20% stratified (551 gambar) dari data yang dimuat, disimpan ke `my_val/`.

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) | `<ISI hash>` | 0,7410 | <ISI setelah diumumkan> |
| Preview 2 (21 Okt) | | | |
| Final (25 Okt)     | | | |

Detail evaluasi lokal Preview 1 (epoch 25, 551 gambar validasi): akurasi 0,7477 · Macro-F1 0,7410 · MCC 0,6816 · specificity macro-avg 0,9361.

| Kelas          | Precision | Recall | F1     |
|----------------|-----------|--------|--------|
| botol_plastik  | 0,6777    | 0,7810 | 0,7257 |
| gelas_plastik  | 0,7356    | 0,6154 | 0,6702 |
| kertas_kardus  | 0,7812    | 0,6849 | 0,7299 |
| saset_kemasan  | 0,7162    | 0,7852 | 0,7491 |
| styrofoam      | 0,8397    | 0,8209 | 0,8302 |

Catatan eksperimen singkat (apa yang dicoba, apa yang berhasil/gagal):

- Model *overfit* kuat: akurasi train mencapai ±99% sedangkan akurasi validasi tertahan di ±70–75% dan val loss naik setelah epoch ±25.
- Kelas tersulit adalah `gelas_plastik` (recall terendah) dan `botol_plastik`, kemungkinan karena keduanya sama-sama plastik bening dan sering tertukar.
- Kelas `styrofoam` paling mudah dikenali (F1 0,83).
- <ISI: eksperimen berikutnya, mis. pretrained backbone, regularisasi, ukuran input lebih besar>

## Cara Reproduksi

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute <NAMA_NOTEBOOK>.ipynb   # melatih model, menyimpan submission/model.pth dan my_val/
python evaluate.py --submission ./submission --data ./my_val
```

Paket yang dibutuhkan untuk pelatihan: `torch`, `torchvision`, `scikit-learn`, `matplotlib`, `seaborn`, `datasets`, `pillow`, `numpy`. Submission sendiri hanya memerlukan `torch`, `torchvision`, dan `pillow` (tanpa paket tambahan).

## Struktur Repositori

```
submission/
  model.py          # get_model() dan get_transform()
  model.pth         # state_dict model terbaik (epoch 25)
  requirements.txt  # (opsional) paket tambahan
evaluate.py         # salinan skrip evaluasi resmi
<NAMA_NOTEBOOK>.ipynb  # notebook pelatihan kelompok
training.log        # log pelatihan per epoch
...
```
