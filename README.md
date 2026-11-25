# Evaluasi Model Klasifikasi Deteksi Fraud

Praktikum Data Mining 10, Program Studi Informatika, Universitas Negeri Padang.

Repositori ini berisi kode, dataset sintetis, hasil eksekusi, dan gambar untuk
praktikum evaluasi model klasifikasi. Model yang dipakai adalah Decision Tree
dengan `max_depth=4` dan `class_weight='balanced'`. Seluruh angka pada laporan
dihasilkan oleh `src/evaluasi.py`, bukan diketik manual.

## Ringkasan hasil

| Ukuran | Nilai |
|---|---|
| Jumlah transaksi | 500 |
| Transaksi normal | 484 |
| Transaksi fraud | 16 |
| Proporsi fraud | 3,20 persen |
| Data latih / uji | 350 / 150 |
| Fraud pada data uji | 5 |
| TP / TN / FP / FN | 2 / 113 / 32 / 3 |
| Accuracy | 0,767 |
| Precision fraud | 0,059 |
| Recall fraud | 0,400 |
| F1 fraud | 0,103 |
| Accuracy baseline semua normal | 0,967 |

Hasil ini menunjukkan masalah data tidak seimbang. Baseline yang selalu
memprediksi normal mendapat accuracy 0,967, lebih tinggi daripada model, tetapi
recall fraud-nya nol. Model Decision Tree menemukan 2 dari 5 fraud, dengan 32
transaksi normal yang salah ditandai fraud.

## Latar belakang

Deteksi fraud adalah contoh klasifikasi dengan kelas positif yang jarang.
Accuracy mudah terlihat tinggi karena mayoritas transaksi memang normal. Model
yang selalu menebak normal dapat memperoleh accuracy di atas 95 persen tanpa
mendeteksi satu fraud pun. Karena itu, evaluasi perlu memakai confusion matrix,
precision, recall, dan F1-score, bukan accuracy saja.

Praktikum ini memakai data sintetis transaksi pembayaran digital. Data dibuat
deterministik dengan seed 42 agar hasil dapat direproduksi. Dataset bukan data
transaksi riil dan tidak dipakai untuk klaim bisnis.

## Dataset

Fitur yang dipakai:

| Fitur | Keterangan |
|---|---|
| `amount` | Nominal transaksi dalam satuan simulasi |
| `hour` | Jam transaksi, 0 sampai 23 |
| `new_device` | Perangkat baru, 1 jika ya |
| `foreign_ip` | IP luar negeri, 1 jika ya |
| `isFraud` | Label target, 1 fraud dan 0 normal |

Data dibangkitkan dengan aturan yang sama seperti modul praktikum:

```python
score = -4.5 + 0.45*(amount > 300) + 1.1*new_device + 1.5*foreign_ip + 0.7*(hour < 5)
prob = 1 / (1 + np.exp(-score))
isFraud = rng.binomial(1, prob)
```

Hasil pembangkitan: 484 transaksi normal dan 16 fraud. Data disimpan pada
`data/dataset_fraud.csv`.

## Metode

1. Dataset dibagi menjadi data latih 70 persen dan data uji 30 persen dengan
   `stratify=y`, sehingga proporsi fraud pada kedua bagian sama.
2. Model Decision Tree dilatih dengan `max_depth=4` dan `class_weight='balanced'`.
3. Prediksi dibuat pada data uji yang tidak dipakai saat pelatihan.
4. Confusion matrix dihitung dengan urutan label `[0, 1]`, sehingga susunannya
   `[[TN, FP], [FN, TP]]`.
5. Metrik dihitung manual dari TP, TN, FP, dan FN, lalu dibandingkan dengan
   fungsi scikit-learn.
6. Baseline semua normal dihitung sebagai pembanding.
7. Ambang probabilitas 0,2; 0,4; 0,5; dan 0,7 diuji untuk melihat perubahan
   precision dan recall.

Rumus yang dipakai:

```
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

## Hasil

### Confusion matrix

```
[[113  32]
 [  3   2]]
```

Dengan urutan label `[0, 1]`, nilai tersebut berarti TN=113, FP=32, FN=3, dan
TP=2. Model menangkap 2 fraud, melewatkan 3 fraud, dan memberi alarm palsu pada
32 transaksi normal.

### Metrik

| Metrik | Manual | scikit-learn |
|---|---|---|
| Accuracy | 0,767 | 0,767 |
| Precision | 0,059 | 0,059 |
| Recall | 0,400 | 0,400 |
| F1-score | 0,103 | 0,103 |

Nilai manual dan scikit-learn sama. Perbedaan hanya dapat muncul karena
pembulatan tampilan.

### Baseline semua normal

Baseline memprediksi seluruh 150 data uji sebagai normal. Hasilnya:

| Metrik | Baseline | Decision Tree |
|---|---|---|
| Accuracy | 0,967 | 0,767 |
| Precision fraud | 0,000 | 0,059 |
| Recall fraud | 0,000 | 0,400 |
| F1 fraud | 0,000 | 0,103 |

Baseline lebih tinggi pada accuracy, tetapi tidak menemukan fraud sama sekali.
Accuracy pada data tidak seimbang karena itu tidak cukup untuk menilai model.

### Pengaruh ambang

| Ambang | Precision | Recall | F1 | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|
| 0,2 | 0,059 | 0,400 | 0,103 | 113 | 32 | 3 | 2 |
| 0,4 | 0,059 | 0,400 | 0,103 | 113 | 32 | 3 | 2 |
| 0,5 | 0,059 | 0,400 | 0,103 | 113 | 32 | 3 | 2 |
| 0,7 | 0,056 | 0,200 | 0,087 | 128 | 17 | 4 | 1 |

Ambang 0,2 sampai 0,5 memberi hasil yang sama karena probabilitas keluaran daun
Decision Tree terbatas pada beberapa nilai. Menurunkan ambang tidak otomatis
menaikkan recall pada model ini. Ambang 0,7 justru menurunkan recall menjadi 0,2.

## Jawaban tugas

### Tugas 1: hitung manual

Diketahui TP=24, TN=156, FP=12, dan FN=8.

| Metrik | Perhitungan | Nilai |
|---|---|---|
| Accuracy | (24+156)/200 | 0,900 |
| Precision | 24/(24+12) | 0,667 |
| Recall | 24/(24+8) | 0,750 |
| F1 | 2 x 0,667 x 0,750 / (0,667+0,750) | 0,706 |

FP berarti 12 transaksi normal mendapat alarm fraud, sehingga pelanggan dapat
terganggu. FN berarti 8 fraud lolos dan berpotensi menimbulkan kerugian. Pada
kasus fraud, FN biasanya lebih mahal daripada FP, sehingga recall perlu
diprioritaskan tanpa mengabaikan precision.

### Tugas 2: verifikasi Python

Eksekusi `src/evaluasi.py` menghasilkan dataset 500 baris, pembagian 350 data
latih dan 150 data uji, confusion matrix di atas, serta metrik manual yang sama
dengan scikit-learn. Bukti tersimpan pada `hasil/output_eksperimen.txt` dan
`hasil/hasil_evaluasi.json`.

### Tugas 3: data tidak seimbang

Baseline accuracy 0,967 lebih tinggi daripada model 0,767, tetapi recall fraud
baseline 0,000. Accuracy tinggi berasal dari 145 transaksi normal yang benar,
bukan dari kemampuan mendeteksi fraud. Ini adalah accuracy paradox. Pada data
dengan 16 fraud dari 500 transaksi, metrik yang lebih informatif adalah recall,
precision, F1 kelas fraud, dan kurva precision-recall.

### Tugas 4: ambang dan rekomendasi

Untuk skenario biaya FN lebih tinggi daripada FP, ambang 0,2 dipilih karena
memberi recall tertinggi di antara empat ambang yang diuji, yaitu 0,400, dengan
precision 0,059 dan F1 0,103. Ambang 0,7 tidak dipilih karena recall turun
menjadi 0,200.

Namun hasil ini menunjukkan bahwa penurunan ambang saja belum cukup. Pada
model ini, ambang 0,2 sampai 0,5 menghasilkan prediksi yang sama. Perbaikan
yang lebih berdampak adalah menambah contoh fraud, memakai penyeimbangan kelas
yang lebih kuat, atau mencoba model dan fitur lain. Ambang 0,2 tetap dapat
dipakai sebagai kebijakan awal, dengan catatan precision rendah dan jumlah alarm
palsu masih 32 transaksi.

## Struktur repositori

```
dm-p10-evaluasi-model-klasifikasi/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── dataset_fraud.csv
├── docs/
│   ├── github.md
│   ├── laporan-tugas-10.pdf
│   └── Data Mining_24343016_Muhammad Rafi Fatihul Ihsan_Tugas10.docx
├── src/
│   ├── evaluasi.py
│   └── gambar.py
├── tests/
│   └── test_metrik.py
├── hasil/
│   ├── hasil_evaluasi.json
│   ├── hasil_eksperimen.json
│   ├── output_eksperimen.txt
│   ├── output_tests.txt
│   ├── tabel_metrik.csv
│   └── tabel_threshold.csv
└── gambar/
    ├── arsitektur_pipeline.png
    ├── confusion_matrix.png
    ├── diagram_pohon.png
    ├── distribusi_kelas.png
    ├── kode_dataset.png
    ├── kode_metrik.png
    ├── kode_model.png
    ├── kode_threshold.png
    ├── perbandingan_metrik.png
    └── threshold.png
```

## Instalasi

```bash
python -m pip install -r requirements.txt
```

Isi `requirements.txt`:

```
numpy>=1.26
pandas>=2.0
scikit-learn>=1.4
matplotlib>=3.8
```

## Cara menjalankan

Dari akar repositori:

```bash
python src/evaluasi.py
python src/gambar.py
python -m unittest discover -s tests -v
```

`src/evaluasi.py` menulis ulang dataset, JSON, CSV, dan log keluaran.
`src/gambar.py` membuat empat gambar dari JSON tersebut. Pengujian memeriksa
rumus metrik dan determinisme dataset.

## Contoh keluaran

```
TP=2, TN=113, FP=32, FN=3
Manual   : accuracy=0.767 precision=0.059 recall=0.400 f1=0.103
Sklearn  : accuracy=0.767 precision=0.059 recall=0.400 f1=0.103
Baseline accuracy: 0.9667
```

## Keterbatasan

- Dataset sintetis dan hanya 500 transaksi dengan 16 fraud.
- Hanya satu split data dan satu model, tanpa validasi silang.
- Data uji hanya berisi 5 fraud, sehingga satu prediksi menggeser recall 0,2.
- Ambang 0,2 sampai 0,5 memberi hasil sama karena probabilitas daun Decision
  Tree bersifat kasar.
- Hasil tidak dapat dipakai untuk klaim performa pada data transaksi nyata.

## Referensi

- Han, J., Kamber, M., & Pei, J. (2012). *Data mining: Concepts and techniques*
  (3rd ed.). Morgan Kaufmann.
- He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE
  Transactions on Knowledge and Data Engineering, 21*(9), 1263-1284.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python.
  *Journal of Machine Learning Research, 12*, 2825-2830.
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more
  informative than the ROC plot when evaluating binary classifiers on
  imbalanced datasets. *PLOS ONE, 10*(3), e0118432.
- Tan, P.-N., Steinbach, M., Karpatne, A., & Kumar, V. (2019). *Introduction to
  data mining* (2nd ed.). Pearson.
