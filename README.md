# Tubes 2 IF3270 — CNN & RNN/LSTM Image Captioning

Implementasi **from-scratch** (NumPy) untuk dua tugas pembelajaran mesin:
1. **CNN** — klasifikasi gambar 6 kelas pada dataset Intel Image Classification, dengan perbandingan Conv2D vs LocallyConnected2D dan visualisasi Grad-CAM.
2. **RNN/LSTM** — image captioning pada dataset Flickr8k menggunakan arsitektur *pre-inject* (Show and Tell), dengan perbandingan SimpleRNN vs LSTM, beam search decoder, dan bonus init-inject.

---

## Anggota Kelompok

| No | Nama | NIM | Tugas |
|---|---|---|---|
| 1 | Stanislaus Ardy Bramantyo | 18223057 | Mengerjakan Laporan, Membuat Notebook CNN & RNN/LSTM |
| 2 | Nazwan Siddqi Muttaqin | 18223066 | Mengerjakan Laporan, Melakukan training model, Membuat implementasi CNN, Memfinalisasi Notebook |
| 3 | Matthew Sebastian Kurniawan | 18223096 | Mengerjakan Laporan, Membuat implementasi RNN & LSTM, Mengimplementasikan backward propagation, Mengimplementasikan beam search decoder, Mengimplementasikan init-inject captioning, Mengimplementasikan visualisasi feature map CNN dan Grad-CAM |

---

## Struktur Repository

```
src/
  cnn/
    layers.py        # Implementasi layer CNN (NumPy only)
    model.py         # CNNFromScratch: inferensi dari bobot Keras
    utils.py         # load_image, extract_features
    visualize.py     # Feature map & Grad-CAM
    train.ipynb      # Notebook training & evaluasi CNN
  rnn_lstm/
    layers.py        # Implementasi RNN/LSTM cell (NumPy only)
    model.py         # CaptioningFromScratch: inferensi dari bobot Keras
    preprocess.py    # load_captions, build_vocab, make_sequences
    train.ipynb      # Notebook training & evaluasi RNN/LSTM
data/
  cnn/               # Intel Image Classification (seg_train/, seg_test/)
  rnn_lstm/          # Flickr8k (Images/, captions.txt)
models/
  cnn/               # Model CNN tersimpan (.keras, .h5)
  rnn_lstm/          # Model RNN/LSTM tersimpan (.h5), vocab.json, hasil evaluasi
```

---

## Setup

> **Requirement:** WSL2 (Ubuntu) dengan Python 3.10. TensorFlow 2.x dengan GPU memerlukan Linux — native Windows tidak didukung untuk TF ≥ 2.11.

### 1. Buat virtual environment

```bash
python3.10 -m venv ~/tf_env
source ~/tf_env/bin/activate
pip install --upgrade pip
pip install tensorflow[and-cuda]==2.21 numpy pandas matplotlib scikit-learn nltk Pillow jupyter
```

### 2. Clone repository & siapkan data

```bash
git clone https://github.com/NazwanSM/Tubes2_ML_2_Karyawan_Tetap.git
cd Tubes2_ML_2_Karyawan_Tetap
```

Download dataset lalu letakkan di:
- **[Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)** → `data/cnn/seg_train/seg_train/` dan `data/cnn/seg_test/seg_test/`
- **[Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)** → `data/rnn_lstm/Images/` dan `data/rnn_lstm/captions.txt`

### 3. Download model (opsional)

Model yang sudah ditraining tersedia di Google Drive (sebagian file melebihi batas 100 MB GitHub):

**[Google Drive — Models](https://drive.google.com/drive/folders/1ZUN1KA9vKoBlpxcc7wy69gvPqioT9v35?usp=sharing)**

Letakkan isi folder tersebut ke dalam `models/rnn_lstm/` dan `models/cnn/`.

---

## Cara Menjalankan

### Jalankan Jupyter dari WSL

```bash
source ~/tf_env/bin/activate
cd /mnt/c/Users/<username>/<Path menuju folder repository>/Tubes2_ML_2_Karyawan_Tetap
jupyter notebook --no-browser
```

Lalu buka URL yang muncul di terminal dari browser atau VS Code (via Remote - WSL extension).

### Notebook

| Notebook | Lokasi | Isi |
|---|---|---|
| CNN | `src/cnn/train.ipynb` | Training 16 konfigurasi, evaluasi Macro F1, Conv2D vs LC2D, Grad-CAM |
| RNN/LSTM | `src/rnn_lstm/train.ipynb` | Training 12 konfigurasi, evaluasi BLEU-4 & METEOR, Keras vs Scratch, beam search |

> **Catatan RTX 3050:** Set environment variable berikut **sebelum** import TensorFlow:
> ```python
> import os
> os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'
> ```
