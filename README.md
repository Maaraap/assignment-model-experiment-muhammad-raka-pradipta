# AI Model Experiment & Evaluation

**Individual Assignment — AI Engineering Bootcamp**

Project ini membandingkan dua pendekatan untuk **sentiment analysis** ulasan pelanggan e-commerce:

1. **Classical Machine Learning:** TF-IDF + Logistic Regression menggunakan Scikit-learn.
2. **LLM API:** Gemini API menggunakan zero-shot prompting.

## 1. Problem Statement

Tujuan eksperimen adalah mengklasifikasikan ulasan pelanggan menjadi:

- `positif`
- `negatif`

Kemudian membandingkan kedua pendekatan berdasarkan Accuracy, Precision, Recall, F1-Score, confusion matrix, effort implementasi, kecepatan inference, biaya, dan keterbatasan.

## 2. Dataset

Dataset:

```text
data/customer_reviews_sentiment.csv
```

Dataset berisi **200 baris** dan **4 kolom**:

| Column | Description |
|---|---|
| `review_id` | ID review |
| `product_name` | Nama produk |
| `review_text` | Isi review pelanggan |
| `sentiment` | Label sentimen |

### Data Leakage Prevention

Dataset memiliki 200 baris tetapi hanya **40 `review_text` unik**. Banyak review muncul berulang. Random row-level split dapat menyebabkan review yang sama masuk ke train dan test sehingga terjadi data leakage.

Karena itu digunakan **group-aware holdout berdasarkan `review_text` unik**:

- 80% unique review groups → training
- 20% unique review groups → testing
- `random_state = 42`
- Stratified berdasarkan sentiment
- Tidak ada `review_text` yang sama di train dan test

Hasil:

- Training: **158 rows**
- Test: **42 rows**
- Test set: **8 unique review texts**

Kedua model menggunakan test set yang sama.

## 3. Experiment Design

```text
Dataset
   ↓
Group-aware Train/Test Split
   ↓
 ┌───────────────────┐
 │                   │
 ▼                   ▼
Classical ML      Gemini API
TF-IDF             Zero-shot
Logistic Reg.      Prompting
 │                   │
 └─────────┬─────────┘
           ▼
      Evaluation
           ▼
    Model Comparison
           ▼
 Technical Recommendation
```

## 4. Approach 1 — Classical Machine Learning

Pipeline:

```text
Review Text
    ↓
TF-IDF Vectorization
    ↓
Logistic Regression
    ↓
Sentiment Prediction
```

Parameter utama:

- `TfidfVectorizer(ngram_range=(1,2), lowercase=True)`
- `LogisticRegression(max_iter=1000, random_state=42)`

## 5. Approach 2 — Gemini API

Gemini digunakan tanpa proses training melalui **zero-shot prompting**.

Konfigurasi:

- Model: `gemini-3.5-flash`
- Prompt: zero-shot
- Temperature: `0.0`
- Max output tokens: `5`

API key disimpan menggunakan **Google Colab Secrets** dengan nama:

```text
GEMINI_API_KEY
```

> Jangan memasukkan API key langsung ke source code atau GitHub.

## 6. Evaluation Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 69.05% | 58.06% | 100.00% | 73.47% |
| **Gemini API** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |

### Classical ML Confusion Matrix

```text
                  Predicted
                  Negatif  Positif
Actual Negatif       11       13
       Positif        0       18
```

Model berhasil menemukan seluruh review positif, tetapi menghasilkan **13 false positive** pada review negatif. Hal ini membuat precision kelas positif hanya **58.06%**.

### Gemini Confusion Matrix

```text
                  Predicted
                  Negatif  Positif
Actual Negatif       24        0
       Positif        0       18
```

Pada test set ini, Gemini mengklasifikasikan seluruh **42 baris** dengan benar.

## 7. Trade-off dan Limitation

### Classical ML

**Kelebihan**
- Tidak membutuhkan API eksternal saat inference.
- Biaya inference relatif rendah setelah training.
- Dapat dijalankan secara lokal.
- Relatif mudah direproduksi.

**Keterbatasan**
- Membutuhkan preprocessing dan training.
- Performa bergantung pada data training.
- Pemahaman konteks bahasa lebih terbatas.
- Pada eksperimen ini menghasilkan cukup banyak false positive.

### Gemini API

**Kelebihan**
- Tidak membutuhkan proses training.
- Implementasi cepat melalui prompting.
- Fleksibel terhadap variasi bahasa.
- Memberikan performa tertinggi pada eksperimen ini.

**Keterbatasan**
- Membutuhkan API dan koneksi internet.
- Memiliki potensi biaya penggunaan dan network latency.
- Output perlu divalidasi.
- Hasil dapat bergantung pada model, prompt, dan parameter.

## 8. Technical Recommendation

Berdasarkan hasil eksperimen, **Gemini API merupakan pendekatan yang paling unggul dari sisi performa pada dataset ini**, dengan Accuracy, Precision, Recall, dan F1-Score masing-masing **100%**.

Namun, hasil ini **tidak berarti Gemini selalu lebih baik**. Dataset hanya memiliki **40 review unik** dan test set hanya **8 review unik (42 baris)**. Dengan ukuran data unik yang kecil, perubahan beberapa prediksi dapat mengubah metrik secara signifikan.

Untuk use case ini, Gemini layak dipilih sebagai pendekatan awal/prototype karena memberikan performa terbaik tanpa training. Sebelum production, perlu dilakukan validasi menggunakan dataset yang lebih besar dan beragam serta mengukur **biaya API, latency, dan reliability**.

Jika prioritas utama production adalah biaya inference rendah, latency yang konsisten, dan tidak bergantung pada layanan eksternal, model classical ML tetap menjadi alternatif yang menarik setelah dilakukan perbaikan data dan model.

## 9. How to Run

Notebook utama:

```text
notebook/experiment_notebook.ipynb
```

Notebook saat ini disiapkan untuk **Google Colab**.

### Step 1 — Dataset

Upload atau gunakan:

```text
data/customer_reviews_sentiment.csv
```

### Step 2 — Classical ML

Jalankan notebook sampai bagian **Classical ML** selesai untuk melatih dan mengevaluasi TF-IDF + Logistic Regression.

### Step 3 — Gemini API

Tambahkan API key ke **Colab Secrets**:

```text
GEMINI_API_KEY
```

Kemudian jalankan bagian **Gemini API**.

### Step 4 — Evaluation

Jalankan bagian **Evaluation and Comparison** untuk menghasilkan:

- Accuracy
- Precision
- Recall
- F1-Score
- Classification report
- Confusion matrix
- Comparison table


## 10. Technologies

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- Google Gemini API
- Google Colab
- Jupyter Notebook
- GitHub


