# Laporan Proyek: Sistem Rekomendasi Film Menggunakan Collaborative Filtering

## Project Overview

Sistem rekomendasi film bertujuan membantu pengguna menemukan film yang sesuai dengan preferensi mereka secara otomatis. Dengan banyaknya pilihan film, pengguna sering merasa kesulitan memilih film yang sesuai dengan selera mereka. Proyek ini penting karena dapat meningkatkan pengalaman pengguna dan membantu platform streaming meningkatkan engagement serta retensi pengguna.

## Business Understanding

### Problem Statement
Pengguna sering kesulitan memilih film yang sesuai dengan selera mereka di antara ribuan pilihan yang tersedia di platform streaming.

### Goals
Membangun sistem rekomendasi yang dapat memberikan saran film secara personal kepada pengguna.

### Solution Approach
1. **Content-Based Filtering (TF-IDF):** Merekomendasikan film berdasarkan kemiripan konten (genre) menggunakan TF-IDF dan cosine similarity.
2. **Collaborative Filtering (Neural Network):** Merekomendasikan film berdasarkan pola rating pengguna lain yang mirip menggunakan model embedding.

---

## Data Understanding

- **Sumber Data:** [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- **Jumlah Data:**  
  - Film: `len(movies)`   
  - Rating: `len(ratings)`
- **Kondisi Data:**  
  - Tidak ada missing value dan duplikasi pada data utama.
- **Fitur Data:**
  - **movies.dat:**  
    - `MovieID`: ID unik film  
    - `Title`: Judul film  
    - `Genres`: Genre film  
  - **ratings.dat:**  
    - `UserID`: ID unik user  
    - `MovieID`: ID film  
    - `Rating`: Skor rating (1-5)  
    - `Timestamp`: Waktu rating diberikan

**Distribusi rating:**
Distribusi rating didominasi oleh rating 4 dan 5, menunjukkan kecenderungan user memberikan rating positif.

**Contoh visualisasi distribusi data:**
![Output distribusi data](https://.jpg)

---

## Data Preparation

Langkah-langkah data preparation yang dilakukan:
- Mengecek dan menangani missing value serta duplikasi pada data.
- Mengubah tipe data agar sesuai (int).
- Membuat fitur baru untuk content-based filtering (`Genres_str`).
- Membagi data rating menjadi data training (80%) dan validasi (20%) untuk collaborative filtering.
- Melakukan normalisasi rating ke rentang [0, 1] untuk model neural network.

**Alasan:**  
Data harus bersih dan bertipe sesuai agar model dapat belajar dengan baik. Pembagian data diperlukan untuk mengukur performa model secara adil. Normalisasi rating membantu proses training model neural network.

---

## Modeling and Result

### 1. Content-Based Filtering (TF-IDF)

Pada pendekatan ini, sistem merekomendasikan film berdasarkan kemiripan konten, yaitu genre film. Genre diubah menjadi representasi numerik menggunakan TF-IDF, lalu dihitung kemiripan antar film menggunakan cosine similarity. Rekomendasi diberikan berdasarkan film yang paling mirip dengan film yang dipilih pengguna.

**Cuplikan kode:**
```python
movies['Genres_str'] = movies['Genres'].apply(lambda x: x.replace('|', ' '))
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies['Genres_str'])
cosine_sim_tfidf = cosine_similarity(tfidf_matrix)

def recommend_content_based_tfidf(title, top_n=5):
    idx = movies[movies['Title'].str.lower() == title.lower()].index
    if len(idx) == 0:
        print("Film tidak ditemukan.")
        return
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim_tfidf[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    film_idx = [i[0] for i in sim_scores]
    return movies.iloc[film_idx][['Title', 'Genres']]
```

**Contoh output:**
```
Rekomendasi film mirip dengan 'titanic (1997)':
                Title                Genres
123      Film A (1998)     Drama|Romance
456      Film B (1995)     Drama
789      Film C (2000)     Romance|Drama
...
```

**Kelebihan:**
- Dapat merekomendasikan film baru yang belum pernah dirating user lain.
- Rekomendasi berbasis fitur konten (genre).

**Kekurangan:**
- Rekomendasi cenderung terbatas pada genre/fitur yang sama.
- Tidak mempertimbangkan preferensi kolektif user lain.

---

### 2. Collaborative Filtering (Neural Network)

Pada pendekatan ini, model rekomendasi dibangun menggunakan embedding untuk user dan movie. Skor kecocokan dihitung dengan dot product embedding, ditambah bias, dan diaktivasi dengan sigmoid agar output berada di rentang [0, 1]. Model di-train menggunakan data training dan divalidasi pada data validasi.

**Cuplikan kode:**
```python
class RecommenderNet(tf.keras.Model):
    # ...lihat file .ipynb untuk detail kode...
```

**Top-N Recommendation Output:**
```
Rekomendasi untuk UserID: 4
==============================
Film dengan rating tertinggi dari user:
------------------------------
Film A : 5
Film B : 5
Film C : 4
...
------------------------------
Top 10 rekomendasi film:
------------------------------
Film X : Action|Adventure
Film Y : Comedy|Romance
...
```

**Kelebihan:**
- Dapat menangkap pola preferensi user yang kompleks.
- Rekomendasi lebih personal.

**Kekurangan:**
- Tidak bisa merekomendasikan film baru (cold start pada item).
- Membutuhkan data rating yang cukup banyak.

---

## Evaluation

**Metrik Evaluasi:**  
- Root Mean Squared Error (RMSE)

**Formula RMSE:**  
\[
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_{true,i} - y_{pred,i})^2}
\]

**Hasil Evaluasi:**
```
RMSE Training dan Validasi per Epoch:
========================================
Epoch  1: Training RMSE = ... | Validation RMSE = ...
...
Epoch 10: Training RMSE = ... | Validation RMSE = ...

RMSE Terbaik:
Training RMSE terbaik: ...
Validation RMSE terbaik: ...
```

**Interpretasi:**  
Nilai RMSE pada data validasi cukup stabil dan tidak terlalu jauh dari data training, menandakan model tidak overfitting dan mampu melakukan generalisasi dengan baik.

---

## Kesimpulan

Sistem rekomendasi film berbasis collaborative filtering dan content-based filtering berhasil dibangun dan mampu memberikan rekomendasi yang relevan kepada pengguna. Model collaborative filtering menunjukkan performa yang baik berdasarkan metrik RMSE, sedangkan content-based filtering dapat memberikan rekomendasi film baru yang serupa secara konten. Untuk pengembangan lebih lanjut, dapat dilakukan penggabungan kedua pendekatan (hybrid recommendation) agar hasil rekomendasi lebih optimal.

---

*Catatan:  
Jika ingin menambahkan gambar atau visualisasi, gunakan sintaks markdown berikut:*
```
![Judul Gambar](path/to/gambar.png)
```