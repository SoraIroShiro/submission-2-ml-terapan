# Laporan Proyek: Sistem Rekomendasi Film Menggunakan Collaborative Filtering


## Project Overview

Dalam era digital saat ini, industri hiburan khususnya layanan streaming film menghadapi tantangan besar dalam membantu pengguna menemukan film yang sesuai dengan preferensi mereka. Banyaknya pilihan film yang tersedia justru sering membuat pengguna kebingungan dan mengalami kesulitan dalam menentukan tontonan yang tepat. Hal ini dikenal sebagai masalah **information overload** di domain sistem rekomendasi (recommender system).

Masalah utama yang dihadapi adalah bagaimana memberikan rekomendasi film yang relevan dan personal kepada setiap pengguna, sehingga mereka tidak perlu menghabiskan waktu lama untuk mencari film yang ingin ditonton. Latar belakang pemilihan masalah ini adalah kebutuhan nyata dari platform streaming untuk meningkatkan kepuasan dan engagement pengguna, sekaligus mendorong pengguna agar tetap setia menggunakan layanan mereka.

Dengan membangun sistem rekomendasi film yang efektif, diharapkan pengguna dapat dengan mudah menemukan film yang sesuai dengan selera mereka, sehingga pengalaman menonton menjadi lebih menyenangkan dan efisien. Proyek ini juga berkontribusi pada pengembangan teknologi di bidang data science dan machine learning, khususnya dalam penerapan sistem rekomendasi pada industri hiburan.


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
![Output distribusi data](https://raw.githubusercontent.com/SoraIroShiro/submission-2-ml-terapan/refs/heads/main/dataunderstanding.png)

---

## Data Preparation

Pada tahap ini dilakukan serangkaian proses untuk memastikan data siap digunakan dalam pemodelan sistem rekomendasi, baik untuk content-based filtering maupun collaborative filtering. Berikut tahapan pemrosesan data yang dilakukan secara runtut:

1. **Handling Missing Value**  
   - Dilakukan pengecekan nilai kosong (missing value) pada dataset `movies` dan `ratings` menggunakan `.isnull().sum()`.  
   - Hasil: Tidak ditemukan missing value sehingga tidak diperlukan penghapusan atau imputasi data.

2. **Handling Duplikat**  
   - Dilakukan pengecekan data duplikat pada kedua dataset menggunakan `.duplicated().sum()`.  
   - Hasil: Tidak ditemukan data duplikat sehingga tidak ada penghapusan data.

3. **Edit/Update Data (Tipe Data)**  
   - Mengubah tipe data pada kolom `MovieID`, `UserID`, dan `Rating` menjadi integer agar sesuai kebutuhan pemodelan.

4. **Data Splitting**  
   - Membagi data rating menjadi data training (80%) dan data validasi (20%) secara acak menggunakan `train_test_split`.  
   - Tujuannya untuk mengukur performa model secara adil dan menghindari data leakage.

5. **Feature Engineering: Ekstraksi Fitur Genre dengan TF-IDF**  
   - Membuat kolom baru `Genres_str` pada dataset `movies` dengan mengubah pemisah genre dari `|` menjadi spasi.
   - Melakukan ekstraksi fitur genre menggunakan TF-IDF (`TfidfVectorizer`) dan menghitung kemiripan antar film dengan cosine similarity.
   - Fitur ini digunakan untuk sistem rekomendasi content-based filtering.

6. **Normalisasi Data**  
   - Melakukan normalisasi nilai rating ke rentang [0, 1] sebelum digunakan pada model collaborative filtering berbasis neural network.


---

## Content-Based Filtering (TF-IDF)

Pada pendekatan ini, sistem merekomendasikan film berdasarkan kemiripan konten, yaitu genre film. Genre diubah menjadi representasi numerik menggunakan TF-IDF, lalu dihitung kemiripan antar film menggunakan cosine similarity. Rekomendasi diberikan berdasarkan film yang paling mirip dengan film yang dipilih pengguna.

**Cuplikan kode:**
```python
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

# Contoh output rekomendasi content-based
judul_film = 'titanic (1997)'
print(f"Rekomendasi film mirip dengan '{judul_film}':")
display(recommend_content_based_tfidf(judul_film))
```

**Contoh output:**

![](https://raw.githubusercontent.com/SoraIroShiro/submission-2-ml-terapan/refs/heads/main/cbf-tfidf.png)

```

**Kelebihan:**
- Dapat merekomendasikan film baru yang belum pernah dirating user lain.
- Rekomendasi berbasis fitur konten (genre).

**Kekurangan:**
- Rekomendasi cenderung terbatas pada genre/fitur yang sama.
- Tidak mempertimbangkan preferensi kolektif user lain.

---

## Modeling and Result

### Collaborative Filtering (Neural Network)

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
Epoch  1: Training RMSE = 0.2392 | Validation RMSE = 0.2221
Epoch  2: Training RMSE = 0.2142 | Validation RMSE = 0.2158
Epoch  3: Training RMSE = 0.2040 | Validation RMSE = 0.2130
Epoch  4: Training RMSE = 0.1947 | Validation RMSE = 0.2129
Epoch  5: Training RMSE = 0.1867 | Validation RMSE = 0.2141
Epoch  6: Training RMSE = 0.1808 | Validation RMSE = 0.2159
Epoch  7: Training RMSE = 0.1768 | Validation RMSE = 0.2175
Epoch  8: Training RMSE = 0.1740 | Validation RMSE = 0.2189
Epoch  9: Training RMSE = 0.1720 | Validation RMSE = 0.2199
Epoch 10: Training RMSE = 0.1706 | Validation RMSE = 0.2208

RMSE Terbaik:
Training RMSE terbaik: 0.1706
Validation RMSE terbaik: 0.2129
```

Contoh visualisasi RMSE:
![grafis](https://raw.githubusercontent.com/SoraIroShiro/submission-2-ml-terapan/refs/heads/main/visualisasi%20rmse.png)


**Interpretasi:**  
Nilai RMSE pada data validasi cukup stabil dan tidak terlalu jauh dari data training, menandakan model tidak overfitting dan mampu melakukan generalisasi dengan baik.

---


## Penggunaan Model

Cuplikan kode
```python
def recommend_movies_for_user(user_id, model, movies, ratings, top_n=10):
    # Cari semua MovieID yang sudah dirating user
    movies_rated = ratings[ratings['UserID'] == user_id]['MovieID'].tolist()
    # Cari MovieID yang belum pernah dirating user
    movies_not_rated = movies[~movies['MovieID'].isin(movies_rated)]
    
    # Siapkan data prediksi: pasangan (user_id, movie_id)
    user_movie_array = np.array([[user_id, movie_id] for movie_id in movies_not_rated['MovieID']])
    
    # Prediksi skor dengan model
    ratings_pred = model.predict(user_movie_array, verbose=0).flatten()
    
    # Ambil indeks top_n skor tertinggi
    top_indices = ratings_pred.argsort()[-top_n:][::-1]
    recommended_movie_ids = movies_not_rated.iloc[top_indices]['MovieID'].values
    
    print(f"Rekomendasi untuk UserID: {user_id}")
    print("="*30)
    print("Film dengan rating tertinggi dari user:")
    print("-"*30)
    # Tampilkan 5 film dengan rating tertinggi yang pernah dirating user
    top_user_movies = (
        ratings[ratings['UserID'] == user_id]
        .sort_values(by='Rating', ascending=False)
        .head(5)
        .merge(movies, on='MovieID')
    )
    for row in top_user_movies.itertuples():
        print(f"{row.Title} : {row.Rating}")
    
    print("-"*30)
    print(f"Top {top_n} rekomendasi film:")
    print("-"*30)
    recommended_movies = movies[movies['MovieID'].isin(recommended_movie_ids)]
    for row in recommended_movies.itertuples():
        print(f"{row.Title} : {row.Genres}")

# penggunaan
recommend_movies_for_user(user_id=4, model=model, movies=movies, ratings=ratings, top_n=10)
```
Output:
![grafis](https://raw.githubusercontent.com/SoraIroShiro/submission-2-ml-terapan/refs/heads/main/visualisasi%20rmse.png)


## Kesimpulan



### Komparasi Skema

Pada proyek ini, dua pendekatan utama digunakan:
- **Content-Based Filtering (TF-IDF):** Tidak menggunakan metrik RMSE karena tidak melakukan prediksi rating, melainkan rekomendasi berdasarkan kemiripan konten.
- **Collaborative Filtering (Neural Network):** Menggunakan RMSE sebagai metrik utama.

Berdasarkan hasil RMSE, collaborative filtering mampu memprediksi rating dengan cukup baik (RMSE validasi stabil dan tidak jauh dari training). Content-based filtering tetap berguna untuk merekomendasikan film baru yang belum pernah dirating user lain.

### Hubungan dengan Business Understanding

- **Problem Statement:**  
  Model collaborative filtering berhasil memberikan rekomendasi yang relevan dan personal, sehingga membantu pengguna menemukan film sesuai preferensi mereka tanpa harus mencari manual di antara ribuan pilihan.

- **Goals:**  
  Sistem rekomendasi yang dibangun mampu memberikan saran film secara personal dengan performa prediksi yang baik (RMSE rendah), sehingga meningkatkan kepuasan dan engagement pengguna.

- **Dampak Solusi:**  
  Dengan RMSE yang rendah dan rekomendasi yang relevan, pengguna lebih mudah menemukan film yang sesuai selera. Hal ini berpotensi meningkatkan waktu tonton, loyalitas, dan retensi pengguna pada platform streaming.

### Kesimpulan

Model collaborative filtering yang dibangun telah memenuhi tujuan bisnis, yaitu memberikan rekomendasi film yang relevan dan personal. Hasil evaluasi menunjukkan model mampu melakukan generalisasi dengan baik dan tidak overfitting.  
Content-based filtering juga memberikan nilai tambah dengan mampu merekomendasikan film baru.  
Secara keseluruhan, solusi yang diimplementasikan telah menjawab problem statement dan mencapai goals yang diharapkan.

Model collaborative filtering menunjukkan performa yang baik berdasarkan metrik RMSE, sedangkan content-based filtering dapat memberikan rekomendasi film baru yang serupa secara konten. Untuk pengembangan lebih lanjut, dapat dilakukan penggabungan kedua pendekatan (hybrid recommendation) agar hasil rekomendasi lebih optimal.

---