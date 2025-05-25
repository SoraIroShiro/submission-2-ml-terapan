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
![Output distribusi data](https://raw.githubusercontent.com/SoraIroShiro/submission-2-ml-terapan/refs/heads/main/dataunderstanding.png)

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
![grafis](https://raw.githubusercontent.com/SoraIroShiro/submission-2-ml-terapan/refs/heads/main/dataunderstanding.png)


**Interpretasi:**  
Nilai RMSE pada data validasi cukup stabil dan tidak terlalu jauh dari data training, menandakan model tidak overfitting dan mampu melakukan generalisasi dengan baik.

---

## Kesimpulan

Sistem rekomendasi film berbasis collaborative filtering dan content-based filtering berhasil dibangun dan mampu memberikan rekomendasi yang relevan kepada pengguna. Model collaborative filtering menunjukkan performa yang baik berdasarkan metrik RMSE, sedangkan content-based filtering dapat memberikan rekomendasi film baru yang serupa secara konten. Untuk pengembangan lebih lanjut, dapat dilakukan penggabungan kedua pendekatan (hybrid recommendation) agar hasil rekomendasi lebih optimal.

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
Output
```
Rekomendasi untuk UserID: 4
==============================
Film dengan rating tertinggi dari user:
------------------------------
Hustler, The (1961) : 5
Raiders of the Lost Ark (1981) : 5
Rocky (1976) : 5
Saving Private Ryan (1998) : 5
Star Wars: Episode IV - A New Hope (1977) : 5
------------------------------
Top 10 rekomendasi film:
------------------------------
Shawshank Redemption, The (1994) : Drama
Schindler's List (1993) : Drama|War
Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963) : Sci-Fi|War
Godfather, The (1972) : Action|Crime|Drama
Casablanca (1942) : Drama|Romance|War
Citizen Kane (1941) : Drama
Monty Python and the Holy Grail (1974) : Comedy
Seven Samurai (The Magnificent Seven) (Shichinin no samurai) (1954) : Action|Drama
American Beauty (1999) : Comedy|Drama
Sanjuro (1962) : Action|Adventure
```