# Sistem Rekomendasi Film Menggunakan Collaborative Filtering

# Project Overview
# Sistem rekomendasi film bertujuan membantu pengguna menemukan film yang sesuai dengan preferensi mereka secara otomatis.
# Dengan banyaknya pilihan film, sistem ini dapat meningkatkan pengalaman pengguna dan membantu platform streaming meningkatkan engagement.

# Business Understanding
# Permasalahan: Pengguna sering kesulitan memilih film yang sesuai dengan selera mereka di antara ribuan pilihan.
# Solusi: Membangun sistem rekomendasi yang dapat memberikan saran film secara personal menggunakan pendekatan Collaborative Filtering.

# Data Understanding
# Dataset MovieLens terdiri dari dua file utama:
# - movies.dat: Informasi film (MovieID, Title, Genres)
# - ratings.dat: Data rating yang diberikan user ke film (UserID, MovieID, Rating, Timestamp)


import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


# Data understanding
movies = pd.read_csv(
    'datasets/movies.dat',
    sep='::',
    engine='python',
    names=['MovieID', 'Title', 'Genres'],
    encoding='latin1'
)
ratings = pd.read_csv(
    'datasets/ratings.dat',
    sep='::',
    engine='python',
    names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
    encoding='latin1'
)

print(movies.head())
print(ratings.head())

# Exploratory Data Analysis (EDA)
print("Jumlah film unik:", movies['MovieID'].nunique())
print("Jumlah user unik:", ratings['UserID'].nunique())
print("Jumlah rating:", ratings.shape[0])
print("\nDistribusi rating:")
print(ratings['Rating'].value_counts().sort_index())

# Data Preparation
print("Missing value pada movies:\n", movies.isnull().sum())
print("Missing value pada ratings:\n", ratings.isnull().sum())
print("Duplikasi pada movies:", movies.duplicated().sum())
print("Duplikasi pada ratings:", ratings.duplicated().sum())

movies['MovieID'] = movies['MovieID'].astype(int)
ratings['UserID'] = ratings['UserID'].astype(int)
ratings['MovieID'] = ratings['MovieID'].astype(int)
ratings['Rating'] = ratings['Rating'].astype(int)

# Membagi Data untuk Training dan Validasi
ratings_train, ratings_val = train_test_split(
    ratings, test_size=0.2, random_state=42, shuffle=True
)

print("Jumlah data train:", ratings_train.shape[0])
print("Jumlah data validasi:", ratings_val.shape[0])

# Training Model Collaborative Filtering (Neural Network)
min_rating = ratings['Rating'].min()
max_rating = ratings['Rating'].max()

x_train = ratings_train[['UserID', 'MovieID']].values
x_val = ratings_val[['UserID', 'MovieID']].values

y_train = ratings_train['Rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
y_val = ratings_val['Rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

num_users = ratings['UserID'].max() + 1
num_movies = ratings['MovieID'].max() + 1

class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.movie_embedding = layers.Embedding(
            num_movies,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.movie_bias = layers.Embedding(num_movies, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = tf.reduce_sum(user_vector * movie_vector, axis=1, keepdims=True)
        x = dot_user_movie + user_bias + movie_bias
        return tf.nn.sigmoid(x)

embedding_size = 50
model = RecommenderNet(num_users, num_movies, embedding_size)

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=64,
    epochs=10,
    validation_data=(x_val, y_val)
)

# Visualisasi hasil training
plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model_metrics')
plt.ylabel('root_mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Visualisasi hasil training dalam bentuk teks
train_rmse = history.history['root_mean_squared_error']
val_rmse = history.history['val_root_mean_squared_error']

print("RMSE Training dan Validasi per Epoch:")
print("="*40)
for epoch, (tr, val) in enumerate(zip(train_rmse, val_rmse), 1):
    print(f"Epoch {epoch:2d}: Training RMSE = {tr:.4f} | Validation RMSE = {val:.4f}")

print("\nRMSE Terbaik:")
print(f"Training RMSE terbaik: {min(train_rmse):.4f}")
print(f"Validation RMSE terbaik: {min(val_rmse):.4f}")

# Hasil Visualisasi Loss/RMSE
# Berdasarkan hasil RMSE pada data training dan validasi di setiap epoch, terlihat bahwa nilai RMSE pada data validasi cukup stabil dan tidak terlalu jauh dari nilai RMSE pada data training.
# Hal ini menunjukkan bahwa model memiliki kemampuan generalisasi yang baik dan tidak mengalami overfitting secara signifikan.
# Dengan demikian, model yang dibangun sudah cukup optimal untuk memberikan rekomendasi film kepada pengguna berdasarkan data yang tersedia.

# Sistem Rekomendasi Film Berdasarkan Model
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

# Contoh penggunaan
recommend_movies_for_user(user_id=4, model=model, movies=movies, ratings=ratings, top_n=10)