import pandas as pd
import numpy as np
import faiss
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Flatten, Concatenate, Input
from tensorflow.keras.models import Model
from fastapi import FastAPI
import streamlit as st
import uvicorn

nltk.download('punkt')

# ============ 🔹 Simulated YouTube Data 🔹 ============
data = {
    "video_id": range(1, 11),
    "title": [
        "Machine Learning Basics", "Deep Learning with TensorFlow",
        "Introduction to Neural Networks", "How to Train a CNN",
        "Reinforcement Learning Explained", "Advanced Data Science Techniques",
        "Python for Data Analysis", "SQL for Data Science",
        "Time Series Forecasting with LSTMs", "GANs: Generative Adversarial Networks"
    ],
    "description": [
        "Learn the basics of ML algorithms",
        "Build deep neural networks with TensorFlow",
        "Understand artificial neurons and perceptrons",
        "Train a convolutional neural network",
        "Exploring Q-learning and policy gradients",
        "Learn data science techniques like PCA, clustering",
        "Analyze data efficiently using pandas",
        "Master SQL queries for data science applications",
        "Predict future trends using LSTMs",
        "Generate realistic images using GANs"
    ],
    "views": [10000, 15000, 12000, 18000, 25000, 11000, 13000, 9000, 14000, 22000]
}

df = pd.DataFrame(data)

# ============ 🔹 Content-Based Filtering (FAISS + TF-IDF) 🔹 ============
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['description']).toarray()

# FAISS Index for Fast Similarity Search
index = faiss.IndexFlatL2(tfidf_matrix.shape[1])
index.add(tfidf_matrix)

def content_recommend(video_id, top_n=3):
    idx = df[df['video_id'] == video_id].index[0]
    _, indices = index.search(np.array([tfidf_matrix[idx]]), top_n + 1)
    rec_videos = df.iloc[indices[0][1:]][['video_id', 'title']]
    return rec_videos

# ============ 🔹 Collaborative Filtering (FAISS + SVD) 🔹 ============
interaction_matrix = np.random.randint(0, 2, (10, 10))
U, sigma, Vt = np.linalg.svd(interaction_matrix)
user_embeddings = np.dot(U, np.diag(sigma))

# FAISS Index for Collaborative Filtering
cf_index = faiss.IndexFlatL2(user_embeddings.shape[1])
cf_index.add(user_embeddings)

def collaborative_recommend(user_id, top_n=3):
    user_idx = int(user_id.split("_")[1])
    _, indices = cf_index.search(np.array([user_embeddings[user_idx]]), top_n + 1)
    rec_videos = df.iloc[indices[0][1:]][['video_id', 'title']]
    return rec_videos

# ============ 🔹 Hybrid Deep Learning Model (TensorFlow) 🔹 ============
num_users = 10
num_videos = len(df)
embedding_dim = 50

user_input = Input(shape=(1,))
user_embedding = Embedding(num_users, embedding_dim)(user_input)
user_embedding = Flatten()(user_embedding)

video_input = Input(shape=(1,))
video_embedding = Embedding(num_videos, embedding_dim)(video_input)
video_embedding = Flatten()(video_embedding)

concat = Concatenate()([user_embedding, video_embedding])
dense1 = Dense(128, activation="relu")(concat)
dense2 = Dense(64, activation="relu")(dense1)
output = Dense(1, activation="sigmoid")(dense2)

model = Model(inputs=[user_input, video_input], outputs=output)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

train_user_ids = np.random.randint(0, num_users, 1000)
train_video_ids = np.random.randint(0, num_videos, 1000)
train_ratings = np.random.rand(1000)

model.fit([train_user_ids, train_video_ids], train_ratings, epochs=10, batch_size=32, verbose=1)

# ============ 🔹 FastAPI Backend 🔹 ============
app = FastAPI()

@app.get("/recommend/{user_id}")
def get_recommendations(user_id: str):
    cf_recommendations = collaborative_recommend(user_id)
    return cf_recommendations.to_dict(orient='records')

@app.get("/content_recommend/{video_id}")
def get_content_recommendations(video_id: int):
    content_recommendations = content_recommend(video_id)
    return content_recommendations.to_dict(orient='records')

# ============ 🔹 Streamlit Frontend 🔹 ============
def streamlit_app():
    st.title("YouTube Recommendation System")

    user_id = st.selectbox("Select User", [f"User_{i}" for i in range(10)])
    video_id = st.selectbox("Select Video", df['video_id'].tolist())

    if st.button("Get Collaborative Recommendations"):
        cf_recommendations = collaborative_recommend(user_id)
        st.write(cf_recommendations)

    if st.button("Get Content-Based Recommendations"):
        content_recommendations = content_recommend(video_id)
        st.write(content_recommendations)

# ============ 🔹 Run FastAPI Server 🔹 ============
if __name__ == "__main__":
    import threading
    threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")).start()
    streamlit_app()
