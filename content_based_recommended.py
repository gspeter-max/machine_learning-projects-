import pandas as pd 

user_path = "/content/drive/MyDrive/youtube_users.csv"
video_path = "/content/drive/MyDrive/youtube_videos.csv"
interactions_path = "/content/drive/MyDrive/youtube_interactions.csv"

user_data = pd.read_csv(user_path)
video_data = pd.read_csv(video_path)
interactions_data = pd.read_csv(interactions_path)
import faiss 
import numpy as np 
from sentence_transformers import SentenceTransformer 
from sklearn.preprocessing import normalize

model = SentenceTransformer('all-MiniLM-L6-v2')

video_data['all_text']  = video_data['title'] + " " + video_data['description']
embiding = model.encode(video_data['all_text'].tolist(), convert_to_numpy= True)

embiddings = normalize(embiding,axis = 1)


M = 32
index = faiss.IndexHNSWFlat(embiddings.shape[1],32)
index.hnsw.efsearch = 50 
# index.hnsw.train(embiddings)
index.add(embiddings) 

faiss.write_index(index, 'faiss_index.faiss')

def find_recommendation(video_datas,top_k): 
    search_embidding = model.encode(video_datas,convert_to_numpy= True)
    search_embidding = normalize(search_embidding,axis = 1)

    index = faiss.read_index('faiss_index.faiss')
    distance , idx = index.search(search_embidding,top_k)
    recommended_data = video_data.loc[idx.flatten()]
    return distance, idx ,recommended_data

video_datas = ['Role across respond plan less']
distance , idx, recommended_data = find_recommendation(video_datas,4)
print(recommended_data)
