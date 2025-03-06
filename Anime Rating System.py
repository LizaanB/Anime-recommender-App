import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Set up Streamlit
st.title("Anime Ratings Analysis & Recommender System")

st.markdown("""
Every streaming content has its own viewers and each content has its rating. Viewers leave some good ratings for the content if they like it. But where does it apply? Viewers can spend hours scrolling through hundreds, sometimes thousands of anime's but never getting a content they like. Businesses need to provide suggestions based on viewers likings and needs in order to create a better streaming environment that boosts revenue and increases the time spent on a website.
""")

# Load datasets
anime = pd.read_csv("anime.csv")
rating = pd.read_csv("rating.csv")

# Basic Exploration
st.subheader("Basic Exploration")
st.write(f"Shape of The Anime Dataset: {anime.shape}")
st.write("Glimpse of The Dataset:")
st.write(anime.head())

st.write("Informations About Anime Dataset:")
st.write(anime.info())

st.write("Shape of The Rating Dataset:")
st.write(f"Shape of The Rating Dataset: {rating.shape}")
st.write("Glimpse of The Dataset:")
st.write(rating.head())

st.write("Informations About Rating Dataset:")
st.write(rating.info())

# Dataset Summary
st.subheader("Dataset Summary")
st.write("Summary of The Anime Dataset:")
st.write(anime.describe().T)

st.write("Summary of The Rating Dataset:")
st.write(rating.describe().T)

# Data Preprocessing
anime.dropna(axis=0, inplace=True)
rating.drop_duplicates(keep='first', inplace=True)

# Merge datasets
fulldata = pd.merge(anime, rating, on="anime_id", suffixes=[None, "_user"])
fulldata = fulldata.rename(columns={"rating_user": "user_rating"})

# Visualization
st.subheader("Top Anime Community")
top_anime = fulldata.copy()
top_anime.drop_duplicates(subset="name", keep="first", inplace=True)
top_anime_temp1 = top_anime.sort_values(["members"], ascending=False)

plt.figure(figsize=(20, 8))
p = sns.barplot(x=top_anime_temp1["name"][:14], y=top_anime_temp1["members"], palette="viridis", saturation=1, edgecolor="#1c1c1c", linewidth=2)
p.set_title("\nTop Anime Community\n", fontsize=25)
plt.ylabel("Total Member", fontsize=20)
plt.xlabel("\nAnime Name", fontsize=20)
plt.xticks(rotation=90)
st.pyplot(plt)

# Collaborative Recommender
st.subheader("Collaborative Recommender")
data_pivot = fulldata.pivot_table(index="name", columns="user_id", values="user_rating").fillna(0)
data_matrix = csr_matrix(data_pivot.values)
model_knn = NearestNeighbors(metric="cosine", algorithm="brute")
model_knn.fit(data_matrix)

query_no = np.random.choice(data_pivot.shape[0])
st.write(f"We will find recommendation for {query_no} no anime which is {data_pivot.index[query_no]}.")
distances, indices = model_knn.kneighbors(data_pivot.iloc[query_no, :].values.reshape(1, -1), n_neighbors=6)

no = []
name = []
distance = []
rating = []

for i in range(0, len(distances.flatten())):
    if i == 0:
        st.write(f"Recommendations for {data_pivot.index[query_no]} viewers:\n")
    else:
        no.append(i)
        name.append(data_pivot.index[indices.flatten()[i]])
        distance.append(distances.flatten()[i])
        rating.append(*anime[anime["name"] == data_pivot.index[indices.flatten()[i]]]["rating"].values)

dic = {"No": no, "Anime Name": name, "Rating": rating}
recommendation = pd.DataFrame(data=dic)
recommendation.set_index("No", inplace=True)
st.write(recommendation)

# Content Based Recommender
st.subheader("Content Based Recommender")
tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents="unicode", analyzer="word", token_pattern=r"\w{1,}", ngram_range=(1, 3), stop_words="english")
rec_data = fulldata.copy()
rec_data.drop_duplicates(subset="name", keep="first", inplace=True)
rec_data.reset_index(drop=True, inplace=True)
genres = rec_data["genre"].str.split(", | , | ,").astype(str)
tfv_matrix = tfv.fit_transform(genres)

def get_recommendations(title, cosine_sim):
    idx = rec_data[rec_data['name'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    anime_indices = [i[0] for i in sim_scores]
    return rec_data['name'].iloc[anime_indices]

cosine_sim = cosine_similarity(tfv_matrix, tfv_matrix)
title = st.text_input("Enter an anime title for recommendations:")
if title:
    recommendations = get_recommendations(title, cosine_sim)
    st.write(recommendations)