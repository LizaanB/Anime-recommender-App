import streamlit as st
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Title of the app
st.title("Simple Rating Based Recommender System")

# Read the data
@st.cache
def load_data():
    return pd.read_csv("anime.csv")

anime = load_data()

# Display the data
st.subheader("Anime Data")
st.write(anime.head())

# Peeking into the data
st.subheader("Data Shape")
st.write(anime.shape)

st.subheader("Data Description")
st.write(anime.describe())

st.subheader("Missing Values")
st.write(anime.isnull().sum())

# Dropping the Anime that have missing rating
anime_with_rating = anime.copy().dropna(subset=['rating'])

st.subheader("Data after Dropping Missing Ratings")
st.write(anime_with_rating.isnull().sum())
st.write(anime_with_rating.head())

# Build the Recommender System
m = anime_with_rating['members'].quantile(0.75)
C = anime_with_rating['rating'].mean()

def weighted_rating(df, m, C):
    term = df['members'] / (m + df['members'])
    return df['rating'] * term + (1-term) * C

anime_with_rating["weighted_rating"] = anime_with_rating.apply(weighted_rating, axis=1, args=(m, C))

st.subheader("Data with Weighted Ratings")
st.write(anime_with_rating.head())

# Top 10 Romance Comedy Anime
st.subheader("Top 10 Romance Comedy Anime")
filter_genre = (anime_with_rating['genre'].str.contains("romance", case=False) & 
                anime_with_rating['genre'].str.contains("comedy", case=False))
filtered = anime_with_rating[filter_genre].sort_values(by=['weighted_rating'], ascending=False)
st.write(filtered.head(10))

# Top 10 Anime Movies
st.subheader("Top 10 Anime Movies")
filtered_movie = anime_with_rating[anime_with_rating['type'] == "Movie"].sort_values(by=['weighted_rating'], ascending=False)
st.write(filtered_movie.head(10))