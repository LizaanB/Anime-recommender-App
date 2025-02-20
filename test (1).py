# app.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Sample data
data = {
    'title': ['Naruto', 'One Piece', 'Attack on Titan', 'Death Note', 'Fullmetal Alchemist'],
    'description': [
        'Naruto Uzumaki, a young ninja who seeks recognition from his peers and dreams of becoming the Hokage, the village leader.',
        'Follows the adventures of Monkey D. Luffy and his pirate crew in order to find the greatest treasure ever left by the legendary Pirate, Gold Roger.',
        'After his hometown is destroyed and his mother is killed, young Eren Yeager vows to cleanse the earth of the giant humanoid Titans that have brought humanity to the brink of extinction.',
        'An intelligent high school student goes on a secret crusade to eliminate criminals from the world after discovering a notebook capable of killing anyone whose name is written into it.',
        'Two brothers search for a Philosopher\'s Stone after an attempt to revive their deceased mother goes awry and leaves them in damaged physical forms.'
    ]
}
df = pd.DataFrame(data)

# Content-Based Recommendation
def get_recommendations(title, cosine_sim):
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5 recommendations
    anime_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[anime_indices]

# Streamlit App
st.title('Anime Recommender')
anime_title = st.selectbox('Select an anime:', df['title'])
if st.button('Recommend'):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['description'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    recommendations = get_recommendations(anime_title, cosine_sim)
    st.write('**Recommendations:**')
    for anime in recommendations:
        st.write(anime)
