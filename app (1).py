{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "935e0e12-dfd4-4650-b765-ffb747ed5f58",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-02-16 17:23:48.628 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-02-16 17:23:49.030 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\lizaa\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-02-16 17:23:49.030 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-02-16 17:23:49.038 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-02-16 17:23:49.039 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-02-16 17:23:49.041 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-02-16 17:23:49.041 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-02-16 17:23:49.041 Session state does not function when running a script without `streamlit run`\n",
      "2025-02-16 17:23:49.046 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-02-16 17:23:49.047 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-02-16 17:23:49.049 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-02-16 17:23:49.050 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-02-16 17:23:49.051 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-02-16 17:23:49.052 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-02-16 17:23:49.056 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "# app.py\n",
    "import streamlit as st\n",
    "import pandas as pd\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.metrics.pairwise import linear_kernel\n",
    "\n",
    "# Sample data\n",
    "data = {\n",
    "    'title': ['Naruto', 'One Piece', 'Attack on Titan', 'Death Note', 'Fullmetal Alchemist'],\n",
    "    'description': [\n",
    "        'Naruto Uzumaki, a young ninja who seeks recognition from his peers and dreams of becoming the Hokage, the village leader.',\n",
    "        'Follows the adventures of Monkey D. Luffy and his pirate crew in order to find the greatest treasure ever left by the legendary Pirate, Gold Roger.',\n",
    "        'After his hometown is destroyed and his mother is killed, young Eren Yeager vows to cleanse the earth of the giant humanoid Titans that have brought humanity to the brink of extinction.',\n",
    "        'An intelligent high school student goes on a secret crusade to eliminate criminals from the world after discovering a notebook capable of killing anyone whose name is written into it.',\n",
    "        'Two brothers search for a Philosopher\\'s Stone after an attempt to revive their deceased mother goes awry and leaves them in damaged physical forms.'\n",
    "    ]\n",
    "}\n",
    "df = pd.DataFrame(data)\n",
    "\n",
    "# Content-Based Recommendation\n",
    "def get_recommendations(title, cosine_sim):\n",
    "    idx = df[df['title'] == title].index[0]\n",
    "    sim_scores = list(enumerate(cosine_sim[idx]))\n",
    "    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)\n",
    "    sim_scores = sim_scores[1:6]\n",
    "    anime_indices = [i[0] for i in sim_scores]\n",
    "    return df['title'].iloc[anime_indices]\n",
    "\n",
    "# Streamlit App\n",
    "st.title('Anime Recommender')\n",
    "anime_title = st.selectbox('Select an anime:', df['title'])\n",
    "if st.button('Recommend'):\n",
    "    tfidf = TfidfVectorizer(stop_words='english')\n",
    "    tfidf_matrix = tfidf.fit_transform(df['description'])\n",
    "    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)\n",
    "    recommendations = get_recommendations(anime_title, cosine_sim)\n",
    "    st.write('**Recommendations:**')\n",
    "    for anime in recommendations:\n",
    "        st.write(anime)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "f1602fef-db8c-4e48-ab40-0608000a9dc0",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
