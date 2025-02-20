{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "1cdc9f5b-00dc-4fc4-b3f6-178bdfdfe7a5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting lightfm (from -r requirements.txt (line 1))\n",
      "  Using cached lightfm-1.17.tar.gz (316 kB)\n",
      "  Preparing metadata (setup.py): started\n",
      "  Preparing metadata (setup.py): finished with status 'done'\n",
      "Requirement already satisfied: pandas in c:\\users\\lizaa\\anaconda3\\lib\\site-packages (from -r requirements.txt (line 2)) (2.2.3)\n",
      "Requirement already satisfied: numpy in c:\\users\\lizaa\\anaconda3\\lib\\site-packages (from -r requirements.txt (line 3)) (1.26.0)\n",
      "Requirement already satisfied: matplotlib in c:\\users\\lizaa\\anaconda3\\lib\\site-packages (from -r requirements.txt (line 4)) (3.9.2)\n",
      "Requirement already satisfied: scipy in c:\\users\\lizaa\\anaconda3\\lib\\site-packages (from -r requirements.txt (line 5)) (1.13.1)\n",
      "Requirement already satisfied: requests in c:\\users\\lizaa\\anaconda3\\lib\\site-packages (from lightfm->-r requirements.txt (line 1)) (2.32.3)\n",
      "Requirement already satisfied: scikit-learn in c:\\users\\lizaa\\anaconda3\\lib\\site-packages (from lightfm->-r requirements.txt (line 1)) (1.6.0)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in c:\\users\\lizaa\\anaconda3\\lib\\site-packages (from pandas->-r requirements.txt (line 2)) (2.9.0.post0)\n",
      "Requirement already satisfied: pytz>=2020.1 in c:\\users\\lizaa\\anaconda3\\lib\\site-packages (from pandas->-r requirements.txt (line 2)) (2024.1)\n",
      "Requirement already satisfied: tzdata>=2022.7 in c:\\users\\lizaa\\anaconda3\\lib\\site-packages (from pandas->-r requirements.txt (line 2)) (2023.3)\n",
      "Requirement already satisfied: contourpy>=1.0.1 in c:\\users\\lizaa\\anaconda3\\lib\\site-packages (from matplotlib->-r requirements.txt (line 4)) (1.2.0)\n",
      "Requirement already satisfied: cycler>=0.10 in c:\\users\\lizaa\\anaconda3\\lib\\site-packages (from matplotlib->-r requirements.txt (line 4)) (0.11.0)\n",
      "Requirement already satisfied: fonttools>=4.22.0 in c:\\users\\lizaa\\anaconda3\\lib\\site-packages (from matplotlib->-r requirements.txt (line 4)) (4.51.0)\n",
      "Requirement already satisfied: kiwisolver>=1.3.1 in c:\\users\\lizaa\\anaconda3\\lib\\site-packages (from matplotlib->-r requirements.txt (line 4)) (1.4.4)\n",
      "Requirement already satisfied: packaging>=20.0 in c:\\users\\lizaa\\anaconda3\\lib\\site-packages (from matplotlib->-r requirements.txt (line 4)) (24.1)\n",
      "Requirement already satisfied: pillow>=8 in c:\\users\\lizaa\\anaconda3\\lib\\site-packages (from matplotlib->-r requirements.txt (line 4)) (10.4.0)\n",
      "Requirement already satisfied: pyparsing>=2.3.1 in c:\\users\\lizaa\\anaconda3\\lib\\site-packages (from matplotlib->-r requirements.txt (line 4)) (3.1.2)\n",
      "Requirement already satisfied: six>=1.5 in c:\\users\\lizaa\\anaconda3\\lib\\site-packages (from python-dateutil>=2.8.2->pandas->-r requirements.txt (line 2)) (1.16.0)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\users\\lizaa\\anaconda3\\lib\\site-packages (from requests->lightfm->-r requirements.txt (line 1)) (3.3.2)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\lizaa\\anaconda3\\lib\\site-packages (from requests->lightfm->-r requirements.txt (line 1)) (3.7)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in c:\\users\\lizaa\\anaconda3\\lib\\site-packages (from requests->lightfm->-r requirements.txt (line 1)) (2.2.3)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\lizaa\\anaconda3\\lib\\site-packages (from requests->lightfm->-r requirements.txt (line 1)) (2024.12.14)\n",
      "Requirement already satisfied: joblib>=1.2.0 in c:\\users\\lizaa\\anaconda3\\lib\\site-packages (from scikit-learn->lightfm->-r requirements.txt (line 1)) (1.4.2)\n",
      "Requirement already satisfied: threadpoolctl>=3.1.0 in c:\\users\\lizaa\\anaconda3\\lib\\site-packages (from scikit-learn->lightfm->-r requirements.txt (line 1)) (3.5.0)\n",
      "Building wheels for collected packages: lightfm\n",
      "  Building wheel for lightfm (setup.py): started\n",
      "  Building wheel for lightfm (setup.py): finished with status 'error'\n",
      "  Running setup.py clean for lightfm\n",
      "Failed to build lightfm\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "  error: subprocess-exited-with-error\n",
      "  \n",
      "  python setup.py bdist_wheel did not run successfully.\n",
      "  exit code: 1\n",
      "  \n",
      "  [38 lines of output]\n",
      "  Compiling without OpenMP support.\n",
      "  C:\\Users\\lizaa\\anaconda3\\Lib\\site-packages\\setuptools\\dist.py:452: SetuptoolsDeprecationWarning: Invalid dash-separated options\n",
      "  !!\n",
      "  \n",
      "          ********************************************************************************\n",
      "          Usage of dash-separated 'description-file' will not be supported in future\n",
      "          versions. Please use the underscore name 'description_file' instead.\n",
      "  \n",
      "          This deprecation is overdue, please update your project and remove deprecated\n",
      "          calls to avoid build errors in the future.\n",
      "  \n",
      "          See https://setuptools.pypa.io/en/latest/userguide/declarative_config.html for details.\n",
      "          ********************************************************************************\n",
      "  \n",
      "  !!\n",
      "    opt = self.warn_dash_deprecation(opt, section)\n",
      "  C:\\Users\\lizaa\\anaconda3\\Lib\\site-packages\\setuptools\\_distutils\\dist.py:261: UserWarning: Unknown distribution option: 'tests_require'\n",
      "    warnings.warn(msg)\n",
      "  running bdist_wheel\n",
      "  running build\n",
      "  running build_py\n",
      "  creating build\\lib.win-amd64-cpython-312\\lightfm\n",
      "  copying lightfm\\cross_validation.py -> build\\lib.win-amd64-cpython-312\\lightfm\n",
      "  copying lightfm\\data.py -> build\\lib.win-amd64-cpython-312\\lightfm\n",
      "  copying lightfm\\evaluation.py -> build\\lib.win-amd64-cpython-312\\lightfm\n",
      "  copying lightfm\\lightfm.py -> build\\lib.win-amd64-cpython-312\\lightfm\n",
      "  copying lightfm\\_lightfm_fast.py -> build\\lib.win-amd64-cpython-312\\lightfm\n",
      "  copying lightfm\\__init__.py -> build\\lib.win-amd64-cpython-312\\lightfm\n",
      "  creating build\\lib.win-amd64-cpython-312\\lightfm\\datasets\n",
      "  copying lightfm\\datasets\\movielens.py -> build\\lib.win-amd64-cpython-312\\lightfm\\datasets\n",
      "  copying lightfm\\datasets\\stackexchange.py -> build\\lib.win-amd64-cpython-312\\lightfm\\datasets\n",
      "  copying lightfm\\datasets\\_common.py -> build\\lib.win-amd64-cpython-312\\lightfm\\datasets\n",
      "  copying lightfm\\datasets\\__init__.py -> build\\lib.win-amd64-cpython-312\\lightfm\\datasets\n",
      "  copying lightfm\\_lightfm_fast_no_openmp.c -> build\\lib.win-amd64-cpython-312\\lightfm\n",
      "  copying lightfm\\_lightfm_fast_openmp.c -> build\\lib.win-amd64-cpython-312\\lightfm\n",
      "  running build_ext\n",
      "  building 'lightfm._lightfm_fast_no_openmp' extension\n",
      "  error: Microsoft Visual C++ 14.0 or greater is required. Get it with \"Microsoft C++ Build Tools\": https://visualstudio.microsoft.com/visual-cpp-build-tools/\n",
      "  [end of output]\n",
      "  \n",
      "  note: This error originates from a subprocess, and is likely not a problem with pip.\n",
      "  ERROR: Failed building wheel for lightfm\n",
      "  error: subprocess-exited-with-error\n",
      "  \n",
      "  python setup.py clean did not run successfully.\n",
      "  exit code: 1\n",
      "  \n",
      "  [20 lines of output]\n",
      "  Compiling without OpenMP support.\n",
      "  C:\\Users\\lizaa\\anaconda3\\Lib\\site-packages\\setuptools\\dist.py:452: SetuptoolsDeprecationWarning: Invalid dash-separated options\n",
      "  !!\n",
      "  \n",
      "          ********************************************************************************\n",
      "          Usage of dash-separated 'description-file' will not be supported in future\n",
      "          versions. Please use the underscore name 'description_file' instead.\n",
      "  \n",
      "          This deprecation is overdue, please update your project and remove deprecated\n",
      "          calls to avoid build errors in the future.\n",
      "  \n",
      "          See https://setuptools.pypa.io/en/latest/userguide/declarative_config.html for details.\n",
      "          ********************************************************************************\n",
      "  \n",
      "  !!\n",
      "    opt = self.warn_dash_deprecation(opt, section)\n",
      "  C:\\Users\\lizaa\\anaconda3\\Lib\\site-packages\\setuptools\\_distutils\\dist.py:261: UserWarning: Unknown distribution option: 'tests_require'\n",
      "    warnings.warn(msg)\n",
      "  running clean\n",
      "  error: [WinError 2] The system cannot find the file specified\n",
      "  [end of output]\n",
      "  \n",
      "  note: This error originates from a subprocess, and is likely not a problem with pip.\n",
      "  ERROR: Failed cleaning build dir for lightfm\n",
      "ERROR: Failed to build installable wheels for some pyproject.toml based projects (lightfm)\n"
     ]
    }
   ],
   "source": [
    "pip install -r requirements.txt\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "de0e618d-39f4-4303-a671-a09d4e6d5b82",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: pip in c:\\users\\lizaa\\anaconda3\\lib\\site-packages (25.0.1)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install --upgrade pip\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "26240cba-5da7-4ea9-bda8-746205b8da44",
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid syntax (62395007.py, line 1)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  Cell \u001b[1;32mIn[3], line 1\u001b[1;36m\u001b[0m\n\u001b[1;33m    streamlit==1.3.0\u001b[0m\n\u001b[1;37m                  ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m invalid syntax\n"
     ]
    }
   ],
   "source": [
    "streamlit==1.3.0\n",
    "pandas==1.3.3\n",
    "scikit-learn==0.24.2\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "134845eb-95cd-4054-8a7d-e532cf593ea3",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-02-20 18:51:45.064 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-02-20 18:51:45.550 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\lizaa\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-02-20 18:51:45.555 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-02-20 18:51:45.560 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-02-20 18:51:45.561 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-02-20 18:51:45.563 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-02-20 18:51:45.564 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-02-20 18:51:45.565 Session state does not function when running a script without `streamlit run`\n",
      "2025-02-20 18:51:45.566 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-02-20 18:51:45.567 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-02-20 18:51:45.570 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-02-20 18:51:45.571 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-02-20 18:51:45.571 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-02-20 18:51:45.572 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-02-20 18:51:45.574 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
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
    "    sim_scores = sim_scores[1:6]  # Top 5 recommendations\n",
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
    "        st.write(anime)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "eac08543-c1b3-4ba4-b625-aeb01ea637d9",
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
