⛩️ Anime Ratings 〽️ Analysis & 🤖 Recommender System
Project Banner

Trello Board: https://trello.com/invite/b/678feaa484f8accb36ff7726/ATTI2d03148c835cac678e9cab516919cc0c62AE427D/unsupervised-learning-project

📜 Overview
Streaming platforms host a vast library of anime titles, each with its own audience and ratings. However, viewers often struggle to find content they enjoy, spending hours scrolling through thousands of titles. To address this, businesses need to provide personalized recommendations that align with viewers' preferences. This project aims to analyze anime ratings and build a recommender system to enhance user experience, increase engagement, and boost revenue for streaming platforms.

📊 About the Dataset
The dataset contains 73,516 users' ratings for 12,294 anime titles. Each user can add anime to their completed list and assign a rating. The dataset is a compilation of these ratings and includes the following key information:

Anime ID: Unique identifier for each anime.

Name: Title of the anime.

Genre: Categories or genres the anime belongs to.

Type: Type of anime (e.g., TV, Movie, OVA, etc.).

Episodes: Number of episodes.

Rating: Average rating given by users.

Members: Number of members who added the anime to their list.

🎯 Objectives
Data Analysis:

Explore trends in anime ratings and popularity.

Identify the most popular genres and anime types.

Analyze user behavior and preferences.

Recommender System:

Build a content-based recommender system to suggest anime based on user preferences.

Implement collaborative filtering to provide personalized recommendations.

Insights for Streaming Platforms:

Provide actionable insights to improve content curation and user engagement.

🛠️ Tools & Technologies
Programming Language: Python

Libraries:

Data Analysis: pandas, numpy, matplotlib, seaborn

Recommender System: scikit-learn, surprise, lightfm

Text Processing: nltk, gensim

Data Visualization: plotly, seaborn

Deployment: streamlit, flask (optional)

📂 Project Structure

anime-recommender-system/
├── data/                    # Dataset files
│   ├── anime.csv            # Anime metadata
│   ├── ratings.csv          # User ratings
├── notebooks/               # Jupyter notebooks for analysis and modeling
│   ├── data_cleaning.ipynb  # Data preprocessing
│   ├── exploratory_analysis.ipynb  # EDA
│   ├── recommender_system.ipynb  # Model building
├── src/                     # Source code for the recommender system
│   ├── preprocessing.py     # Data preprocessing functions
│   ├── recommender.py       # Recommender system implementation
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies

🚀 How to Use
Clone the Repository:

git clone https://github.com/your-username/anime-recommender-system.git
cd anime-recommender-system

Install Dependencies:

pip install -r requirements.txt

Run the Analysis:

Open the Jupyter notebooks in the notebooks/ directory to explore the data and build models.

Run the Recommender System:

Use the src/recommender.py script to generate recommendations for a given user or anime.

📈 Key Insights
Most Popular Genres:

Action, Adventure, and Comedy are the most popular genres.

Romance and Fantasy also have a significant following.

Top-Rated Anime:

Anime with high ratings often have a large number of members.

Long-running series like "Naruto" and "One Piece" dominate in popularity.

User Behavior:

Users tend to rate anime they complete, with an average rating of 7.5.

A small percentage of users contribute the majority of ratings.

🤖 Recommender System
Content-Based Filtering
Uses anime metadata (e.g., genre, type, synopsis) to recommend similar titles.

Example: If a user likes "Naruto," the system recommends other action-adventure anime.

Collaborative Filtering
Leverages user ratings to find patterns and recommend anime based on similar users' preferences.

Example: If User A and User B have similar ratings, recommend anime liked by User B to User A.

📝 Future Work
Hybrid Recommender System: Combine content-based and collaborative filtering for better accuracy.

Deployment: Build a web app using streamlit or flask to showcase the recommender system.

Real-Time Recommendations: Integrate with a streaming platform to provide real-time suggestions.

🙏 Acknowledgments
Dataset sourced from Kaggle.

Inspired by the need to improve user experience on streaming platforms.

📧 Contact
For questions or feedback, feel free to reach out:

Your Name: Your Email
Lutho Ntsepe             Luthon@nedbank.co.za

Thabiso Nyokolodi        Thabisony@nedbank.co.za

Richard Marais           RichardMar@nedbank.co.za

Amukelani Khosa          Amukelani.khosa1@fnb.co.za

Susheila Naick           Susheilan@nedbank.co.za

Lizaan Botha             Lizaanb@nedbank.co.za



