import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("movies.csv")

# Ensure correct column names
movies.columns = movies.columns.str.strip().str.lower()

# Handle missing or bad genres
movies['genres'] = movies['genres'].fillna('unknown')
movies['genres'] = movies['genres'].astype(str)

# Replace separators & bad text
movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)
movies['genres'] = movies['genres'].str.replace('(no genres listed)', 'unknown', regex=False)

# TF-IDF (robust settings)
tfidf = TfidfVectorizer(
    stop_words=None,
    token_pattern=r'(?u)\b\w+\b',
    min_df=1
)

tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Similarity
cosine_sim = cosine_similarity(tfidf_matrix)

def recommend(movie_title, n=5):
    if movie_title not in movies['title'].values:
        print("❌ Movie not found")
        return

    idx = movies[movies['title'] == movie_title].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print(f"\n🎬 Movies similar to {movie_title}:\n")
    for i in scores[1:n+1]:
        print(movies.iloc[i[0]]['title'])

# Test
recommend(movies['title'].iloc[0])
