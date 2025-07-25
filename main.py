# Movie Recommendation System - Content Based using TF-IDF

# Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Step 2: Load the datasets
movies = pd.read_csv('/content/movies.csv')
credits = pd.read_csv('/content/credits.csv')

# Step 3: Merge the datasets on title
movies = movies.merge(credits, on='title')

# Step 4: Select and preprocess the relevant features
def extract_names(text):
    try:
        return [d['name'] for d in ast.literal_eval(text)]
    except:
        return []

def extract_director(crew):
    try:
        for person in ast.literal_eval(crew):
            if person['job'] == 'Director':
                return person['name']
    except:
        return ''
    return ''

movies['genres'] = movies['genres'].apply(extract_names)
movies['cast'] = movies['cast'].apply(lambda x: extract_names(x)[:3])  # Top 3 cast members
movies['crew'] = movies['crew'].apply(extract_director)
movies['keywords'] = movies['keywords'].apply(extract_names)
movies['overview'] = movies['overview'].fillna('')

# Step 5: Create a combined metadata column
movies['metadata'] = (
    movies['overview'] + ' ' +
    movies['genres'].apply(lambda x: ' '.join(x)) + ' ' +
    movies['keywords'].apply(lambda x: ' '.join(x)) + ' ' +
    movies['cast'].apply(lambda x: ' '.join(x)) + ' ' +
    movies['crew']
)

# Step 6: Vectorize the text using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['metadata'])

# Step 7: Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Step 8: Create a reverse mapping of movie titles to indices
movie_indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Step 9: Define the recommendation function
def recommend(title, cosine_sim=cosine_sim):
    if title not in movie_indices:
        return f"Movie '{title}' not found in dataset."

    idx = movie_indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]  # Top 10 similar movies
    movie_recs = [movies.iloc[i[0]].title for i in sim_scores]

    return movie_recs

# Step 10: Try the recommender!
movie_title = "Jerry Maguire"  # Change this title to test other movies
recommend(movie_title)
