from flask import Flask, request, render_template
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load movie data
try:
    movies = pd.read_csv('movies.csv')  # Make sure movies.csv is in the same folder
except FileNotFoundError:
    raise FileNotFoundError("The file 'movies.csv' was not found. Please place it in the same directory as app.py")

# Clean and process data
movies['overview'] = movies['overview'].fillna('')

# Vectorize the movie overviews
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Build a reverse map from title to index
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def get_recommendations(title):
    idx = indices.get(title)
    if idx is None:
        return ["Movie not found. Try another title."]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    error = None
    if request.method == 'POST':
        title = request.form.get('title')
        if title:
            recommendations = get_recommendations(title)
        else:
            error = "Please enter a movie title."
    return render_template('index.html', recommendations=recommendations, error=error)

if __name__ == '__main__':
    app.run(debug=True)