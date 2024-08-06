from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)
CORS(app)

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

def prepare_data(data):
    df = pd.DataFrame(data)
    df['title_lower'] = df['title'].str.lower()
    return df

def create_model(df):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['overview'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def get_recommendations(title, df, cosine_sim):
    title_lower = title.lower()
    idx = df.index[df['title_lower'] == title_lower].tolist()
    if not idx:
        return ["Film non trouvé"]
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

data = load_data('films.json')
df = prepare_data(data)
cosine_sim = create_model(df)

@app.route('/recommend', methods=['GET'])
def recommend():
    title = request.args.get('title')
    if not title:
        return jsonify({'error': 'Titre requis'}), 400
    recommendations = get_recommendations(title, df, cosine_sim)
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)
