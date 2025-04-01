from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import re

app = Flask(__name__)

# Load and preprocess data (your original code)
data = pd.read_csv("IMDB Dataset.csv", encoding="utf-8", on_bad_lines="skip")
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text
data['review'] = data['review'].astype(str).apply(preprocess_text)

# Train model
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data['review'])
y = data['sentiment'].map({'positive': 1, 'negative': 0})
model = LogisticRegression()
model.fit(X, y)

@app.route('/predict', methods=['POST'])
def predict():
    review = request.json['review']
    clean_review = preprocess_text(review)
    vector = tfidf.transform([clean_review])
    pred = model.predict(vector)[0]
    prob = model.predict_proba(vector)[0][pred]
    sentiment = 'Positive' if pred == 1 else 'Negative'
    return jsonify({'sentiment': sentiment, 'confidence': f'{prob:.2%}'})

if __name__ == '__main__':
    app.run(debug=True)