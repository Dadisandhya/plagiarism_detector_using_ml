from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# app = Flask(__name__)
from flask import Flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_plagiarism():

    text1 = request.form.get('text1', '')
    text2 = request.form.get('text2', '')

    if not text1 or not text2:
        return "Please enter both texts to check plagiarism."

    similarity_percentage = detect(text1, text2)

    return render_template('result.html', percentage=similarity_percentage)

def detect(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(similarity * 100, 2)  

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))  #port 8000
    app.run(debug=True, host="0.0.0.0", port=port)



