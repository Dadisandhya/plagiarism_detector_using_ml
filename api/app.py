from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Set template and static folders to be outside 'api/'
project_root = os.path.dirname(os.path.dirname(_file_))  # Goes one level up from api/
template_folder = os.path.join(project_root, 'templates')
static_folder = os.path.join(project_root, 'static')

app = Flask(_name_, template_folder=template_folder, static_folder=static_folder)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Detect plagiarism route
@app.route('/detect', methods=['POST'])
def detect_plagiarism():
    text1 = request.form.get('text1', '')
    text2 = request.form.get('text2', '')

    if not text1 or not text2:
        return "Please enter both texts to check plagiarism."

    similarity_percentage = detect(text1, text2)
    return render_template('result.html', percentage=similarity_percentage)

# Plagiarism detection function
def detect(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(similarity * 100, 2)

# Run server
if _name_ == "_main_":
    port = int(os.environ.get("PORT", 8000))  # Render provides PORT
    app.run(debug=True, host="0.0.0.0", port=port)

