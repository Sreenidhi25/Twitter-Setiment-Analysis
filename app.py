from flask import Flask, render_template, request
import joblib
import re

# Initialize Flask app
app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def preprocess_text(text):
    # Clean and preprocess the text
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    return text.lower().strip()

def predict_sentiment(text):
    processed_text = preprocess_text(text)
    text_vectorized = vectorizer.transform([processed_text])
    prediction = model.predict(text_vectorized)
    return "Positive" if prediction[0] == 1 else "Negative"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    sentiment = predict_sentiment(text)
    return render_template('index.html', text=text, sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
