
import joblib
import re

# Load the saved model and vectorizer
print("Loading model and vectorizer...")
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
print("Model and vectorizer loaded successfully.")

def preprocess_text(text):
    """
    This function cleans the input text by removing URLs, mentions, hashtags,
    and any non-alphabetical characters, and then converts it to lowercase.
    """
    print("Preprocessing text...")
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags
    text = re.sub(r'#', '', text)
    # Keep only alphabetical characters
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Convert to lowercase and strip extra spaces
    text = text.lower().strip()
    print(f"Processed text: {text}")
    return text

def predict_sentiment(text):
    """
    This function takes the preprocessed text, transforms it using the
    vectorizer, and predicts the sentiment using the loaded model.
    """
    print("Predicting sentiment...")
    processed_text = preprocess_text(text)
    text_vectorized = vectorizer.transform([processed_text])
    print(f"Vectorized text: {text_vectorized}")
    prediction = model.predict(text_vectorized)
    sentiment = "Positive" if prediction[0] == 1 else "Negative"
    print(f"Predicted sentiment: {sentiment}")
    return sentiment

# Example usage with user input
if __name__ == "__main__":
    # Get user input
    user_input = input("Enter a tweet or text to analyze sentiment: ")
    sentiment = predict_sentiment(user_input)
    print(f"Sentiment: {sentiment}")
