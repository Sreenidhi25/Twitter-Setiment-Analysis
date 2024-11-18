import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load your dataset (change the file path if needed)
# Assuming your dataset has 'text' and 'sentiment' columns
df = pd.read_csv('train_tweet.csv')

# Split the data into features (X) and target (y)
X = df['text']
y = df['sentiment']  # Assuming sentiment is labeled as 0 for negative and 1 for positive

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a CountVectorizer instance
vectorizer = CountVectorizer(stop_words='english', max_features=5000)

# Fit the vectorizer on the training data
X_train_vectorized = vectorizer.fit_transform(X_train)

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vectorized, y_train)

# Make predictions and evaluate the model
X_test_vectorized = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')


print("Model and vectorizer saved successfully!")
