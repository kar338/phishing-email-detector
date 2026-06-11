import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Load dataset
data = pd.read_csv("emails.csv")

# Features and labels
X = data["text"]
y = data["label"]

# Convert text to numerical features
vectorizer = TfidfVectorizer()

X_features = vectorizer.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Results
print("=" * 50)
print("PHISHING EMAIL DETECTION MODEL")
print("=" * 50)

print("\nAccuracy:")
print(f"{accuracy_score(y_test, predictions) * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Test Email
sample_email = [
    "Urgent! Verify your account and click the link now"
]

sample_vector = vectorizer.transform(sample_email)

prediction = model.predict(sample_vector)

print("\nSample Email Analysis")
print("-" * 30)
print("Email:", sample_email[0])
print("Prediction:", prediction[0].upper())
