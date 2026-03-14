from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample Data
emails = [
    "Hey, are we still meeting for lunch today?",
    "CONGRATULATIONS! You've won a $1,000 Walmart gift card. Click here.",
    "Can you send me the report by 5 PM?",
    "URGENT: Your account has been compromised. Verify now for cash.",
    "Don't forget to pick up milk on your way home.",
    "FREE entry to our annual prize draw! Text WIN to 80085"
]
# 0 = Not Spam (Ham), 1 = Spam
labels = [0, 1, 0, 1, 0, 1]

# Vectorize the text (Convert words to numbers)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Split data (Normally you'd use a larger dataset)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Train the Naive Bayes Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on new data
new_emails = ["Get a free gift card!", "Are you home?"]
new_emails_counts = vectorizer.transform(new_emails)
predictions = model.predict(new_emails_counts)

print(f"Predictions: {predictions}") # 1 is Spam, 0 is Not Spam
