# spam_detection_aipython

# Import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 1. Load the Data
# Sample dataset: Email text and labels (1 = Spam, 0 = Not Spam)
emails = [
    "Win a free iPhone now!",  # Spam
    "Meeting at 3 PM in the conference room",  # Not Spam
    "Congratulations, you won a lottery!",  # Spam
    "Can we reschedule our call?",  # Not Spam
    "Exclusive deal just for you, claim it now!",  # Spam
    "Please send me the files by tomorrow"  # Not Spam
]

labels = [1, 0, 1, 0, 1, 0]  # Corresponding labels

# 2. Convert Text to Numeric
# Use CountVectorizer to transform text into numerical data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)  # Convert email text into a bag-of-words representation

# 3. Split the Data
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# 4. Train the Model
# Use a Naive Bayes classifier for training
model = MultinomialNB()
model.fit(X_train, y_train)

# 5. Test the Model
# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")  # Print the accuracy of the model

# 6. Predict New Data
# Test the model on new, unseen emails
new_emails = [
    "Get rich quick with this amazing offer!",  # Likely spam
    "Can we meet tomorrow to discuss the project?"  # Likely not spam
]
new_emails_transformed = vectorizer.transform(new_emails)  # Transform new emails into numerical data
predictions = model.predict(new_emails_transformed)  # Predict the labels for the new emails

# Print the predictions
for email, prediction in zip(new_emails, predictions):
    print(f"Email: '{email}' -> {'Spam' if prediction == 1 else 'Not Spam'}")
