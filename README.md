# Logistic-regression

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the dataset
file_path = r"C:\Users\Lenovo\Downloads\spam_text_data.csv"  # Update the path if needed
df = pd.read_csv(file_path)

# Step 2: Data Preprocessing (Check for missing values)
print("Missing Values:\n", df.isnull().sum())

df.dropna(inplace=True)  # Remove missing values

# Split features and target
X = df['Message']  # Text messages
y = df['Spam']  # Target variable

# Step 3: Convert text data to numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Step 4: Split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)

# Step 5: Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Model Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 7: Save the trained model & vectorizer
with open("spam_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

print("Model and Vectorizer saved successfully!")

# Step 8: Function to Predict New Emails
def predict_spam(new_text):
    with open("spam_model.pkl", "rb") as model_file:
        loaded_model = pickle.load(model_file)
    with open("vectorizer.pkl", "rb") as vec_file:
        loaded_vectorizer = pickle.load(vec_file)
    
    new_text_tfidf = loaded_vectorizer.transform([new_text])
    prediction = loaded_model.predict(new_text_tfidf)
    return prediction[0]

# Example: Predict spam on a new email
new_email = "great offer! dont miss 50% off please do shopping right now"
y_new_pred = predict_spam(new_email)
print("Predicted Spam Label:", y_new_pred)



