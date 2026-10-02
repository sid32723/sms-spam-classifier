# import joblib
# from preprocessing import create_message_features
# import pandas as pd
# from scipy.sparse import hstack

# model = joblib.load("models/spam_model.pkl")
# vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
# scaler = joblib.load("models/feature_scaler.pkl")

# # print("Model and preprocessing objects loaded successfully!")

# message = input("Enter an SMS message: ")
# # print("You entered:", message)

# message_series = pd.Series([message])
# features = create_message_features(message_series)
# # print(features)

# message_tfidf = vectorizer.transform(message_series)
# # print("TF-IDF shape:", message_tfidf.shape)

# features_scaled = scaler.transform(features)
# # print("Scaled feature shape:", features_scaled.shape)

# combined_features = hstack([
#     message_tfidf,
#     features_scaled
# ])
# # print("Combined feature shape:", combined_features.shape)

# prediction = model.predict(combined_features)[0]
# probabilities = model.predict_proba(combined_features)[0]
# spam_probability = probabilities[list(model.classes_).index("spam")]

# print("Prediction:", prediction.upper())
# print(f"Spam probability: {spam_probability:.2%}")

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
import joblib

# Load the complete trained pipeline
pipeline = joblib.load(
    PROJECT_ROOT / "models" / "final_spam_pipeline.pkl"
)

# Get message from user
message = input("Enter an SMS message: ")

# Make prediction
message_series = pd.Series([message])

prediction = pipeline.predict(message_series)[0]

# Get probabilities
probabilities = pipeline.predict_proba(message_series)[0]

# Find spam probability
spam_index = list(pipeline.classes_).index("spam")
spam_probability = probabilities[spam_index]

print()
print("Prediction:", prediction.upper())
print(f"Spam probability: {spam_probability:.2%}")