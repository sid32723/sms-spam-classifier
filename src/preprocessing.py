# import pandas as pd


# def create_message_features(messages):
#     features = pd.DataFrame(index=messages.index)

#     features["message_length"] = messages.str.len()
#     features["digit_count"] = messages.str.count(r"\d")
#     features["exclamation_count"] = messages.str.count("!")
#     features["uppercase_count"] = messages.apply(
#         lambda x: sum(char.isupper() for char in x)
#     )

#     return features


import pandas as pd

from scipy.sparse import hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


def create_message_features(messages):
    features = pd.DataFrame(index=messages.index)

    features["message_length"] = messages.str.len()
    features["digit_count"] = messages.str.count(r"\d")
    features["exclamation_count"] = messages.str.count("!")
    features["uppercase_count"] = messages.apply(
        lambda x: sum(char.isupper() for char in x)
    )

    return features


class CombinedFeatures(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.word_vectorizer = TfidfVectorizer()
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5)
        )
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.word_vectorizer.fit(X)
        self.char_vectorizer.fit(X)

        engineered = create_message_features(X)
        self.scaler.fit(engineered)

        return self

    def transform(self, X):
        word_features = self.word_vectorizer.transform(X)
        char_features = self.char_vectorizer.transform(X)

        engineered = create_message_features(X)
        engineered_scaled = self.scaler.transform(engineered)

        return hstack([
            word_features,
            char_features,
            engineered_scaled
        ])