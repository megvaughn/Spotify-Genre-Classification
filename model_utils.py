from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

class AudioTextCombiner(BaseEstimator, TransformerMixin):
    def __init__(self, audio_features, text_col="lyrics",
                 tfidf_max_features=20000, tfidf_ngram_range=(1,2), tfidf_min_df=5):
        self.audio_features = audio_features
        self.text_col = text_col
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range = tfidf_ngram_range
        self.tfidf_min_df = tfidf_min_df
        self.scaler_ = StandardScaler()
        self.tfidf_ = None

    def fit(self, X, y=None):
        Xa = X[self.audio_features]
        Xt = X[self.text_col].astype(str)
        self.scaler_.fit(Xa)
        self.tfidf_ = TfidfVectorizer(
            stop_words="english",
            max_features=self.tfidf_max_features,
            ngram_range=self.tfidf_ngram_range,
            min_df=self.tfidf_min_df
        )
        self.tfidf_.fit(Xt)
        return self

    def transform(self, X):
        Xa = X[self.audio_features]
        Xt = X[self.text_col].astype(str)
        Xa_scaled = self.scaler_.transform(Xa)
        Xa_sparse = csr_matrix(Xa_scaled)
        Xt_tfidf = self.tfidf_.transform(Xt)
        return hstack([Xa_sparse, Xt_tfidf])
