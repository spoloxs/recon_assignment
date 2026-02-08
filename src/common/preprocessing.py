from sklearn.feature_extraction.text import CountVectorizer

class TermExtractor:
    def __init__(self):
        self.vectorizer = CountVectorizer(stop_words='english', min_df=1)
        
    def fit(self, texts):
        self.vectorizer.fit(texts)
        
    def transform(self, texts):
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()
