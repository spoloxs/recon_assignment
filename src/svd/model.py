from sklearn.decomposition import TruncatedSVD
from ..common.config import SVD_COMPONENTS

class SVDReconciler:
    def __init__(self, n_components=SVD_COMPONENTS):
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        
    def fit(self, matrix):
        print(f"Fitting SVD ({self.svd.n_components} components)...")
        self.svd.fit(matrix)
        
    def transform(self, matrix):
        return self.svd.transform(matrix)
