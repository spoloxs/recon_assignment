from sklearn.decomposition import TruncatedSVD
from ..common.config import SVD_COMPONENTS
from ..common.logger import setup_logger

logger = setup_logger("SVDModel")

class SVDReconciler:
    def __init__(self, n_components=SVD_COMPONENTS):
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        
    def fit(self, matrix):
        logger.info(f"Fitting SVD ({self.svd.n_components} components)...")
        self.svd.fit(matrix)
        
    def transform(self, matrix):
        return self.svd.transform(matrix)
