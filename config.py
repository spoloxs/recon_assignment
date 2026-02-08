import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
BANK_FILE = os.path.join(DATA_DIR, 'bank_statements.csv')
REGISTER_FILE = os.path.join(DATA_DIR, 'check_register.csv')

UNIQUE_DATE_TOLERANCE=5

# ML Config
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
MIN_CONFIDENCE = 0.40
MAX_AMOUNT_DIFF_PERCENT = 0.10
DATE_LOOKBACK_DAYS = 5
DATE_LOOKAHEAD_DAYS = 15
ANN_CANDIDATES = 50
