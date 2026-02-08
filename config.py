import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
BANK_FILE = os.path.join(DATA_DIR, 'bank_statements.csv')
REGISTER_FILE = os.path.join(DATA_DIR, 'check_register.csv')

UNIQUE_DATE_TOLERANCE=5