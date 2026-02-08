import pandas as pd
import os
import re
from .config import BANK_FILE, REGISTER_FILE
def normalize_columns(df, source_name):
    """
    Standardizes column names and formats.
    """
    print(f"Normalizing {source_name} data...")
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        print(f"{source_name} missing 'date' column")
        raise ValueError("Missing date column")

    if 'transaction_id' in df.columns:
        df['true_id'] = df['transaction_id'].str.extract(r'(\d+)').fillna(-1).astype(int)
    
    return df

def load_data():
    """
    Loads Bank and Register datasets.
    Returns: (bank_df, reg_df)
    """
    if not os.path.exists(BANK_FILE):
        print(f"Bank file not found: {BANK_FILE}")
        raise FileNotFoundError(f"Bank file missing: {BANK_FILE}")
        
    if not os.path.exists(REGISTER_FILE):
        print(f"Register file not found: {REGISTER_FILE}")
        raise FileNotFoundError(f"Register file missing: {REGISTER_FILE}")

    print(f"Loading data from {os.path.dirname(BANK_FILE)}")
    
    try:
        bank_df = pd.read_csv(BANK_FILE)
        reg_df = pd.read_csv(REGISTER_FILE)
        
        bank_df = normalize_columns(bank_df, "Bank")
        reg_df = normalize_columns(reg_df, "Register")
        
        print(f"Loaded {len(bank_df)} bank transactions and {len(reg_df)} register transactions")
        return bank_df, reg_df
        
    except Exception as e:
        print(f"Failed to load data: {e}")
        raise
