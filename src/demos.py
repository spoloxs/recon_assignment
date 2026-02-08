import pandas as pd
import numpy as np
from .common.data_loader import load_data
from .common.unique_matcher import UniqueAmountMatcher
from .common.alignment import AlignmentModel
from .common.preprocessing import TermExtractor
from .common.evaluation import evaluate_results
from .common.logger import setup_logger
from .common.config import SVD_COMPONENTS
from .svd.model import SVDReconciler
from .svd.matcher import SVDMatchingEngine
from .embedding.model import TransformerModel
from .embedding.matcher import MLMatchingEngine

logger = setup_logger("Demos")

def run_svd_progress_demo():
    """
    Demonstrates incremental improvement using SVD Model.
    """
    logger.info("--- Starting SVD Progress Demonstration ---")
    bank_df, reg_df = load_data()
    
    unique_matcher = UniqueAmountMatcher(bank_df, reg_df)
    matches_unique, _, _ = unique_matcher.find_matches()
    
    full_train_bank = bank_df.loc[matches_unique['bank_idx'], 'description']
    full_train_reg = reg_df.loc[matches_unique['reg_idx'], 'description']
    
    # Steps: 0%, 50%, 100% of training data
    total_train = len(full_train_bank)
    steps = [0, int(total_train * 0.5), total_train]
    
    for n in steps:
        _run_svd_iteration(bank_df, reg_df, full_train_bank, full_train_reg, n)

def _run_svd_iteration(bank_df, reg_df, full_train_bank, full_train_reg, n):
    logger.info(f"\n[SVD] Training Size: {n}")
    
    if n == 0:
        logger.info("Zero-Shot LSA (No Alignment)")
        extractor = TermExtractor()
        all_text = pd.concat([bank_df['description'], reg_df['description']])
        extractor.fit(all_text)
        
        aligned_bank_vec = extractor.transform(bank_df['description'])
        reg_vec = extractor.transform(reg_df['description'])
        
        svd_model = SVDReconciler(n_components=SVD_COMPONENTS) 
        svd_model.fit(reg_vec)
    else:
        logger.info(f"Training Alignment with {n} pairs...")
        train_bank = full_train_bank.iloc[:n]
        train_reg = full_train_reg.iloc[:n]
        
        alignment_model = AlignmentModel()
        alignment_model.fit(train_bank, train_reg)
        
        aligned_bank_vec, reg_vec = alignment_model.transform(
            bank_df['description'], 
            reg_df['description']
        )
        
        _, all_reg_vec = alignment_model.transform([], reg_df['description'])
        svd_model = SVDReconciler(n_components=SVD_COMPONENTS)
        svd_model.fit(all_reg_vec)

    # Project
    bank_svd = svd_model.transform(aligned_bank_vec)
    reg_svd = svd_model.transform(reg_vec)
    
    # Match
    matcher = SVDMatchingEngine(bank_df, reg_df, bank_svd, reg_svd)
    matches = matcher.run()
    
    evaluate_results(matches, bank_df, reg_df)

def run_embedding_demo():
    """
    Demonstrates Zero-Shot vs Fine-Tuned Embedding Model.
    """
    logger.info("--- Starting Embedding Model Demonstration ---")
    bank_df, reg_df = load_data()
    
    # 1. Zero-Shot
    logger.info("\n[Embedding] Iteration 1: Zero-Shot (Pre-trained)")
    model = TransformerModel()
    _run_emb_iteration(model, bank_df, reg_df)
    
    # 2. Fine-Tuned
    logger.info("\n[Embedding] Iteration 2: Fine-Tuned (Feedback)")
    unique_matcher = UniqueAmountMatcher(bank_df, reg_df)
    matches_unique, _, _ = unique_matcher.find_matches()
    
    train_bank = bank_df.loc[matches_unique['bank_idx'], 'description']
    train_reg = reg_df.loc[matches_unique['reg_idx'], 'description']
    
    # Fine-tune on subset for speed/demo (or full set)
    sample_size = min(100, len(train_bank))
    model.fit(train_bank[:sample_size], train_reg[:sample_size], epochs=10) # 10 epochs for demo convergence
    
    _run_emb_iteration(model, bank_df, reg_df)

def _run_emb_iteration(model, bank_df, reg_df):
    bank_vec = model.transform(bank_df['description'])
    reg_vec = model.transform(reg_df['description'])
    
    matcher = MLMatchingEngine(bank_df, reg_df, bank_vec, reg_vec)
    matches = matcher.run()
    
    evaluate_results(matches, bank_df, reg_df)
