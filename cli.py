import typer
import pandas as pd
from typing import Optional
from src.common.logger import setup_logger
from src.common.data_loader import load_data
from src.common.unique_matcher import UniqueAmountMatcher
from src.common.alignment import AlignmentModel
from src.common.preprocessing import TermExtractor
from src.common.evaluation import evaluate_results
from src.common.config import SVD_COMPONENTS
from src.svd.model import SVDReconciler
from src.svd.matcher import SVDMatchingEngine
from src.embedding.model import TransformerModel
from src.embedding.matcher import MLMatchingEngine
from src.demos import run_svd_progress_demo, run_embedding_demo

logger = setup_logger("CLI")
app = typer.Typer(help="Financial Reconciliation System CLI", add_completion=False)

@app.command()
def reconcile(model: str = typer.Option("SVD", help="Model type: 'SVD' (Option A) or 'Embedding' (Option B)")):
    """
    Run the full reconciliation pipeline (Unique -> ML).
    """
    logger.info(f"--- Running Full Pipeline ({model}) ---")
    try:
        bank_df, reg_df = load_data()
    except Exception as e:
        logger.error(e)
        return

    # 1. Unique Match (Core 2.1)
    unique_matcher = UniqueAmountMatcher(bank_df, reg_df)
    matches_unique, rem_bank, rem_reg = unique_matcher.find_matches()
    
    matches_ml = pd.DataFrame()
    if not rem_bank.empty and not rem_reg.empty:
        logger.info(f"Matching remaining {len(rem_bank)} transactions...")
        
        train_bank = bank_df.loc[matches_unique['bank_idx'], 'description']
        train_reg = reg_df.loc[matches_unique['reg_idx'], 'description']
        
        if model.upper() == "SVD":
            # Alignment + SVD
            alignment = AlignmentModel()
            if len(train_bank) > 0:
                alignment.fit(train_bank, train_reg)
                aligned_bank, reg_vec = alignment.transform(rem_bank['description'], rem_reg['description'])
                _, all_reg_vec = alignment.transform([], reg_df['description'])
            else:
                logger.warning("No training data for Alignment. Using Raw LSA.")
                # Fallback logic
                extractor = TermExtractor()
                all_text = pd.concat([bank_df['description'], reg_df['description']])
                extractor.fit(all_text)
                aligned_bank = extractor.transform(rem_bank['description'])
                reg_vec = extractor.transform(rem_reg['description'])
                all_reg_vec = extractor.transform(reg_df['description'])

            svd = SVDReconciler(n_components=SVD_COMPONENTS)
            svd.fit(all_reg_vec)
            
            bank_vec_final = svd.transform(aligned_bank)
            reg_vec_final = svd.transform(reg_vec)
            
            matcher = SVDMatchingEngine(rem_bank, rem_reg, bank_vec_final, reg_vec_final)

        elif model.upper() == "EMBEDDING":
            transformer = TransformerModel()
            if len(train_bank) > 0:
                transformer.fit(train_bank, train_reg, epochs=1)
            
            bank_vec_final = transformer.transform(rem_bank['description'])
            reg_vec_final = transformer.transform(rem_reg['description'])
            
            matcher = MLMatchingEngine(rem_bank, rem_reg, bank_vec_final, reg_vec_final)
        else:
            logger.error(f"Unknown model: {model}")
            return

        matches_ml = matcher.run()

    all_matches = pd.concat([matches_unique, matches_ml], ignore_index=True)
    filename = f"reconciliation_report_{model.lower()}.csv"
    all_matches.to_csv(filename, index=False)
    logger.info(f"Report saved to {filename}")
    
    evaluate_results(all_matches, bank_df, reg_df)

@app.command()
def match_unique():
    """
    Run ONLY the Unique Amount Matcher (Core 2.1).
    """
    logger.info("--- Unique Amount Matching Only ---")
    try:
        bank_df, reg_df = load_data()
        matcher = UniqueAmountMatcher(bank_df, reg_df)
        matches, _, _ = matcher.find_matches()
        
        matches.to_csv("unique_matches.csv", index=False)
        logger.info(f"Unique Matches saved to unique_matches.csv")
        evaluate_results(matches, bank_df, reg_df)
    except Exception as e:
        logger.error(e)

@app.command()
def test():
    """
    Run Unit Tests using pytest.
    """
    import pytest
    logger.info("Running all unit tests...")
    # Point to main/tests directory
    pytest.main(["-v", "main/tests/"])

@app.command()
def demo_svd():
    """Run SVD Progress Demo"""
    run_svd_progress_demo()

@app.command()
def demo_embedding():
    """Run Embedding Progress Demo"""
    run_embedding_demo()

if __name__ == "__main__":
    app()
