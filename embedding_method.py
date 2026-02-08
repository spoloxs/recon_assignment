from src.common.data_loader import load_data
from src.common.evaluation import evaluate_results
from src.embedding.model import TransformerModel
from src.embedding.matcher import MLMatchingEngine

if __name__ == "__main__":
    bank_df , reg_df = load_data()

    print("--Running ML Matcher before fine-tuning...--")
    transformer = TransformerModel()
    bank_vecs = transformer.transform(bank_df['description'])
    reg_vecs = transformer.transform(reg_df['description'])
    ml_matcher = MLMatchingEngine(bank_df, reg_df, bank_vecs, reg_vecs)
    ml_matches = ml_matcher.run()
    print(f"ML Matcher found {len(ml_matches)} matches\n")
    evaluate_results(ml_matches, bank_df, reg_df)

    print("\n--Running ML Matcher after fine-tuning...--")
    transformer.fit(bank_df['description'], reg_df['description'], epochs=50)
    bank_vecs = transformer.transform(bank_df['description'])
    reg_vecs = transformer.transform(reg_df['description'])
    ml_matcher = MLMatchingEngine(bank_df, reg_df, bank_vecs, reg_vecs)
    ml_matches = ml_matcher.run()
    print(f"ML Matcher found {len(ml_matches)} matches")
    evaluate_results(ml_matches, bank_df, reg_df)
