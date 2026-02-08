# Financial Reconciliation System

A robust, scalable, and structured application for reconciling financial transactions between Bank Statements and Check Registers.

## Technologies Used
- **Python 3.14**: Core logic.
- **Pandas**: Data manipulation.
- **Scikit-Learn**: SVD (LSA) and Term Vectorization.
- **FAISS**: Efficient similarity search.
- **Sentence-Transformers**: BERT Embeddings.
- **Typer**: CLI Interface.
- **Pytest**: Unit Testing.

## Quick Start

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run CLI:**
    ```bash
    python main.py --help
    ```

## CLI Commands

- `reconcile`: Run the full pipeline (Unique -> ML).
    - `--model SVD` (Default): Uses LSA + Alignment.
    - `--model Embedding`: Uses BERT (Fine-tuned).
- `match-unique`: Run only the unique amount matcher (High confidence).
- `test`: Run unit tests.
- `demo-svd`: Show SVD improvement cycle.
- `demo-embedding`: Show Embedding performance.

## Documentation
- **[Method 1 Analysis (SVD)](METHOD_1_SVD.md)**: Detailed breakdown of the SVD algorithm.
- **[Method 2 Analysis (Embedding)](METHOD_2_EMBEDDING.md)**: Breakdown of the Deep Learning approach.
- **[Full Analysis Report](ANALYSIS_REPORT.md)**: Comprehensive metrics, trade-offs, and design decisions.

## Testing
```bash
python -m pytest main/tests/
```
