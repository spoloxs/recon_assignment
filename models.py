from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch
from config import EMBEDDING_MODEL

class TransformerModel:
    def __init__(self, model_name=EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
    
    def fit(self, bank_texts, reg_texts, epochs=1):
        print("Fine-tuning started.")
        train_examples = []
        for b_text, r_text in zip(bank_texts, reg_texts):
            train_examples.append(InputExample(texts=[str(b_text), str(r_text)], label=1.0))
        print(f"Created {len(train_examples)} training examples for fine-tuning.")
        
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        eval_loader = DataLoader(train_examples, shuffle=False, batch_size=32, 
                               collate_fn=self.model.smart_batching_collate)
    
        self.model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs, show_progress_bar=False)
                
        print("Fine-tuning complete.")

    def transform(self, texts):
        if hasattr(texts, 'tolist'):
            texts = texts.tolist()
        return self.model.encode(texts, show_progress_bar=False)
