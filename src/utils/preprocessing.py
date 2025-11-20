import pandas as pd
import torch
from transformers import BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(csv_path):
    """Load and preprocess the phishing email dataset"""
    df = pd.read_csv(csv_path)
    df = df.dropna()
    
    # Drop date column if it exists
    if df.columns.str.contains('date').any():
        df = df.drop('date', axis=1)
    
    X = df['text_combined'].tolist()
    Y = df['label'].tolist()
    
    return X, Y

def get_bert_embeddings(texts, model_name='bert-base-uncased', batch_size=32):
    """Generate BERT embeddings for texts"""
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # This is a simplified version - you should implement batching
    # For production, use your existing embedding generation from the notebook
    embeddings = []
    labels = []
    
    return torch.tensor(embeddings), torch.tensor(labels)