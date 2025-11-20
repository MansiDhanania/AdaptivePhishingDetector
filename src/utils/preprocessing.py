import pandas as pd
import torch
from transformers import BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(csv_path: str) -> tuple:
    """
    Load and preprocess the phishing email dataset.
    Args:
        csv_path (str): Path to CSV file.
    Returns:
        tuple: List of texts, list of labels.
    """
    try:
        df = pd.read_csv(csv_path)
        df = df.dropna()
        if df.columns.str.contains('date').any():
            df = df.drop('date', axis=1)
        X = df['text_combined'].tolist()
        Y = df['label'].tolist()
        return X, Y
    except Exception as e:
        raise RuntimeError(f"Error loading data from {csv_path}: {e}")

def get_bert_embeddings(texts: list, model_name: str = 'bert-base-uncased', batch_size: int = 32) -> torch.Tensor:
    """
    Generate BERT embeddings for texts (batched).
    Args:
        texts (list): List of input texts.
        model_name (str): BERT model name.
        batch_size (int): Batch size for embedding generation.
    Returns:
        torch.Tensor: BERT embeddings.
    """
    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.to(device)
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeds = outputs.last_hidden_state[:, 0, :].cpu()
            embeddings.append(batch_embeds)
    return torch.cat(embeddings)