import torch
import torch.nn as nn
import PyPDF2
import io

def get_device():
    """Return the best available device (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF file.
    Args:
        pdf_path (str): Path to PDF file.
    Returns:
        str: Extracted text.
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text

def load_trained_model(model_path: str, model_class: nn.Module, device=None) -> nn.Module:
    """
    Load a trained model from disk.
    Args:
        model_path (str): Path to model weights.
        model_class (nn.Module): Model class to instantiate.
    Returns:
        nn.Module: Loaded model.
    """
    try:
        if device is None:
            device = get_device()
        model = model_class().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model from {model_path}: {e}")

import time
from typing import Any, Dict
def predict_phishing(pdf_path_or_text: Any, model: nn.Module, tokenizer=None, device=None) -> Dict[str, Any]:
    """
    Predict if an email is phishing.
    Args:
        pdf_path_or_text (str): Path to PDF or text string.
        model (nn.Module): Trained classifier model.
        tokenizer: BERT tokenizer (if needed for embedding generation).
    Returns:
        dict: Prediction results.
    """
    if device is None:
        device = get_device()
    start_time = time.time()
    if isinstance(pdf_path_or_text, str) and pdf_path_or_text.lower().endswith('.pdf'):
        text = extract_text_from_pdf(pdf_path_or_text)
    else:
        text = pdf_path_or_text

    if not text.strip():
        return {"error": "No text extracted"}

    from transformers import BertTokenizer, BertModel
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

    # Tokenize and get embedding
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        bert_output = bert_model(**inputs)
        embedding = bert_output.last_hidden_state[:, 0, :]  # CLS token
        output = model(embedding)
        probs = torch.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1).item()

    result = {
        "prediction": "Phishing" if prediction == 1 else "Legitimate",
        "confidence": probs[0][prediction].item() * 100,
        "probabilities": {
            "legitimate": probs[0][0].item() * 100,
            "phishing": probs[0][1].item() * 100
        }
    }

    elapsed = time.time() - start_time
    if elapsed > 1.0:
        print(f"[Profiling] Prediction took {elapsed:.2f} seconds.")
    return result

if __name__ == "__main__":
    from train import FrozenBERTClassifier
    
    # Load model
    model = load_trained_model('models/dqn_finetuned_bert.pth', FrozenBERTClassifier)
    
    # Example prediction
    test_text = "Congratulations! You've won $1,000,000! Click here to claim..."
    result = predict_phishing(test_text, model)
    
    # ...existing code...