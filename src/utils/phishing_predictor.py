import torch
import torch.nn as nn
import PyPDF2
import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    
    return text

def load_trained_model(model_path, model_class):
    """Load a trained model"""
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_phishing(pdf_path_or_text, model, tokenizer=None):
    """
    Predict if an email is phishing
    
    Args:
        pdf_path_or_text: Either path to PDF or text string
        model: Trained classifier model
        tokenizer: BERT tokenizer (if needed for embedding generation)
    
    Returns:
        dict: Prediction results
    """
    # Extract text if PDF path provided
    if pdf_path_or_text.endswith('.pdf'):
        text = extract_text_from_pdf(pdf_path_or_text)
    else:
        text = pdf_path_or_text
    
    if not text.strip():
        return {"error": "No text extracted"}
    
    # Generate embedding (simplified - you need proper implementation)
    # In production, use your existing embedding generation code
    from transformers import BertTokenizer, BertModel
    
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    
    # Tokenize and get embedding
    inputs = tokenizer(text, return_tensors='pt', truncation=True, 
                      max_length=512, padding=True).to(device)
    
    with torch.no_grad():
        bert_output = bert_model(**inputs)
        embedding = bert_output.last_hidden_state[:, 0, :]  # CLS token
    
    # Predict
    with torch.no_grad():
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
    
    return result

if __name__ == "__main__":
    from train import FrozenBERTClassifier
    
    # Load model
    model = load_trained_model('models/dqn_finetuned_bert.pth', FrozenBERTClassifier)
    
    # Example prediction
    test_text = "Congratulations! You've won $1,000,000! Click here to claim..."
    result = predict_phishing(test_text, model)
    
    print(f"\nPrediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2f}%")