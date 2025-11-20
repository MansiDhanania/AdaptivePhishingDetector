import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay,
    matthews_corrcoef
)
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_accuracy(model, dataloader):
    """Evaluate classification accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            embeddings, labels = [x.to(device) for x in batch]
            outputs = model(embeddings)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total * 100
    print(f"Validation Accuracy: {accuracy:.2f}%")
    return accuracy

def get_predictions(model, dataloader):
    """Get all predictions and labels"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            embeddings, labels = [x.to(device) for x in batch]
            outputs = model(embeddings)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_preds, all_labels

def evaluate_model(model, test_loader, model_name="Model"):
    """Comprehensive model evaluation"""
    preds, labels = get_predictions(model, test_loader)
    
    # Classification report
    print(f"\n=== {model_name} Classification Report ===")
    print(classification_report(labels, preds, target_names=['Non-Spam', 'Spam']))
    
    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(labels, preds)
    print(f"\nMatthews Correlation Coefficient: {mcc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Spam', 'Spam'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.show()
    
    return {
        'predictions': preds,
        'labels': labels,
        'mcc': mcc,
        'confusion_matrix': cm
    }

if __name__ == "__main__":
    from train import FrozenBERTClassifier
    
    # Load model
    model = FrozenBERTClassifier().to(device)
    model.load_state_dict(torch.load('models/frozen_bert.pth'))
    
    # Load test data
    data = torch.load('data/phishing_email_bert_embeddings.pt')
    # Create test loader (implement this based on your needs)
    
    # Evaluate
    results = evaluate_model(model, test_loader, "FrozenBERT")