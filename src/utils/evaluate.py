import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay,
    matthews_corrcoef,
    roc_curve,
    auc,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import time
def evaluate_accuracy(model: nn.Module, dataloader: torch.utils.data.DataLoader) -> float:
    """
    Evaluate classification accuracy.
    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): PyTorch DataLoader.
    Returns:
        float: Accuracy percentage.
    """
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()
    with torch.no_grad():
        for batch in dataloader:
            embeddings, labels = [x.to(device) for x in batch]
            outputs = model(embeddings)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total * 100
    elapsed = time.time() - start_time
    if elapsed > 1.0:
        print(f"[Profiling] Accuracy evaluation took {elapsed:.2f} seconds.")
    return accuracy

def get_predictions(model: nn.Module, dataloader: torch.utils.data.DataLoader) -> tuple:
    """
    Get all predictions and labels from a model and dataloader.
    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): PyTorch DataLoader.
    Returns:
        tuple: predictions, labels
    """
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

def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    model_name: str = "Model",
    return_probs: bool = False
) -> dict:
    """
    Comprehensive model evaluation. If return_probs, also returns probabilities and labels for ROC/PR curves.
    Args:
        model (nn.Module): Trained model.
        test_loader (DataLoader): PyTorch DataLoader.
        model_name (str): Name for reporting.
        return_probs (bool): Whether to return probabilities for ROC/PR curves.
    Returns:
        dict or tuple: Evaluation metrics or (probs, preds, labels)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    start_time = time.time()
    with torch.no_grad():
        for batch in test_loader:
            embeddings, labels = [x.to(device) for x in batch]
            outputs = model(embeddings)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
    mcc = matthews_corrcoef(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    elapsed = time.time() - start_time
    if elapsed > 2.0:
        print(f"[Profiling] Model evaluation took {elapsed:.2f} seconds.")
    if return_probs:
        return np.array(all_probs), np.array(all_preds), np.array(all_labels)
    return {
        'predictions': all_preds,
        'labels': all_labels,
        'mcc': mcc,
        'confusion_matrix': cm
    }
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
    # Classification report
    # ...existing code...
    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(all_labels, all_preds)
    # ...existing code...
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Spam', 'Spam'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.show()
    if return_probs:
        return np.array(all_probs), np.array(all_preds), np.array(all_labels)
    return {
        'predictions': all_preds,
        'labels': all_labels,
        'mcc': mcc,
        'confusion_matrix': cm
    }
def plot_curves(labels: np.ndarray, probs: np.ndarray, model_name: str):
    """
    Plot ROC and Precision-Recall curves for model evaluation.
    Args:
        labels (np.ndarray): True labels.
        probs (np.ndarray): Predicted probabilities.
        model_name (str): Name for plot title.
    """
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(labels, probs)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    axes[0].plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    axes[0].plot([0, 1], [0, 1], linestyle='--')
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[1].plot(recall, precision)
    axes[1].set_title("Precision-Recall Curve")
    plt.tight_layout()
    plt.suptitle(model_name)
    plt.show()

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