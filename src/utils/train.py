import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FrozenBERTClassifier(nn.Module):
    """Simple MLP classifier over BERT embeddings"""
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=2):
        super(FrozenBERTClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

def create_dataloaders(X, Y, batch_size=32, val_ratio=0.15, test_ratio=0.15):
    """Create train, validation, and test dataloaders"""
    dataset = TensorDataset(X, Y)
    total_size = len(dataset)
    val_size = int(val_ratio * total_size)
    test_size = int(test_ratio * total_size)
    train_size = total_size - val_size - test_size
    
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    return train_loader, val_loader, test_loader

def train_frozen_bert(model, dataloader, optimizer, train_losses):
    """Training loop for FrozenBERT"""
    model.train()
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss()
    
    for batch in dataloader:
        embeddings, labels = [x.to(device) for x in batch]
        
        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    train_losses.append(avg_loss)
    print(f"Training Loss: {avg_loss:.4f}")
    
    return avg_loss

def train_bert_model(data_path, epochs=10, learning_rate=5e-5, save_path='models/frozen_bert.pth'):
    """Main training function"""
    # Load pre-computed embeddings (you need to generate these first)
    data = torch.load(data_path)
    X = data['embeddings']
    Y = data['labels']
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(X, Y)
    
    # Initialize model
    model = FrozenBERTClassifier().to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_accuracies = []
    
    # Training loop
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        train_frozen_bert(model, train_loader, optimizer, train_losses)
        
        # Validation
        from evaluate import evaluate_accuracy
        val_acc = evaluate_accuracy(model, val_loader)
        val_accuracies.append(val_acc)
    
    # Save model
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")
    
    return model, train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Example usage
    model, train_loader, val_loader, test_loader = train_bert_model(
        data_path='data/phishing_email_bert_embeddings.pt',
        epochs=10
    )