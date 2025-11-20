import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FrozenBERTClassifier(nn.Module):
    """
    Simple MLP classifier over BERT embeddings.
    Args:
        input_dim (int): Input embedding dimension.
        hidden_dim (int): Hidden layer dimension.
        num_classes (int): Number of output classes.
    """
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.classifier(x)

def create_dataloaders(
    X: torch.Tensor,
    Y: torch.Tensor,
    batch_size: int = 32,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> tuple:
    """
    Create train, validation, and test dataloaders.
    Args:
        X (torch.Tensor): Features.
        Y (torch.Tensor): Labels.
        batch_size (int): Batch size for training.
        val_ratio (float): Validation split ratio.
        test_ratio (float): Test split ratio.
    Returns:
        tuple: train_loader, val_loader, test_loader
    """
    try:
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
    except Exception as e:
        raise RuntimeError(f"Error creating dataloaders: {e}")

import time
def get_device():
    """Return the best available device (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_frozen_bert(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    train_losses: list,
    device=None
) -> float:
    """
    Training loop for FrozenBERT.
    Args:
        model (nn.Module): Model to train.
        dataloader (DataLoader): Training data loader.
        optimizer (Optimizer): Optimizer.
        train_losses (list): List to append training losses.
    Returns:
        float: Average training loss.
    """
    if device is None:
        device = get_device()
    model.train()
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss()
    start_time = time.time()
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
    elapsed = time.time() - start_time
    if elapsed > 2.0:
        print(f"[Profiling] Training loop took {elapsed:.2f} seconds for {len(dataloader)} batches.")
    return avg_loss

def train_bert_model(data_path, epochs=10, learning_rate=5e-5, save_path='models/frozen_bert.pth', device=None):
    """Main training function"""
    if device is None:
        device = get_device()
    # Load pre-computed embeddings (you need to generate these first)
    data = torch.load(data_path)
    X = torch.tensor(data['embeddings']).to(device)
    Y = torch.tensor(data['labels']).to(device)
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(X, Y, device=device)
    # Initialize model
    model = FrozenBERTClassifier().to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    train_losses = []
    val_accuracies = []
    # Training loop
    for epoch in range(epochs):
        train_frozen_bert(model, train_loader, optimizer, train_losses, device=device)
        # Validation
        from evaluate import evaluate_accuracy
        val_acc = evaluate_accuracy(model, val_loader, device=device)
        val_accuracies.append(val_acc)
    # Save model
    torch.save(model.state_dict(), save_path)
    return model, train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Example usage
    model, train_loader, val_loader, test_loader = train_bert_model(
        data_path='data/phishing_email_bert_embeddings.pt',
        epochs=10
    )