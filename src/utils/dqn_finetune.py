import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    """DQN Agent to control BERT fine-tuning"""
    def __init__(self, state_size=1, action_size=2):
        self.state_size = state_size
        self.action_size = action_size  # 0: do nothing, 1: fine-tune
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.lr = 0.001
        self.model = self._build_model()
    
    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        ).to(device)
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor([state]).to(device)
        with torch.no_grad():
            act_values = self.model(state_tensor)
        return torch.argmax(act_values).item()
    
    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state in minibatch:
            target = reward + self.gamma * torch.max(
                self.model(torch.FloatTensor([next_state]).to(device))
            ).item()
            
            target_f = self.model(torch.FloatTensor([state]).to(device)).detach().cpu().numpy()
            target_f[0][action] = target
            
            output = self.model(torch.FloatTensor([state]).to(device))
            loss = nn.MSELoss()(output, torch.FloatTensor(target_f).to(device))
            loss.backward()
            
            for param in self.model.parameters():
                param.data -= self.lr * param.grad
            self.model.zero_grad()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def gradual_unfreeze(model, epoch, total_epochs, state):
    """Gradually unfreeze BERT layers"""
    if epoch >= total_epochs // 3 and not state.get("classifier_unfrozen", False):
        for param in model.classifier.parameters():
            param.requires_grad = True
        state["classifier_unfrozen"] = True
        print(f"[INFO] Unfroze classifier layers at epoch {epoch}")
    
    if epoch >= 2 * total_epochs // 3 and not state.get("all_unfrozen", False):
        for param in model.parameters():
            param.requires_grad = True
        state["all_unfrozen"] = True
        print(f"[INFO] Unfroze all layers at epoch {epoch}")
    
    return model

def train_with_dqn(model, train_loader, val_loader, epochs=10):
    """Train model with DQN-based fine-tuning"""
    agent = DQNAgent()
    unfreeze_state = {"classifier_unfrozen": False, "all_unfrozen": False}
    action_history = []
    reward_history = []
    
    from .train import train_frozen_bert
    from .evaluate import evaluate_accuracy
    import torch.optim as optim
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        
        # Train
        train_frozen_bert(model, train_loader, optimizer, train_losses)
        
        # Evaluate and decide action
        model.eval()
        correct = 0
        total = 0
        
        for batch in val_loader:
            embeddings, labels = [x.to(device) for x in batch]
            with torch.no_grad():
                outputs = model(embeddings)
                preds = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        
        error_rate = 1 - (correct / total)
        action = agent.act(error_rate)
        reward = 1 if correct / total > 0.95 else -1
        
        agent.remember(error_rate, action, reward, error_rate)
        action_history.append(action)
        reward_history.append(reward)
        
        # Fine-tune if needed
        if action == 1 and error_rate > 0.005:
            print(f"DQN triggering fine-tune (error rate: {error_rate:.4f})")
            gradual_unfreeze(model, epoch, epochs, unfreeze_state)
            train_frozen_bert(model, train_loader, optimizer, train_losses)
        
        agent.replay()
        val_acc = evaluate_accuracy(model, val_loader)
        val_accuracies.append(val_acc)
    
    return model, action_history, reward_history

if __name__ == "__main__":
    from train import FrozenBERTClassifier, create_dataloaders
    
    # Load data
    data = torch.load('data/phishing_email_bert_embeddings.pt')
    X = data['embeddings']
    Y = data['labels']
    
    train_loader, val_loader, test_loader = create_dataloaders(X, Y)
    
    # Initialize and train
    model = FrozenBERTClassifier().to(device)
    model, actions, rewards = train_with_dqn(model, train_loader, val_loader, epochs=10)
    
    # Save
    torch.save(model.state_dict(), 'models/dqn_finetuned_bert.pth')