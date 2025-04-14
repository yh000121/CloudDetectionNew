import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_model(model, train_loader, criterion, optimizer, num_epochs=20, device='cpu'):
    """
    Train the model using the given DataLoader, loss function, and optimizer.
    """
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()       # Clear gradients
            outputs = model(inputs)     # Forward pass
            loss = criterion(outputs, targets)
            loss.backward()             # Backward pass (autograd)
            optimizer.step()            # Update parameters
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
        epoch_loss = running_loss / total_samples
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

def evaluate_model(model, data_loader, device='cpu'):
    """
    Evaluate the model on the provided DataLoader and print accuracy.
    """
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = correct / total
    print(f"Evaluation Accuracy: {accuracy:.4f}")
    return accuracy
