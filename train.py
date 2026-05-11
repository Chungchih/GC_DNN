import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_epoch(model, loader, criterion, optimizer, device, noise_std=0.0):
    model.train()
    model.set_noise(noise_std)
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc='Training'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / len(loader), 100. * correct / total

@torch.no_grad()
def evaluate(model, loader, device, noise_std=0.0):
    model.eval()
    model.set_noise(noise_std)
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

def train_full(model, train_loader, test_loader, epochs, lr, device, noise_train=0.0):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_losses, train_accs, test_acc_clean, test_acc_noisy = [], [], [], []
    for epoch in range(epochs):
        loss, acc = train_epoch(model, train_loader, criterion, optimizer, device, noise_std=noise_train)
        clean = evaluate(model, test_loader, device, noise_std=0.0)
        noisy = evaluate(model, test_loader, device, noise_std=0.75)
        train_losses.append(loss)
        train_accs.append(acc)
        test_acc_clean.append(clean)
        test_acc_noisy.append(noisy)
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Train Acc={acc:.2f}%, Test Clean={clean:.2f}%, Test Noisy={noisy:.2f}%")
    return train_losses, train_accs, test_acc_clean, test_acc_noisy