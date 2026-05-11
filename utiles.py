import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def get_mnist_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def plot_results(train_losses, train_accs, test_acc_clean, test_acc_noisy, save_path='results.png'):
    epochs = range(1, len(train_losses)+1)
    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    axes[0].plot(epochs, train_losses, 'b-')
    axes[0].set_title('Training Loss')
    axes[1].plot(epochs, train_accs, 'r-', label='Train')
    axes[1].plot(epochs, test_acc_clean, 'g-', label='Test Clean')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[2].bar(['Clean', 'Noisy (0.75 LSB)'], [test_acc_clean[-1], test_acc_noisy[-1]], color=['green', 'orange'])
    axes[2].set_title('Final Test Accuracy')
    for i, v in enumerate([test_acc_clean[-1], test_acc_noisy[-1]]):
        axes[2].text(i, v+0.5, f"{v:.2f}%", ha='center')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()