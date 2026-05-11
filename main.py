import torch
from network import MNIST_CNN
from utils import get_mnist_loaders, plot_results
from train import train_full

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    model = MNIST_CNN(Qmax=127, noise_std_lsb=0.0).to(device)

    epochs = 15
    lr = 1e-3
    metrics = train_full(model, train_loader, test_loader, epochs, lr, device, noise_train=0.0)

    torch.save(model.state_dict(), "mnist_lsq.pth")
    plot_results(*metrics, save_path="training_results.png")

if __name__ == "__main__":
    main()