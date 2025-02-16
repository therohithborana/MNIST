import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CNN
import torch.optim as optim

def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root='data', train=True, transform=transform, download=True)
    test_data = datasets.MNIST(root='data', train=False, transform=transform, download=True)
    
    loaders = {
        'train': DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1),
        'test': DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1)
    }
    return loaders

def train_model(epochs=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    loaders = load_data()
    
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 20 == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loaders['train'].dataset)} ({100. * batch_idx / len(loaders['train']):.0f}%)]\tLoss: {loss.item():.6f}")
    
    torch.save(model.state_dict(), 'mnist_cnn.pt')
    print("Model saved as mnist_cnn.pt")

if __name__ == "__main__":
    train_model(epochs=1) 