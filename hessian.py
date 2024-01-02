import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Define the two-layer neural network
class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize the model, criterion, optimizer
input_size = 28 * 28
hidden_size = 128
output_size = 10

model = TwoLayerNet(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
train_losses = []
test_losses = []
ft_values = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Calculate ft
        grad = torch.cat([param.grad.view(-1) for param in model.parameters()])
        hessian = torch.autograd.functional.hessian(lambda x: criterion(model(x), labels).sum(), inputs=images)
        eigenvalues, eigenvectors = torch.linalg.eigh(hessian[0])
        top_k_eigenvalues, indices = torch.topk(eigenvalues, k=10, largest=True)
        top_k_eigenvectors = eigenvectors[:, indices]
        g_bulk = torch.mm(top_k_eigenvectors, torch.mm(top_k_eigenvectors.t(), grad.unsqueeze(1))).squeeze()
        ft = torch.norm(g_bulk) / torch.norm(grad)
        ft_values.append(ft.item())
        
        optimizer.step()
        running_loss += loss.item()
    
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    
    # Validation
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss = test_loss / len(test_loader)
    test_losses.append(test_loss)
    
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
          f"Test Accuracy: {accuracy:.2f}%")

# Plotting
plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(ft_values, label='f_t')
plt.xlabel('Iteration')
plt.ylabel('f_t')
plt.legend()

plt.tight_layout()
plt.show()