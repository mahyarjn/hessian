import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pyhessian import hessian
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./hessian/dataa', train=True, download=False, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./hessian/dataa', train=False, download=False, transform=transform)

    # train_loader2 = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=len(train_dataset))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

    # Define the two-layer neural network
    class TwoLayerNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(TwoLayerNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.fc3 = nn.Softmax(dim=1)

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
    optimizer = optim.Adam(model.parameters(), lr=0.0006)
    model = model.cuda()

    # for i, (imagesall, labelsall) in enumerate(train_loader2):
    #     break
    # imagesall, labelsall  = imagesall.cuda(), labelsall.cuda()

    # Training loop
    num_epochs = 50
    train_losses = []
    test_losses = []
    ft_values = []
    hessiana = []
    grads = []
    bulk = []
    for epoch in range(num_epochs):
        mean = 0
        counter =0
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            counter += 1
            images , labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            # Calculate ft
            #hessian_comp = hessian(model, criterion, data=(images, labels), cuda = True)
            #hessian = torch.autograd.functional.hessian(lambda x: criterion(model(x), labels).sum(), inputs=images)
            #eigenvalues, eigenvectors = torch.linalg.eigh(hessian[0])
            #print(eigenvalues.shape)
            #top_k_eigenvalues, indices = torch.topk(eigenvalues, k=10, largest=True)
            #top_k_eigenvectors = eigenvectors[:, indices]
            #g_bulk = torch.mm(top_k_eigenvectors, torch.mm(top_k_eigenvectors.t(), grad.unsqueeze(1))).squeeze()
            #ft = torch.norm(g_bulk) / torch.norm(grad)
            #ft_values.append(ft.item())
            optimizer.step()
            running_loss += loss.item()
            model.eval()
            top_number = 10
            # gradient = grads[i]

            grad = [param.grad for param in model.parameters()]
            hess_object = hessian(model, criterion, data = (images, labels), cuda = True)
            top_eigenvalues, top_eigenvector = hess_object.eigenvalues(top_n = 10)

            projection_norm = 0
            norm_gradient=0
            for j in range(top_number):
                projection_norm_temp=0
                for a,b in zip(grad, top_eigenvector[j]):
                    projection_norm_temp += torch.sum((a*b))
                projection_norm += projection_norm_temp**2
            for c in grad:
                norm_gradient += torch.sum((c*c))
            mean += projection_norm/norm_gradient   
            model.train()


        bulk.append(mean/counter)
        # grad = [param.grad for param in model.parameters()]
        # grads.append(grad)
        # model.eval()
        #model.to('cuda:0')
        #model2 = model.cuda()
        #model.eval()
        # hess_object = hessian(model, criterion, data = (imagesall, labelsall), cuda = True)
        # top_eigenvalues, top_eigenvector = hess_object.eigenvalues(top_n = 10)
        # hessiana.append([top_eigenvalues, top_eigenvector])
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        #model.to('cpu')
        # Validation
        # model.eval()
        # test_loss = 0.0
        # correct = 0
        # total = 0
        # with torch.no_grad():
        #     for images, labels in test_loader:
        #         outputs = model(images)
        #         loss = criterion(outputs, labels)
        #         test_loss += loss.item()
        #         _, predicted = torch.max(outputs.data, 1)
        #         total += labels.size(0)
        #         correct += (predicted == labels).sum().item()

        # test_loss = test_loss / len(test_loader)
        # test_losses.append(test_loss)

        # accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}")
        print(bulk[-1])

    bulkk = [f.cpu() for f in bulk]
    print(bulkk)
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6), nrows=2, ncols=1)
    ax[0].plot(train_losses, label='Training Loss')
    # ax[0].plot(test_losses, label='Test Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[1].plot(bulkk, label='f_t')
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('f_t')
    ax[1].legend()

    fig.savefig('./hessian.png', dpi=300)


if __name__ == '__main__':
    cudnn.benchmark = True
    main()