from rich import print
import torch
import torch.nn as nn
from dataset import getMNIST
from lenet5.nn import ConvNeuralNet


def getTorchDevice():
    if not torch.backends.mps.is_available():
        device = torch.device("cpu")
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                  "built with MPS enabled.")

        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                  "and/or you do not have an MPS-enabled device on this machine.")

    else:
        mps_device = torch.device("mps")
    if (mps_device is not None):
        device = mps_device

    return device


def trainModel(num_epochs, model, train_loader, learning_rate):
    # Setting the loss function
    cost = nn.CrossEntropyLoss()

    # Setting the optimizer with the model parameters and learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # this is defined to print how many steps are remaining when training
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = cost(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 400 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


def test(model, test_loader):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: {} %'.format(
            100 * correct / total))


if __name__ == "__main__":
    print("COEN 6331 - LeNet5")
    batch_size = 64
    num_classes = 10
    learning_rate = 0.001
    num_epochs = 100
    device = getTorchDevice()
    print(f"Torch using device: {device}.")

    train_dataset, test_dataset, train_loader, test_loader = getMNIST(
        batch_size=batch_size)

    model = ConvNeuralNet(num_classes=num_classes).to(device)

    SHOULD_TRAIN = True

    if (SHOULD_TRAIN):
        trainModel(num_epochs=num_epochs, model=model,
                   train_loader=train_loader, learning_rate=learning_rate)
        model.save("lenet5.pth")

    model.load("lenet5.pth")
    test(model=model, test_loader=test_loader)
