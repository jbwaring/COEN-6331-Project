from rich import print
import torch
import torch.nn as nn
from dataset import getMNIST
from lenet5.nn import ConvNeuralNet
from utils import getTorchDevice




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
    num_epochs = 10
    device = getTorchDevice()
    print(f"Torch using device: {device}.")

    train_dataset, test_dataset, train_loader, test_loader = getMNIST(
        batch_size=batch_size)

    model = ConvNeuralNet(num_classes=num_classes).to(device)

    # Training and Testing
    SHOULD_TRAIN = False
    SHOULD_TEST = False
    if (SHOULD_TRAIN):
        trainModel(num_epochs=num_epochs, model=model,
                   train_loader=train_loader, learning_rate=learning_rate)
        model.save("lenet5.pth")

    model.load("lenet5.pth")
    if (SHOULD_TEST):
        test(model=model, test_loader=test_loader)

    # # Inference with only two first layers:
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     for images, labels in test_loader:
    #         images = images.to(device)
    #         labels = labels.to(device)
    #         outputs = model(images)
    #         print(outputs)
