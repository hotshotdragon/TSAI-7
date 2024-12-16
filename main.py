import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from model import Net
from dataset import get_dataloaders
from transforms import get_train_transforms, get_test_transforms
from train_test import train, test
from torchsummary import summary

def main():
    # Constants
    SEED = 1
    BATCH_SIZE = 128
    LEARNING_RATE = 0.01
    MOMENTUM = 0.9
    EPOCHS = 15
    DROPOUT_VALUE = 0.05

    # Set random seed for reproducibility
    torch.manual_seed(SEED)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(SEED)

    # Get data loaders
    train_transforms = get_train_transforms()
    test_transforms = get_test_transforms()
    train_loader, test_loader = get_dataloaders(train_transforms, test_transforms, BATCH_SIZE, cuda)

    # Initialize model, optimizer, and scheduler
    device = torch.device("cuda" if cuda else "cpu")
    model = Net(DROPOUT_VALUE).to(device)

    # Print model summary
    summary(model, input_size=(1, 28, 28))

    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    # scheduler = StepLR(optimizer, step_size=4, gamma=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam usually works well with lr=0.001
    scheduler = StepLR(optimizer, step_size=4, gamma=0.1)

    # Training and Testing
    train_losses, train_acc = [], []
    test_losses, test_acc = [], []

    for epoch in range(1, EPOCHS + 1):
        print(f"EPOCH: {epoch}")
        epoch_train_losses, epoch_train_acc = train(model, device, train_loader, optimizer)
        train_losses.extend(epoch_train_losses)
        train_acc.extend(epoch_train_acc)
        scheduler.step()
        epoch_test_losses, epoch_test_acc = test(model, device, test_loader)
        test_losses.extend(epoch_test_losses)
        test_acc.extend(epoch_test_acc)

if __name__ == '__main__':
    main() 