from torchvision import datasets
from torch.utils.data import DataLoader

def get_dataloaders(train_transforms, test_transforms, batch_size, cuda):
    train = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    test = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)

    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=0, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

    train_loader = DataLoader(train, **dataloader_args)
    test_loader = DataLoader(test, **dataloader_args)

    return train_loader, test_loader 