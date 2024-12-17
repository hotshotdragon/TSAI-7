import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self,dropout_value = 0.05):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        )  # output_size = 26, RF = 3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        )  # output_size = 24, RF = 5

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2)  # output_size = 12, RF = 10

        # CONVOLUTION BLOCK 2
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        )  # output_size = 10, RF = 14

        # TRANSITION BLOCK 2
        self.pool2 = nn.MaxPool2d(2, 2)  # output_size = 5, RF = 28

        # Global Average Pooling (GAP)
        self.gap = nn.AdaptiveAvgPool2d(1)  # Output shape will be (batch_size, 16, 1, 1)

        # Final Classification Layer (Linear)
        self.fc = nn.Linear(16, 10)  # 16 channels after GAP, output size = 10 classes

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.pool2(x)

        x = self.gap(x)  # Apply Global Average Pooling, output shape will be (batch_size, 16, 1, 1)
        
        x = x.view(x.size(0), -1)  # Flatten the output from GAP for classification (batch_size, 16)

        x = self.fc(x)  # Classification layer

        return F.log_softmax(x, dim=-1)
