import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class CNN(nn.Module):
    """Convolutional Neural Network.
    
    We provide a simple network with a Conv layer, followed by pooling,
    and a fully connected layer. Modify this to test different architectures,
    and hyperparameters, i.e. different number of layers, kernel size, feature
    dimensions etc.

    See https://pytorch.org/docs/stable/nn.html for a list of different layers
    in PyTorch.
    """

    def __init__(self):
        super().__init__()
        
        # ---------------------------------------------------------
        # 1. The Feature Extractor ("The Eye")
        # ---------------------------------------------------------
        self.features = nn.Sequential(
            # Block 1: Input 3x50x50 -> Output 32x25x25
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), # Optional: Helps training stabilize faster
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: Input 32x25x25 -> Output 64x12x12
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: Input 64x12x12 -> Output 128x6x6
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # ---------------------------------------------------------
        # 2. The Classifier ("The Brain")
        # ---------------------------------------------------------
        # Calculation: 128 channels * 6 width * 6 height = 4608
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 512), # Dense layer
            nn.ReLU(),
            nn.Dropout(0.5),             # Prevents overfitting
            nn.Linear(512, 6)            # Output layer (6 classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def write_weights(self, fname):
        """ Store learned weights of CNN """
        torch.save(self.state_dict(), fname)

    def load_weights(self, fname):
        """
        Load weights from file in fname.
        The evaluation server will look for a file called checkpoint.pt
        """
        ckpt = torch.load(fname, weights_only=True)
        self.load_state_dict(ckpt)


def get_loss_function():
    """Return the loss function to use during training. We use
       the Cross-Entropy loss for now.
    
    See https://pytorch.org/docs/stable/nn.html#loss-functions.
    """
    return nn.CrossEntropyLoss()


def get_optimizer(network, lr=0.001, momentum=0.9):
    """Return the optimizer to use during training.
    
    network specifies the PyTorch model.

    See https://pytorch.org/docs/stable/optim.html#how-to-use-an-optimizer.
    """

    # The fist parameter here specifies the list of parameters that are
    # learnt. In our case, we want to learn all the parameters in the network
    return optim.SGD(network.parameters(), lr=lr, momentum=momentum)
