from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Network settings
    batch_size: int = 8     # Number of images in each batch
    num_workers: int = 4
    learning_rate = 0.001    # Learning rate in the optimizer
    momentum = 0.9           # Momentum in SGD
    num_epochs = 2           # Number of passes over the entire dataset
    print_every_iters = 100


    # Output Handling
    out_dir = Path('C:/Users/tovon/projects_programming/IACV/ex4_dl_classification/output/')
    generate_submission = True

