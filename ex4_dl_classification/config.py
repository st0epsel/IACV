from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class Config:
    # Network settings
    batch_size: int = 8     # Number of images in each batch
    num_workers: int = 3
    learning_rate = 0.001    # Learning rate in the optimizer
    momentum = 0.9           # Momentum in SGD
    num_epochs = 2           # Number of passes over the entire dataset
    print_every_iters = 100


    # Output Handling
    out_dir = Path('C:/Users/tovon/projects_programming/IACV/ex4_dl_classification/output/')
    generate_submission = True
    Plot_training_curves = True

    # Create output directory if it doesn't exist
    
    SESSION_NAME = "getting_it_to_work"

    id = 0
    while os.path.isdir(os.path.join(out_dir, f"{SESSION_NAME}_{id}")):
        id += 1
    os.makedirs(os.path.join(out_dir, f"{SESSION_NAME}_{id}"))

    SESSION_FOLDER = os.path.join(out_dir, f"{SESSION_NAME}_{id}")


