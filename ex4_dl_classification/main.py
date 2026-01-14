import sys
import shutil
from pathlib import Path
import matplotlib
import numpy as np
import time

import matplotlib.pyplot as plt
matplotlib.use('Agg')

import torch
from torch.utils.data import DataLoader

from dataset import ImageDataset
from utils import compute_accuracy, CLASSES
from transforms import get_transforms_train, get_transforms_val
from cnn_network import CNN, get_loss_function, get_optimizer



if __name__ == '__main__':

    def saveimg(ims, gt_labels, pred_labels=None, filename="output.png"):
        fig, ax = plt.subplots(1, len(ims), figsize=(20, 20))
        for id in range(len(ims)):
            im = ims[id]
            im = im / 2 + 0.5     # unnormalize
            im_np = im.numpy()

            ax[id].imshow(np.transpose(im_np, (1, 2, 0)))

            if pred_labels is None:
                im_title = f'GT: {CLASSES[gt_labels[id]]}'
            else:
                im_title = f'GT: {CLASSES[gt_labels[id]]}   Pred: {CLASSES[pred_labels[id]]}'
            ax[id].set_title(im_title)
        plt.savefig(Path(f"{Config.SESSION_FOLDER}/{filename}"))
        plt.close()

    from config import Config


    ex_path = Path(r"C:/Users/tovon/projects_programming/IACV/ex4_dl_classification")
    data_path = ex_path / Path('data')

    # Possibly append to PATH
    if str(ex_path) not in sys.path:
        sys.path.append(str(ex_path))

    start = time.time()
    program_start = start

    ##########################################
    # 1. Data Loading
    ##########################################
    print("1. Initializing Datasets...")

    # DataLoader for training data (Transforms include preprocessing + augmentation)
    dataset_train = ImageDataset(data_path/'train.csv', data_path/'train.hdf5', get_transforms_train())

    # DataLoader for validation data (Transforms include only preprocessing)
    dataset_valid = ImageDataset(data_path/'val.csv', data_path/'val.hdf5', get_transforms_val())

    batch_size = Config.batch_size
    num_workers = Config.num_workers

    # Generate our data loaders
    # We use the num_workers from Config.py. If it crashes again, set Config.num_workers to 0.
    train_loader = DataLoader(dataset_train, batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(dataset_valid, batch_size, shuffle=True, num_workers=num_workers)

    # get some random training images to verify loading works
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    saveimg(images[:6], labels[:6], filename="loaded_data")
    print(f'Image batch shape: {images.size()}')

    print(f"   Complete - took {time.time() - start:.2f} seconds\n")
    start = time.time()

    ##########################################
    # 2. Network Definition
    ##########################################
    print("2. Initializing CNN model...")

    cnn_network = CNN()  # create CNN model

    print("CNN Architecture:")
    print(cnn_network)

    print("")
    print("Testing network input / output.")
    test_input = torch.zeros(1, 3, 50, 50)  # batch size X 3 (RGB) X width X height
    print(f"Input shape: {test_input.shape}")
    test_output = cnn_network(test_input)  # batch size X # classes
    print(f"Output shape: {test_output.shape}")

    assert tuple(test_output.shape) == (1, 6), "Output shape should be [1, 6]"
    print(f"   Complete - took {time.time() - start:.2f} seconds\n")
    start = time.time()
    

    ##########################################
    # 3. Cost Function Definition & Optimizer
    ##########################################

    print("3. Initializing Loss Function and Optimizer...")
    criterion = get_loss_function()  # get loss function
    optimizer = get_optimizer(cnn_network, lr=Config.learning_rate, momentum=Config.momentum)  # get optimizer
    print(f"   Complete - took {time.time() - start:.2f} seconds\n")
    start = time.time()

    ##########################################
    # 4. Training Loop
    ##########################################
    print("4. Starting Training Loop...")
    
    # Use settings from Config class
    num_epochs = Config.num_epochs
    print_every_iters = Config.print_every_iters

    print(f"Starting training for {num_epochs} epochs...")

    training_loss_per_epoch = []
    val_loss_per_epoch = []

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        # Put the model in training mode
        cnn_network.train()

        # First we loop over training dataset
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()  # zero the gradients from previous iteration

            # forward + backward + optimize
            outputs = cnn_network(inputs)  # forward pass to obtain network outputs
            loss = criterion(outputs, labels)  # compute loss with respect to labels
            loss.backward()  # compute gradients with backpropagation (autograd)
            optimizer.step()  # optimize network parameters

            # print statistics
            running_loss += loss.item()
            if (i + 1) % print_every_iters == 0:
                print(
                    f'[Epoch: {epoch + 1} / {num_epochs},'
                    f' Iter: {i + 1:5d} / {len(train_loader)}]'
                    f' Training loss: {running_loss / (i + 1):.3f}'
                )

        mean_loss = running_loss / len(train_loader)
        training_loss_per_epoch.append(mean_loss)

        # Put model in evaluation mode for validation
        cnn_network.eval()

        # Loop over validation dataset
        running_loss = 0.0
        for i, data in enumerate(valid_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # on validation dataset, we only do forward, without computing gradients
            with torch.no_grad():
                outputs = cnn_network(inputs)
                loss = criterion(outputs, labels)

            # print statistics
            running_loss += loss.item()

        mean_loss = running_loss / len(valid_loader)
        val_loss_per_epoch.append(mean_loss)

        print(
            f'[Epoch: {epoch + 1} / {num_epochs}]'
            f' Validation loss: {mean_loss:.3f}'
        )

    print('Finished Training')

    # Plot the training curves
    if Config.Plot_training_curves:
        plt.figure()
        plt.plot(np.array(training_loss_per_epoch))
        plt.plot(np.array(val_loss_per_epoch))
        plt.legend(['Training loss', 'Val loss'])
        plt.xlabel('Epoch')
        plt.savefig(Path(Config.SESSION_FOLDER) / 'training_curves.png') 
        plt.close() # Closes the figure to free memory

    print(f"   Complete - took {time.time() - start:.2f} seconds\n")
    start = time.time()

    ##########################################
    # 5. Network Testing & Saving
    ##########################################
    print("5. Testing network on validation data...")

    # get a few validation samples
    images, labels = next(iter(valid_loader))

    # get network output
    outputs = cnn_network(images)  # classification scores
    _, predicted = torch.max(outputs, 1)  # use maximum as prediction label

    # visualize ground truth and predictions
    saveimg(images[:4], labels[:4], predicted[:4], filename="predictions.png")

    acc = compute_accuracy(cnn_network, valid_loader)
    print(f'Accuracy of the network on the 1500 validation images: {acc:.2f} %')

    out_dir = Config.out_dir 

    cnn_network.write_weights(out_dir / 'checkpoint.pt')
    shutil.copyfile(ex_path / 'cnn_network.py', out_dir / 'cnn_network.py')
    shutil.copyfile(ex_path / 'transforms.py', out_dir / 'transforms.py')
    
    print(f"Saved checkpoint and scripts to {out_dir}")
    print(f"   Complete - took {time.time() - start:.2f} seconds")
    print(f" Total program time: {time.time() - program_start:.2f} seconds")