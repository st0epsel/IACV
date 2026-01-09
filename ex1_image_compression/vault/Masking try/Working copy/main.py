import os
import sys
import random
import numpy as np
from pathlib import Path

# Import evaluation package
import eval_utils
from image_compressor import ImageCompressor, ImageReconstructor
from helper_fcts import load_image, show_images, img_to_vec, vec_to_img


iacv_path = Path('C:/Users/tovon/IACV/ex1_image_compression')
data_path = Path('C:/Users/tovon/IACV/ex1_image_compression/data')


# Add the handout folder to python paths
if str(iacv_path) not in sys.path:
    sys.path.append(str(iacv_path))

# Load images
print("Loading train images")
train_path = data_path / 'train_n'
train_images = {f.stem: load_image(str(f)) for f in train_path.glob('*.png')}

print("Loading test images")
test_path = data_path / 'test_n'
test_images = {f.stem: load_image(str(f)) for f in test_path.glob('*.png')}

print("Loading val images")
val_path = data_path / 'val_n'
val_images = {f.stem: load_image(str(f)) for f in val_path.glob('*.png')}

print("Loading val ground truths")
val_path_c = data_path / 'val_c'
val_gt = {f.stem: load_image(str(f)) for f in val_path_c.glob('*.png')}

# Show random ground truth
key = random.choice(list(val_gt.keys()))

# show_images([val_images[key], val_gt[key]], ['Noisy', 'Ground Truth'])


# Initialize the compressor object and learn the codebook
compressor = ImageCompressor()
compressor.set_k(50)
# print(f"list(train_images.values(): \n {list(train_images.values())}")
compressor.train(list(train_images.values()))

# get the learnt codebook
codebook = compressor.get_codebook()

# Initialize the reconstructor (This will be done on the server side)
reconstructor = ImageReconstructor(codebook)

### Compress Images
key = random.choice(list(val_gt.keys()))

img_in = val_images[key]
img_gt = val_gt[key]

img_compressed = compressor.compress(img_in)

# Reconstruct image
img_rec = reconstructor.reconstruct(img_compressed)
show_images([img_in, img_rec, img_gt], ['Input', 'Reconstructed', 'Ground Truth'])

rmse = eval_utils.compute_rmse(img_rec, img_gt)
img_code_size = img_compressed.nbytes
codebook_size = compressor.get_codebook().nbytes
w_score, w_rmse, w_img_code_size, w_codebook_size = eval_utils.weight_scores(
    rmse, img_code_size, codebook_size
)

print(f'Evaluation Score is: {w_score:0.3f}')
print()
print(f'Weighted Reconstruction Error is: {w_rmse:0.3f}')
print(f'Weighted Compressed Image Size is: {w_img_code_size}')
print(f'Weighted Codebook size is: {w_codebook_size}')

