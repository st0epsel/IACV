import os
import sys
import random
from pathlib import Path


import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
%matplotlib inline

# Import package
import eval_utils
from helper_fcts import load_image, show_images
from image_compressor import ImageCompressor, ImageReconstructor


iacv_path = Path('C:/Users/tovon/IACV/ex1_image_compression')
data_path = Path('C:/Users/tovon/IACV/ex1_image_compression/data')





if __name__ == "__main__":
    # Add the handout folder to python paths
    if str(iacv_path) not in sys.path:
        sys.path.append(str(iacv_path))

    #Load the data
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

    key = random.choice(list(val_gt.keys()))
    show_images([val_images[key], val_gt[key]], ['Noisy', 'Ground Truth'])


    # Initialize the compressor object and learn the codebook
    compressor = ImageCompressor()
    compressor.train(list(train_images.values()))

    # get the learnt codebook
    codebook = compressor.get_codebook()

    # Initialize the reconstructor (This will be done on the server side)
    reconstructor = ImageReconstructor(codebook)

    key = random.choice(list(val_gt.keys()))

    img_in = val_images[key]
    img_gt = val_gt[key]
    img_compressed = compressor.compress(img_in)

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



    from image_compressor import ImageCompressor, ImageReconstructor

    # Learn the compressor
    compressor = ImageCompressor()
    compressor.train(list(train_images.values()))

    # Compress all images
    img_code = {k: compressor.compress(v) for k, v in val_images.items()}

    codebook = compressor.get_codebook()

    # Init reconstructor - will be done on server side
    reconstructor = ImageReconstructor(codebook)

    eval_score, recon_error, compressed_im_size, codebook_size = eval_utils.compute_evaluation_score(img_code, val_gt, reconstructor)

    print(f'Evaluation Score is: {eval_score:0.3f}')
    print(f'Mean Reconstruction Error is: {recon_error:0.3f}')
    print(f'Mean Compressed Image Size is: {compressed_im_size}')
    print(f'Codebook size is: {codebook_size}')



    from image_compressor import ImageCompressor
    import shutil

    compressor = ImageCompressor()
    compressor.train(list(train_images.values()))

    out_dir = env_path / 'submission'
    out_dir.mkdir(exist_ok=True)

    for img_id, img in test_images.items():
        img_code = compressor.compress(img)
        np.save(out_dir / f'{img_id}.npy', img_code)

    codebook = compressor.get_codebook()
    np.save(out_dir / 'codebook.npy', codebook)

    shutil.copyfile(env_path / 'image_compressor.py', out_dir / 'image_compressor.py')