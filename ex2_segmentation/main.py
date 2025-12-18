import sys
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Now we can import our other files
from utils import (
    compute_iou,
    evaluate_segmentation,
    load_sample,
    show_sample
)

from kmeans import kmeans_fit, check_kmeans, kNN, check_kNN

from image_segmenter import ImageSegmenter

iacv_path = Path(r'C:/Users/tovon/IACV/ex2_segmentation')
data_path = Path(r'C:/Users/tovon/IACV/ex2_segmentation/data')

if str(iacv_path) not in sys.path:
    sys.path.append(str(iacv_path))


# Go through 'val' folder and get names of subdirectories - one subdir is one sample
sample_dirs = [dd for dd in data_path.iterdir() if dd.is_dir()]

sample_path = data_path / '04'
sample_dd = load_sample(sample_path)
show_sample(sample_dd, show_scribble=True)



# This code tests your kNN implementation
check_kNN(kNN, display_prediction=True)

# Run your segmentation algorithm
segmenter = ImageSegmenter(k_fg=2, k_bg=5, mode='kmeans')

mask_pred = segmenter.segment_image(sample_dd)

iou_score = compute_iou(
    sample_dd['mask_true'].astype(bool),
    mask_pred.astype(bool)
)

# Visualize your prediction
show_sample(sample_dd, mask_pred=mask_pred)
print(f'IoU score is: {iou_score:0.3f}')