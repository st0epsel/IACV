import numpy as np
from pathlib import Path

from utils import compute_iou, show_sample, show_image, evaluate_segmentation, load_sample

from kmeans import (
    compute_distance,
    kmeans_fit,
    kmeans_predict_idx,
    kNN,
    check_kNN,
    check_kmeans
)
from extract_patches import extract_patches
from enum import IntEnum, unique


@unique
class FS(IntEnum):
    INTENSITY = 0
    RGB = 1
    TEXTURE = 2
    POSITION = 3
    HSL = 4


class ImageSegmenter:
    def __init__(self, k_fg, k_bg, mode='kmeans'):
        """ Feel free to add any hyperparameters to the ImageSegmenter.

            But note:
            For the final submission the default hyperparameteres will be used.
            In particular the segmetation will likely crash, if no defaults are set.
        """
        self.mode = mode
        self.k_fg = k_fg
        self.k_bg = k_bg

        # During evaluation, this will be replaced by a generator with different
        # random seeds. Use this generator, whenever you require random numbers,
        # otherwise your score might be lower due to stochasticity
        self.rng = np.random.default_rng(42)

    def extract_features_(self, sample_dd):
        """
        Extract (and normalize) features (color & position) from the RGB image
        Args:
            sample_dd       ... n_channels x H x W x C
        Returns:
            features        ... n_samples x 5 (2x position, 3x color)
        """

        feature_space = FeatureSpace(sample_dd, (FS.POSITION, FS.HSL))

        features_array = feature_space.extract_features()
        normalized_features_array = feature_space.normalize(features_array)

        return normalized_features_array

    def segment_image_dummy(self, sample_dd):
        return sample_dd['scribble_fg']

    def segment_image_kmeans(self, sample_dd):
        """ Segment images using k means """
        H, W, C = sample_dd['img'].shape
        features = self.extract_features_(sample_dd)

        fg_scribble = sample_dd['scribble_fg'].flatten()
        bg_scribble = sample_dd['scribble_bg'].flatten()

        fg_scribble_idx = np.where(fg_scribble == 255)
        bg_scribble_idx = np.where(bg_scribble == 255)

        features_fg = features[fg_scribble_idx]
        features_bg = features[bg_scribble_idx]
        data_train = np.concatenate((features_fg, features_bg))

        nr_features_fg, nr_features_bg = features_fg.shape[0], features_bg.shape[0]

        labels_fg = np.ones(nr_features_fg)
        labels_bg = np.zeros(nr_features_bg)
        labels = np.concatenate((labels_fg, labels_bg))

        data_test = features

        mask = kNN(data_train, labels, data_test).reshape((H, W))

        return mask

    def segment_image(self, sample_dd):
        """ Feel free to add other methods """
        if self.mode == 'dummy':
            return self.segment_image_dummy(sample_dd)

        elif self.mode == 'kmeans':
            return self.segment_image_kmeans(sample_dd)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class FeatureSpace:
    def __init__(self, sample_dd, feature_space, weights=None):
        """
        Args:
            features        ... Selection of FS.INTENSITY, FS.COLOR, FS.TEXTURE and/or FS.POSITION
            sample_dd       ... n_channels x H x W x C
        """
        self.feature_space = feature_space
        img = sample_dd['img']
        H, W, C = img.shape

        feature_dim_list = {
            FS.INTENSITY: 1,
            FS.RGB: C,
            FS.POSITION: 2,
            FS.TEXTURE: 2,
            FS.HSL: 3
        }
        feature_range_list = {
            FS.INTENSITY: [[0, 255]],
            FS.RGB: [[0, 255] for _ in range(C)],
            FS.POSITION: [[0, W], [0, H]],
            FS.TEXTURE: [[-255, 255], [-255, 255]],
            FS.HSL: [[0, 1], [0, 1], [0, 1]]
        }
        default_feature_weight_list = {
            FS.INTENSITY: [1],
            FS.RGB: [1 for _ in range(C)],
            FS.POSITION: [1, W/H],
            FS.TEXTURE: [1, 1],
            FS.HSL: [1, 2.6, 1.7]
        }

        if weights:
            self.feature_weight_list = weights
        else:
            self.feature_weight_list = default_feature_weight_list

        n_feature_dims = 0
        for feature in feature_space:
            n_feature_dims += feature_dim_list[feature]

        self.weights = weights
        self.feature_dim_list = feature_dim_list
        self.feature_range_list = feature_range_list
        self.n_feature_dims = n_feature_dims
        self.img = sample_dd["img"]
        self.H = H
        self.W = W
        self.C = C

    def extract_features(self):
        """
        Extract (and normalize) features (color & position) from the RGB image
        Returns:
            features        ... n_samples x feature_space_dims (in provided order)
        """
        features_data = np.zeros((self.H * self.W, self.n_feature_dims))
        feature_start_pos = 0

        for feature in self.feature_space:
            feature_end_pos = feature_start_pos + self.feature_dim_list[feature]
            if feature == FS.RGB:
                features_data[:, feature_start_pos: feature_end_pos] = self.extract_RGB()
            elif feature == FS.POSITION:
                features_data[:, feature_start_pos: feature_end_pos] = self.extract_position()
            elif feature == FS.INTENSITY:
                features_data[:, feature_start_pos: feature_end_pos] = self.extract_intensity()
            elif feature == FS.HSL:
                features_data[:, feature_start_pos: feature_end_pos] = self.extract_HSL()
            feature_start_pos = feature_end_pos

        return features_data

    def normalize(self, features_data):
        """
        Returns:
            normalized_features ... Array of shape n_samples x feature_dims
        """
        normalized_features = np.zeros_like(features_data, dtype=float)

        feature_start_pos = 0
        for feature in self.feature_space:
            feature_end_pos = feature_start_pos + self.feature_dim_list[feature]
            for i in range(self.feature_dim_list[feature]):
                dim = feature_start_pos + i
                normalized_features[:, dim] = np.interp(
                    features_data[:, dim],
                    self.feature_range_list[feature][i],
                    np.array([0, 1])
                ) * self.feature_weight_list[feature][i]
            feature_start_pos = feature_end_pos
        return normalized_features

    def de_normalize(self, normalized_feature_data):
        """
        Args:
            normalized_features ... Array of shape n_samples x feature_dims

        Returns:
            features         ... Array of shape n_samples x feature_dims
        """
        features_data = np.zeros_like(normalized_feature_data, dtype=float)

        # Possibly write this in vector form
        feature_start_pos = 0
        for feature in self.feature_space:
            feature_end_pos = feature_start_pos + self.feature_dim_list[feature]
            for i in range(self.feature_dim_list[feature]):
                dim = feature_start_pos + i
                features_data[:, dim] = np.interp(
                    normalized_feature_data[:, dim],
                    self.feature_range_list[feature][i],
                    np.array([0, 1])
                ) / self.feature_weight_list[feature][i]
            feature_start_pos = feature_end_pos
        return features_data

    def extract_RGB(self):
        """
        Returns             ... Array of shape n_samples x 3
        """
        color = self.img.reshape(self.H * self.W, self.C)
        return color

    def extract_intensity(self):
        """
        Returns             ... Array of shape n_samples
        """
        intensity = np.average(self.extract_RGB(), axis=1)
        intensity = intensity.reshape((intensity.shape[0], 1))
        return intensity

    def extract_position(self):
        """
        Returns             ... Array of shape n_samples x 2
        """
        x_values = np.arange(self.W)
        y_values = np.arange(self.H)
        x_coords, y_coords = np.meshgrid(x_values, y_values)
        position = np.dstack([x_coords, y_coords])
        position_flat = position.reshape(self.H * self.W, 2)
        return position_flat

    def extract_HSL(self):
        return self.rgb_to_hsl(self.img).reshape(self.H * self.W, 3)

    def rgb_to_hsl(self, rgb):
        """
        Converts an RGB image (uint8 [0, 255]) to HSL (float [0, 1]) in NumPy.

        Args:
            rgb: A NumPy array with shape (..., 3) containing RGB values.
                 Assumes a uint8 input in the range [0, 255].

        Returns:
            A NumPy array with the same shape as input, containing HSL values.
            H, S, and L are all in the range [0, 1].
        """
        # Normalize RGB values to [0, 1]
        rgb_norm = rgb.astype(float) / 255.0
        r, g, b = rgb_norm[..., 0], rgb_norm[..., 1], rgb_norm[..., 2]

        # Find min and max values
        vmax = np.max(rgb_norm, axis=-1)
        vmin = np.min(rgb_norm, axis=-1)
        delta = vmax - vmin

        # Calculate L (Lightness)
        l = (vmax + vmin) / 2.0

        with np.errstate(divide='ignore', invalid='ignore'):
            # Calculate S (Saturation)
            s = np.zeros_like(l)

            denominator_s = 1.0 - np.abs(2.0 * l - 1.0)

            s = np.where(
                denominator_s != 0,
                delta / denominator_s,
                0
            )

            # Calculate H (Hue)
            h = np.zeros_like(l)

            mask_r = (vmax == r) & (delta != 0)
            mask_g = (vmax == g) & (delta != 0)
            mask_b = (vmax == b) & (delta != 0)

            h_r = (((g - b) / delta) % 6) / 6.0
            h_g = (((b - r) / delta) + 2) / 6.0
            h_b = (((r - g) / delta) + 4) / 6.0

            # Apply the calculations using the masks
            h = np.where(mask_r, h_r, h)
            h = np.where(mask_g, h_g, h)
            h = np.where(mask_b, h_b, h)

        # Stack H, S, L
        hsl = np.zeros_like(rgb_norm)
        hsl[..., 0] = h
        hsl[..., 1] = s
        hsl[..., 2] = l

        return hsl


if __name__ == "__main__":
    segmenter = ImageSegmenter(mode='kmeans')
    data_path = Path(r'C:/Users/tovon/IACV/ex2_segmentation/data/')

    sample_dirs = [dd for dd in data_path.iterdir() if dd.is_dir()]

    score, time = evaluate_segmentation(segmenter, sample_dirs, 42, False)

    sample_dd = load_sample(sample_dirs[0])
    mask_pred = segmenter.segment_image(sample_dd)

    #show_sample(sample_dd, mask_pred=mask_pred)

    print(f"Mean IoU: {score:0.3f} with average time per image of {time:0.3f} s")




