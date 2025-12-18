import numpy as np
from enum import IntEnum, unique

@unique
class FS(IntEnum):
    INTENSITY = 0
    RGB = 1
    TEXTURE = 2
    POSITION = 3
    HSL = 4


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
        return rgb_to_hsl(self.img).reshape(self.H * self.W, 3)



def rgb_to_hsl(rgb):
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

    # Calculate S (Saturation)
    s = np.zeros_like(l)

    # Calculate s for non-gray colors
    s = np.where(
        delta != 0,
        delta / (1.0 - np.abs(2.0 * l - 1.0)),
        0
    )

    # Calculate H (Hue)
    h = np.zeros_like(l)

    # Create masks for which channel was the max (and where delta is not zero)
    mask_r = (vmax == r) & (delta != 0)
    mask_g = (vmax == g) & (delta != 0)
    mask_b = (vmax == b) & (delta != 0)

    # Calculate H for each case
    # The (x % 6) / 6.0 converts the [0, 6] range to [0, 1]
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