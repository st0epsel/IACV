import numpy as np

from kmeans import (
    compute_distance,
    kmeans_fit,
    kmeans_predict_idx,
    kNN,
)

from extract_patches import extract_patches

from structures import FS




class ImageSegmenter:
    def __init__(self, mode='kmeans'):
        """ Feel free to add any hyperparameters to the ImageSegmenter.
            
            But note:
            For the final submission the default hyperparameteres will be used.
            In particular the segmetation will likely crash, if no defaults are set.
        """
        self.mode = mode

        # During evaluation, this will be replaced by a generator with different
        # random seeds. Use this generator, whenever you require random numbers,
        # otherwise your score might be lower due to stochasticity
        self.rng = np.random.default_rng(42)
        
    def extract_features_(self, sample_dd, feature_spaces=(FS.INTENSITY,)):
        """
        Extract features in feature_spaces from the RGB image
        Args:
            sample_dd       ... n_channels x H x W x C
            features        ... selection of ("intensity", "color", "position", "texture")
        Returns:
            features        ... n_samples x n_features
        """
        img = sample_dd['img']
        H, W, C = img.shape

        feature_dims = {
            FS.INTENSITY: 1,
            FS.COLOR: C,
            FS.POSITION: 2,
            FS.TEXTURE: 2
        }

        n_feature_dims = sum(feature_dims[feature] for feature in feature_spaces)
        features_array = np.zeros(H*W, n_feature_dims)
        dim_counter = 0

        # Iterate over feature spaces
        for feature_space in feature_spaces:
            if feature_space == FS.INTENSITY:
                feature = np.average(img, axis=2).flatten()
            elif feature_space == FS.COLOR:
                feature = img.reshape(H*W, C)
            elif feature_space == FS.POSITION:
                x_values = np.arange(W)
                y_values = np.arange(H)
                x_coords, y_coords = np.meshgrid(x_values, y_values)
                feature = np.dstack([x_coords, y_coords])
            else:
                assert True, f"feature space {feature_space} is not defined"
                feature = np.zeros((H * W, feature_dims[FS.TEXTURE]))

            dim_counter += feature_dims[feature_space]
            features_array[:, dim_counter:dim_counter+feature_dims[feature_space]] = feature

        return features_array

    def normalize_features(self, sample_dd, feature_spaces, features):
        """
        Args:
            sample_dd       ... n_channels x H x W x C
            feature_spaces  ... selection of ("intensity", "color", "position", "texture")
            features        ... n_samples x n_features

        Returns:
            normalized_features        ... n_samples x n_features
        """
        img = sample_dd['img']
        H, W, C = img.shape

        feature_ranges = {
            FS.INTENSITY: [0, 255],
            FS.COLOR: [[0, 255], [0, 255], [0, 255]],
            FS.POSITION: [[0, H], [0, W]],
            FS.TEXTURE: [[-255, 255], [-255, 255]]
        }

        feature_dims = {
            FS.INTENSITY: 1,
            FS.COLOR: C,
            FS.POSITION: 2,
            FS.TEXTURE: 2
        }

        n_samples = H * W
        normalized_features = np.zeros_like(features, dtype=float)

        # Possibly write this in vector form
        for n_sample in range(n_samples):
            for n_feature in range(n_feature):
                normalized_features[n_feature] = np.interp(
                    features[n_feature],
                    self.feature_range,
                    np.array([0, 1])
                )
        normalized_features *= self.weight
        return normalized_features
    
    def segment_image_dummy(self, sample_dd):
        return sample_dd['scribble_fg']

    def segment_image_kmeans(self, sample_dd):
        """ Segment images using k means """
        H, W, C = sample_dd['img'].shape
        feature_spaces = (FS.COLOR, FS.POSITION)
        features = self.extract_features_(sample_dd, feature_spaces=feature_spaces)


        # For now return scribble
        return self.segment_image_dummy(sample_dd)

    def segment_image(self, sample_dd):
        """ Feel free to add other methods """
        if self.mode == 'dummy':
            return self.segment_image_dummy(sample_dd)
        
        elif self.mode == 'kmeans':
            return self.segment_image_kmeans(sample_dd)
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")



