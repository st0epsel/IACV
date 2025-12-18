import numpy as np
import matplotlib.pyplot as plt

from utils import load_sample


def kmeans_fit(data, k, rng, n_iter=500, tol=1.e-4):
    """
    Fit kmeans
    Args:
        data    ... Array of shape n_samples x n_features
        k       ... Number of clusters
        rng     ... Random number generator object from NumPy
        n_iter  ... Maximum number of iterations
        tol     ... Stop if total change in centroids is smaller than tol

    Intermediate:
        closest_centroid ... Array of shape n_samples (containing the closest centroid index)
        
    Returns:
        centers   ... Cluster centers. Array of shape k x n_features
    """
    n_samples, n_features = data.shape
    
    # Initialise clusters - use the provided random number generator
    centroids = data[rng.choice(n_samples, k, replace=False)]

    # Iterate the k-means update steps
    for i in range(n_iter):  # Terminating loop
        # Assign points to closest cluster centers
        closest_centroid = kmeans_predict_idx(data, centroids)

        # Find center points of new clusters and assign to be new centroids
        new_centroids = np.zeros_like(centroids)
        for cluster_index in range(k):
            cluster_points = data[closest_centroid == cluster_index]
            if cluster_points.shape[0] > 0:  # Check if cluster is not empty
                new_centroids[cluster_index] = np.mean(cluster_points, axis=0)

        if np.sum(np.abs(new_centroids-centroids)) < tol:
            centroids = new_centroids
            break

        centroids = new_centroids
    # Return cluster centers
    return centroids


def compute_distance(data, clusters):
    """
    Compute all distances of every sample in data, to every center in clusters.
    Args:
        data     ... n_samples x n_features
        clusters ... n_clusters x n_features

    Returns:
        distances ... n_samples x n_clusters
    """
    # Since centroids change every iteration of k means, pre-computation is too memory-write intense
    # But vectorizsation is more important
    # Use distance squared (less computation)
    data_sq_norms = np.sum(data ** 2, axis=1).reshape(-1, 1)
    cluster_sq_norms = np.sum(clusters ** 2, axis=1).reshape(1, -1)
    dot_product = 2 * np.dot(data, clusters.T)
    distances = (data_sq_norms + cluster_sq_norms - dot_product)**2
    return distances

def kmeans_predict_idx(data, clusters):
    """
    Predict index of closest cluster for every sample
    
    Args:
        data     ... n_samples x n_features
        clusters ... n_clusters x n_features
    Returns:
        idx      ... r n_samples x 1 (cluster index)
    """
    distances = compute_distance(data, clusters)
    idx = np.argmin(distances, axis=1)
    return idx


def kNN(data_train, labels_train, data_test, k: int = 1):
    """
    Function to run k nearest neighbor prediction.
    No need to implement this for k > 1.

    Args:
        data_train    ... n_training_samples x n_features (training example features)
        labels_train  ... n_training_samples (training sample labels)
        data_test     ... n_testing_samples x n_features (query vector??)
        k             ... Find k nearest neighbors & predict by majority voting

    Returns:
        labels_test   ... n_testing_samples
    """
    prediction_indices = kmeans_predict_idx(data_test, data_train)
    predictions = labels_train[prediction_indices]

    return predictions


def check_kmeans(kmeans_fit_fn, val_path, max_iter=500):
    """
    Here we provide the centroids, which the baseline code extracts from some
    validation images, if all pixels are used to fit the kmeans model with K = 4

    Args:
        kmeans_fit_fn   ... The function fitting the kmeans model
        val_path        ... Path to validation images
    """
    rng = np.random.default_rng(42)

    # Centroids extracted by baseline code
    baseline_centroids = {
        "01": np.array(
            [
                [0.16280249, 0.11134844, 0.10853657],
                [0.39967992, 0.21585965, 0.20279051],
                [0.78526716, 0.72989798, 0.67762712],
                [0.61857210, 0.54688075, 0.49287591],
            ]
        ),
        "02": np.array(
            [
                [0.69110873, 0.62125490, 0.53331500],
                [0.83364685, 0.77513072, 0.70440564],
                [0.20545708, 0.15125980, 0.14535034],
                [0.46416962, 0.39247356, 0.34935396],
            ]
        ),
    }

    # Extract centroids
    for idx in ["01", "02"]:
        sample_dd = load_sample(val_path / idx)
        val_img = sample_dd["img"].astype(float) / 255.
        val_img = val_img.reshape(-1, 3)

        centroids = kmeans_fit_fn(val_img, k=4, rng=rng, n_iter=max_iter, tol=1.e-6)
        assert centroids.shape == (4, 3), "Centroids have the wrong shape!"
        
        if not np.allclose(centroids, baseline_centroids[idx]):
            print(f"Check failed for validation image {idx}.")
            print(f"Computed centroids are:\n {centroids}")
            print(f"Expected centroids are:\n {baseline_centroids[idx]}")
            assert False, "Check failed :-("

    print("Check passed :-)")
    return None


def check_kNN(kNN_fn, display_prediction=True):
    """
    Check kNN implementation for some toy data
    """
    X_train = np.array(
        [
            [-1.12589499,  0.11954409],
            [-0.11427603,  0.31518343],
            [ 0.31733705, -0.31019391],
            [ 1.16328902, -0.80477520],
            [ 0.47749020,  0.19007394],
            [ 1.85933124,  0.07207587],
        ]
    )

    X_test = np.array(
        [
            [-0.19520917,  0.78595339],
            [ 0.59093115,  0.60207282],
            [-0.08560921,  0.85151863],
            [ 1.76291504, -0.00500372],
        ]
    )

    y_train = np.array([0, 1, 1, 1, 0, 1])
    y_pred_baseline = np.array([1, 0, 1, 1])

    y_pred = kNN_fn(X_train, y_train, X_test, k=1)

    if display_prediction:
        colors = np.array(["#377eb8", "#ff7f00"])

        fig, ax = plt.subplots(1, 1)
        ax.scatter(
            X_train[:, 0],
            X_train[:, 1],
            s=10,
            c=colors[y_train],
            marker="o",
            label="Training data"
        )
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            s=20,
            c=colors[y_pred],
            marker="x",
            label="Testing data"
        )

        ax.set_title(
            "Colour of dots: Training labels\n"
            "Colour of crosses: Predicted labels"
        )

        # Avoid confusion in label colours
        leg = ax.legend()
        leg.legend_handles[0].set_facecolor('gray')
        leg.legend_handles[0].set_edgecolor('gray')

        leg.legend_handles[1].set_facecolor('gray')
        leg.legend_handles[1].set_edgecolor('gray')

    if np.allclose(y_pred, y_pred_baseline):
        print("Check passed :-)")
    else:
        print("Check failed :-(\n")
        print(f"Computed the following labels: {y_pred}")
        print(f"kNN for K = 1 should give: {y_pred_baseline}")



