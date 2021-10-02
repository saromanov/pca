import numpy as np


class PCA:
    """:class:`PCA` is a class for dimensionality reduction
    """

    def __init__(self, n_components: int, whiten: bool = False):
        self.n_components = n_components
        self.whiten = whiten
        self.e_values = None
        self.w = None

    def fit(self, x_mat: np.ndarray):
        x_mat = x_mat - x_mat.mean(axis=0)
        cov = np.cov(x_mat.T) / x_mat.shape[0]
        e_values, e_vectors = np.linalg.eig(cov)
        idx = e_values.argsort()[::-1]
        e_values = e_values[idx]
        e_vectors = e_vectors[:, idx]
        self.w = e_vectors
        self.e_values = e_values

    def transform(self, x_mat: np.ndarray) -> np.ndarray:
        if not self.w:
            return
        x_mat_projected = x_mat.dot(self.w[:, : self.n_components])
        if self.whiten:
            return x_mat_projected / np.sqrt(self.e_values[0 : self.n_components])
        else:
            return x_mat_projected

    def fit_transform(self, x_mat: np.ndarray) -> np.ndarray:
        self.fit(x_mat)
        return self.transform(x_mat)