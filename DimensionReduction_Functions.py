from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# UMAP is optional — import lazily so the rest of the script works without it.




# ---------------------------------------------------------------------------
# Reduction
# ---------------------------------------------------------------------------

def _lda_with_residual_axes(X: np.ndarray, Y: np.ndarray,
                            n_components: int) -> np.ndarray:
    """
    Two-class LDA gives only one discriminative axis. To still produce
    2D/3D plots, we use that LDA axis as the first dimension and fill the
    remaining dimensions with PCA on the data projected orthogonally to it.

    """
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda_axis = lda.fit_transform(X, Y)              # (N, 1)

    if n_components == 1:
        return lda_axis

    # Residual: data with LDA direction projected out, then PCA on the rest.
    direction = lda.scalings_[:, 0]
    direction = direction / np.linalg.norm(direction)
    residual = X - np.outer(X @ direction, direction)
    pca_extra = PCA(n_components=n_components - 1).fit_transform(residual)

    return np.concatenate([lda_axis, pca_extra], axis=1)


def reduce_dimensions(
    X: np.ndarray,
    Y: np.ndarray,
    method: str,
    n_components: int = 2,
    standardize: bool = True,
    seed: int = 42,
    **kwargs,
) -> np.ndarray:
    """
    Project (N, L) features down to (N, n_components) using `method`.

    Parameters
    ----------
    X, Y : input features and labels.
    method : one of 'pca', 'lda', 'tsne', 'umap'.
    n_components : 2 or 3.
    standardize : z-score columns first. Recommended for everything except
        when X is already normalized; PCA in particular is sensitive to scale.
    seed : random seed for stochastic methods (t-SNE, UMAP).
    **kwargs : forwarded to the underlying estimator. Useful overrides:
        - PCA   : whiten=True
        - t-SNE : perplexity=30, learning_rate='auto'
        - UMAP  : n_neighbors=15, min_dist=0.1, metric='euclidean'

    Returns
    -------
    Z : np.ndarray of shape (N, n_components)
    """
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3")

    method = method.lower()
    Xs = StandardScaler().fit_transform(X) if standardize else np.asarray(X)

    if method == 'pca':
        return PCA(n_components=n_components,
                   random_state=seed, **kwargs).fit_transform(Xs)

    if method == 'lda':
        return _lda_with_residual_axes(Xs, Y, n_components)

    if method == 'tsne':
        # learning_rate='auto' and init='pca' are the modern recommended defaults.
        defaults = dict(perplexity=50, learning_rate='auto',
                        init='pca', random_state=seed)
        defaults.update(kwargs)
        return TSNE(n_components=n_components, **defaults).fit_transform(Xs)

    if method == 'umap':
        try:
            import umap
        except ImportError as e:
            raise ImportError(
                "UMAP not installed. Install with: pip install umap-learn"
            ) from e
        defaults = dict(n_neighbors=30, min_dist=0.1, random_state=seed)
        defaults.update(kwargs)
        return umap.UMAP(n_components=n_components,
                         **defaults).fit_transform(Xs)

    raise ValueError(f"Unknown method {method!r}. "
                     f"Choose from: pca, lda, tsne, umap.")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_embedding(
    Z: np.ndarray,
    Y: np.ndarray,
    method_name: str = '',
    class_names: Sequence[str] | None = None,
    ax=None,
    alpha: float = 0.7,
    s: float = 18,
    figsize: tuple[float, float] = (7, 6),
):
    """
    Scatter plot of a (N, 2) or (N, 3) embedding, colored by class.

    For 3D, a new figure is created with a 3D axis if `ax` is None.
    """
    Z = np.asarray(Z)
    Y = np.asarray(Y)
    n_components = Z.shape[1]
    if n_components not in (2, 3):
        raise ValueError("Z must have 2 or 3 columns")

    classes = np.unique(Y)
    if class_names is None:
        class_names = [f'class {c}' for c in classes]
    if len(class_names) != len(classes):
        raise ValueError(f"class_names ({len(class_names)}) doesn't match "
                         f"number of unique labels ({len(classes)})")

    # Colorblind-friendly palette
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    if ax is None:
        fig = plt.figure(figsize=figsize)
        if n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

    for i, c in enumerate(classes):
        mask = (Y == c)
        if n_components == 2:
            ax.scatter(Z[mask, 0], Z[mask, 1],
                       c=palette[i % len(palette)], label=class_names[i],
                       alpha=alpha, s=s, edgecolors='none')
        else:
            ax.scatter(Z[mask, 0], Z[mask, 1], Z[mask, 2],
                       c=palette[i % len(palette)], label=class_names[i],
                       alpha=alpha, s=s, edgecolors='none')

    title = method_name.upper() if method_name else 'Embedding'
    ax.set_title(f'{title} ({n_components}D)')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    if n_components == 3:
        ax.set_zlabel('Component 3')
    ax.legend(loc='best', frameon=True)
    if n_components == 2:
        ax.grid(alpha=0.3)
    return ax


def reduce_and_plot(
    X: np.ndarray,
    Y: np.ndarray,
    method: str,
    n_components: int = 2,
    class_names: Sequence[str] | None = None,
    standardize: bool = True,
    seed: int = 42,
    show: bool = True,
    **kwargs,
) -> np.ndarray:
    """
    Run reduction and visualize in one call. Returns the embedding.
    """
    Z = reduce_dimensions(X, Y, method, n_components=n_components,
                          standardize=standardize, seed=seed, **kwargs)
    plot_embedding(Z, Y, method_name=method, class_names=class_names)
    if show:
        plt.tight_layout()
        plt.show()
    return Z


def compare_methods(
    X: np.ndarray,
    Y: np.ndarray,
    methods: Sequence[str] = ('pca', 'lda', 'tsne', 'umap'),
    n_components: int = 2,
    class_names: Sequence[str] | None = None,
    standardize: bool = True,
    seed: int = 42,
    show: bool = True,
    **kwargs,
) -> dict[str, np.ndarray]:
    """
    Run multiple reduction methods and plot them side-by-side.

    Useful workshop view: lets students see which method best separates
    the classes for their feature set. Returns a dict {method: embedding}.
    """
    methods = list(methods)
    n = len(methods)
    cols = 2
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 6, rows * 5))
    embeddings = {}

    for i, m in enumerate(methods):
        try:
            Z = reduce_dimensions(X, Y, m, n_components=n_components,
                                  standardize=standardize, seed=seed,
                                  **kwargs.get(m, {}))
            embeddings[m] = Z

            if n_components == 3:
                ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
            else:
                ax = fig.add_subplot(rows, cols, i + 1)
            plot_embedding(Z, Y, method_name=m,
                           class_names=class_names, ax=ax)
        except Exception as e:
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.text(0.5, 0.5, f"{m.upper()} failed:\n{e}",
                    ha='center', va='center', wrap=True)
            ax.set_axis_off()

    if show:
        plt.tight_layout()
        plt.show()
    return embeddings














