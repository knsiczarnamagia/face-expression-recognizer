import os
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import imagehash
from typing import Callable
from datetime import datetime as dt

_DATASET_AVG_MEAN = 129.38489987766278
_DATASET_AVG_STD = 54.084109207654805
_OPTIM_MEAN_THRESH = 107
_OPTIM_STD_THRESH = 51
_OPTIM_NUM_COMPONENTS = 3
_OPTIM_ERROR_THRESH = 87


# Mean pixel intensity and mean standard deviation threshold strategy

def save_to_file(location: str = './outliers.txt') -> Callable:
    def decorator(fn: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            paths: list[str] = fn(*args, **kwargs)
            with open(location, 'a') as file:
                file.write('\nFiles to remove [TIMESTAMP {}]:\n'.format(dt.now().strftime('%Y%m%d%H%M%S')))
                for p in paths:
                    file.write(f'{p}\n')
            return paths

        return wrapper

    return decorator


def remove(fn: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        paths: list[str] = fn(*args, **kwargs)
        for path in paths:
            print(f'Removing {path}')
            os.remove(path)
        return paths

    return wrapper


def visualize(show_limit: int = -1) -> Callable:
    def decorator(fn: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            paths: list[str] = fn(*args, **kwargs)
            if show_limit != -1:
                paths = paths[:show_limit]

            num_cols = 8
            num_rows = len(paths) // num_cols + 1

            fig = plt.figure(figsize=(8, 8))
            for i, path in enumerate(paths, start=1):
                plt.subplot(num_rows, num_cols, i)
                plt.imshow(Image.open(path), cmap='gray')
                plt.title(f'{Path(path).parent.name}', fontsize=7)
                plt.axis('off')
            fig.tight_layout()
            plt.tight_layout()
            fig.subplots_adjust(hspace=0.6, top=0.97)
            plt.show()
            return paths

        return wrapper

    return decorator


def compute_dataset_stats(data_root: str) -> dict[str, float]:
    img_paths = list(Path(data_root).glob('**/*.jpg'))
    num_images = len(img_paths)
    mean_sum = 0
    std_sum = 0

    for img_path in img_paths:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img_mean = np.mean(img)
        img_std = np.std(img)
        mean_sum += img_mean
        std_sum += img_std

    avg_mean = mean_sum / num_images
    avg_std = std_sum / num_images
    stats_dict = {
        'avg_mean': avg_mean,
        'avg_std': avg_std,
    }
    return stats_dict


# WARNING: uncommenting the line below will remove dataset files
# @remove
@visualize()
@save_to_file()
def extract_outliers_stat(data_root: str | Path,
                          dataset_avg_mean: float,
                          dataset_avg_std: float,
                          mean_thresh: float,
                          std_thresh: float,
                          console_progress: bool = False) -> list[str]:
    outlier_paths = []
    count = 0
    _, _, paths = _load_data(data_root)
    total_len = len(paths)
    for path in iter(paths):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if abs(dataset_avg_mean - np.mean(img)) > mean_thresh or abs(dataset_avg_std - np.std(img)) > std_thresh:
            outlier_paths.append(path)
        if console_progress:
            count += 1
            print(f'Computed {count}/{total_len} images ({count / total_len * 100:.2f}%)')
    return outlier_paths


# PCA strategy

def _load_data(dir_: str) -> tuple[list[np.ndarray], list[str], list[str]]:
    images = []
    class_names = []
    paths = []

    for path in Path(dir_).glob('**/*.jpg'):
        label = path.parent.name
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if not (image is None or label is None):
            images.append(np.array(image))
            class_names.append(label)
            paths.append(str(path))

    return images, class_names, paths


# WARNING: uncommenting the line below will remove dataset files
# @remove
@visualize()
@save_to_file()
def extract_outliers_pca(dir_: str | Path) -> list[str]:
    x, y_labels, img_paths = _load_data(dir_)
    x, y_labels = np.array(x), np.array(y_labels)
    num_samples, height, width = x.shape
    X_flattened = x.reshape(num_samples, height * width)

    outlier_indices = _detect_outliers_with_pca(X_flattened, _OPTIM_NUM_COMPONENTS, _OPTIM_ERROR_THRESH)
    img_paths_to_remove = [img_paths[i] for i in outlier_indices.tolist()]
    return img_paths_to_remove


def _detect_outliers_with_pca(orig_data: np.ndarray,
                              num_components: int,
                              error_thresh: float) -> np.ndarray:
    pca = PCA(n_components=num_components)
    X_reduced = pca.fit_transform(orig_data)

    X_reconstructed = pca.inverse_transform(X_reduced)
    reconstruction_errors = np.sqrt(np.mean((orig_data - X_reconstructed) ** 2, axis=1))

    outlier_indices = np.where(reconstruction_errors > error_thresh)[0]
    return outlier_indices


# WARNING: uncommenting the line below will remove dataset files
# @remove
@visualize(60)
@save_to_file()
def extract_duplicates_hash(dir_: str | Path, hash_size: int = 8) -> list[str]:
    _, _, paths = _load_data(dir_)
    hashes = set()
    duplicates = []

    for path in paths:
        hash_ = imagehash.dhash(Image.open(path), hash_size)
        if hash_ in hashes:
            duplicates.append(path)
        else:
            hashes.add(hash_)
    return duplicates


if __name__ == '__main__':
    dataset_dir = Path('./dataset')
    extract_outliers_stat(dataset_dir, _DATASET_AVG_MEAN, _DATASET_AVG_STD,
                          _OPTIM_MEAN_THRESH, _OPTIM_STD_THRESH, console_progress=True)
    extract_outliers_pca(dataset_dir / 'train')
    extract_outliers_pca(dataset_dir / 'test')
    extract_duplicates_hash(dataset_dir)
