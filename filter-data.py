import os
from pathlib import Path
import cv2
import numpy as np
from sklearn.decomposition import PCA
from typing import Callable
from datetime import datetime as dt


_DATASET_AVG_MEAN = 129.38489987766278
_DATASET_AVG_STD = 54.084109207654805
_OPTIM_MEAN_THRESH = 105
_OPTIM_STD_THRESH = 51
_OPTIM_NUM_COMPONENTS = 3
_OPTIM_ERROR_THRESH = 85


# Mean pixel intensity and mean standard deviation threshold strategy

def save_to_file(location: str = './outliers.txt') -> Callable:
    def decorator(fn: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            paths = fn(*args, **kwargs)
            with open(location, 'a') as file:
                file.write('TIMESTAMP {}\n'.format(dt.now().strftime('%Y%m%d%H%M%S')))
                for p in paths:
                    file.write(f'{p}\n')
            return paths
        return wrapper
    return decorator


def remove(fn: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        file_paths = fn(*args, **kwargs)
        for path in file_paths:
            print(f'Removing {path}')
            os.remove(path)
        return file_paths
    return wrapper


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


@remove
@save_to_file()
def extract_outliers_stat(data_root: str,
                          dataset_avg_mean: float,
                          dataset_avg_std: float,
                          mean_thresh: float = None,
                          std_thresh: float = None) -> list[str]:
    if ((mean_thresh is not None and std_thresh is not None) or
            (mean_thresh is None and std_thresh is None)):
        raise ValueError('Either "mean_thresh" or "std_thresh" parameter has to be specified.')

    outlier_paths = []
    for path in Path(data_root).glob('**/*.jpg'):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mean_thresh is not None and abs(dataset_avg_mean - np.mean(img)) > mean_thresh:
            outlier_paths.append(str(path))
        elif std_thresh is not None and abs(dataset_avg_std - np.std(img)) > std_thresh:
            outlier_paths.append(str(path))
    return outlier_paths


# PCA strategy

def load_data(dir_: str) -> tuple[list[np.ndarray], list[str], list[str]]:
    images = []
    class_names = []
    paths = []

    x_paths = list(Path(dir_).glob('**/*.jpg'))
    for path in x_paths:
        label = path.parent.name
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if not (image is None or label is None):
            images.append(np.array(image))
            class_names.append(label)
            paths.append(str(path))

    return images, class_names, paths


@remove
@save_to_file()
def extract_outliers_pca(dir_: str) -> list[str]:
    x, y_labels, img_paths = load_data(dir_)
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


if __name__ == '__main__':
    extract_outliers_stat('./dataset', _DATASET_AVG_MEAN, _DATASET_AVG_STD, mean_thresh=_OPTIM_MEAN_THRESH)
    extract_outliers_stat('./dataset', _DATASET_AVG_MEAN, _DATASET_AVG_STD, std_thresh=_OPTIM_STD_THRESH)
    extract_outliers_pca('./dataset/train')
    extract_outliers_pca('./dataset/test')
