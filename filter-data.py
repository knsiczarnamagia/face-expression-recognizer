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
from abc import ABC, abstractmethod

_DATASET_AVG_MEAN = 129.38489987766278
_DATASET_AVG_STD = 54.084109207654805


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


class DataFilter(ABC):
    def __init__(self):
        self.paths = []

    @abstractmethod
    def extract(self, data_dir: str | Path) -> list[str]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def filter(self) -> bool:
        pass

    @staticmethod
    def _load_data(dir_: str) -> tuple[list[np.ndarray], list[str], list[str]]:
        images = []
        class_names = []
        paths = []

        for path in Path(dir_).glob('**/*.jpg'):
            label = path.parent.name
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is not None and label is not None:
                images.append(np.array(image))
                class_names.append(label)
                paths.append(str(path))

        return images, class_names, paths


class DataFilterCompose(DataFilter):
    def __init__(self, components: list[DataFilter]):
        super().__init__()
        self.components = components

    @staticmethod
    def build(components: list[DataFilter]) -> DataFilter:
        return DataFilterCompose(components)

    def extract(self, data_dir: str | Path) -> list[str]:
        extracted_paths = []
        for component in self.components:
            cur_extracted_paths = component.extract(data_dir)
            extracted_paths += cur_extracted_paths
        self.paths += extracted_paths
        return extracted_paths

    def clear(self) -> None:
        for component in self.components:
            component.clear()

    def filter(self):
        for component in self.components:
            component.filter()

    def add_component(self, component: DataFilter, position: int) -> None:
        self.components.insert(position, component)

    def rm_component(self, position: int) -> None:
        self.components.pop(position)


class StatsDataFilter(DataFilter):
    _OPTIM_MEAN_THRESH = 107
    _OPTIM_STD_THRESH = 51

    def __init__(self, data_avg_mean: float = None, data_avg_std: float = None, console_output: bool = False):
        super().__init__()
        self.data_avg_mean = data_avg_mean
        self.data_avg_std = data_avg_std
        self.console_output = console_output

    @visualize()
    # @save_to_file()
    def extract(self, data_dir) -> list[str]:
        if self.data_avg_mean is None or self.data_avg_std is None:
            stats = self._compute_dataset_stats(data_dir)
            self.data_avg_mean = stats['avg_mean']
            self.data_avg_std = stats['avg_std']

        extracted_paths = self._extract_outliers_by_stats(
            data_dir,
            self.data_avg_mean,
            self.data_avg_std,
            StatsDataFilter._OPTIM_MEAN_THRESH,
            StatsDataFilter._OPTIM_STD_THRESH,
            self.console_output)

        self.paths += extracted_paths
        return extracted_paths

    def clear(self) -> None:
        self.paths.clear()
        if self.console_output:
            print(f'[{self.__class__.__name__}]: Paths memory cleared.')

    def filter(self) -> bool:
        has_error = False
        for path in self.paths:
            if not Path(path).exists():
                has_error = True
                continue
            os.remove(path)
            if self.console_output:
                print(f'[{self.__class__.__name__}]: Removed {path}')
        return has_error

    @classmethod
    def _extract_outliers_by_stats(cls,
                                   data_root: str | Path,
                                   dataset_avg_mean: float,
                                   dataset_avg_std: float,
                                   mean_thresh: float,
                                   std_thresh: float,
                                   console_output: bool = False) -> list[str]:
        outlier_paths = []
        count = 0
        _, _, paths = StatsDataFilter._load_data(data_root)
        total_len = len(paths)
        for path in iter(paths):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if abs(dataset_avg_mean - np.mean(img)) > mean_thresh or abs(
                    dataset_avg_std - np.std(img)) > std_thresh:
                outlier_paths.append(path)
            if console_output:
                count += 1
                print(f'[{cls.__name__}]: Computed {count}/{total_len} images ({count / total_len * 100:.2f}%)')
        return outlier_paths

    @staticmethod
    def _compute_dataset_stats(data_dir: str) -> dict[str, float]:
        img_paths = list(Path(data_dir).glob('**/*.jpg'))
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


class PcaDataFilter(DataFilter):
    _OPTIM_NUM_COMPONENTS = 4
    _OPTIM_ERROR_THRESH = 87

    def __init__(self, console_output: bool = False):
        super().__init__()
        self.console_output = console_output

    @visualize()
    # @save_to_file()
    def extract(self, data_dir: str | Path) -> list[str]:
        extracted_paths = self._extract_outliers_with_pca(data_dir)
        self.paths += extracted_paths
        return extracted_paths

    def clear(self) -> None:
        self.paths.clear()
        if self.console_output:
            print(f'[{self.__class__.__name__}]: Paths memory cleared.')

    def filter(self) -> bool:
        has_error = False
        for path in self.paths:
            if not Path(path).exists():
                has_error = True
                continue
            os.remove(path)
            if self.console_output:
                print(f'[{self.__class__.__name__}]: Removed {path}')
        return has_error

    @staticmethod
    def _extract_outliers_with_pca(dir_: str | Path) -> list[str]:
        x, _, img_paths = PcaDataFilter._load_data(dir_)
        x = np.array(x)
        num_samples, height, width = x.shape
        X_flattened = x.reshape(num_samples, height * width)

        outlier_indices = PcaDataFilter._detect_outliers_with_pca(X_flattened,
                                                                  PcaDataFilter._OPTIM_NUM_COMPONENTS,
                                                                  PcaDataFilter._OPTIM_ERROR_THRESH)
        img_paths_to_remove = [img_paths[i] for i in outlier_indices.tolist()]
        return img_paths_to_remove

    @staticmethod
    def _detect_outliers_with_pca(orig_data: np.ndarray,
                                  num_components: int,
                                  error_thresh: float) -> np.ndarray:
        pca = PCA(n_components=num_components)
        X_reduced = pca.fit_transform(orig_data)

        X_reconstructed = pca.inverse_transform(X_reduced)
        reconstruction_errors = np.sqrt(np.mean((orig_data - X_reconstructed) ** 2, axis=1))

        outlier_indices = np.where(reconstruction_errors > error_thresh)[0]
        return outlier_indices


class DHashDuplicateFilter(DataFilter):
    def __init__(self, hash_size: int = 8, console_output: bool = False):
        super().__init__()
        self.hash_size = hash_size
        self.console_output = console_output

    @visualize(60)
    # @save_to_file()
    def extract(self, data_dir: str | Path) -> list[str]:
        _, _, paths = self._load_data(data_dir)
        hashes = set()
        duplicates = []

        for path in paths:
            hash_ = imagehash.dhash(Image.open(path), self.hash_size)
            if hash_ in hashes:
                duplicates.append(path)
                if self.console_output:
                    print(f'[{self.__class__.__name__}]: Duplicate found at {path}')
            else:
                hashes.add(hash_)

        self.paths += duplicates
        return duplicates

    def clear(self) -> None:
        self.paths.clear()
        if self.console_output:
            print(f'[{self.__class__.__name__}]: Paths memory cleared.')

    def filter(self) -> bool:
        has_error = False
        for path in self.paths:
            if not Path(path).exists():
                has_error = True
                continue
            os.remove(path)
            if self.console_output:
                print(f'[{self.__class__.__name__}]: Removed {path}')
        return has_error


if __name__ == '__main__':
    dataset_dir = Path('./dataset')

    stats_filter = StatsDataFilter(_DATASET_AVG_MEAN, _DATASET_AVG_STD, True)
    pca_filter = PcaDataFilter(console_output=True)
    duplicate_filter = DHashDuplicateFilter(console_output=True)

    compose = DataFilterCompose.build([
        stats_filter,
        pca_filter,
        duplicate_filter
    ])
    compose.extract(dataset_dir)

    # WARNING: uncommenting the line below will irreversibly remove dataset files
    # compose.filter()
