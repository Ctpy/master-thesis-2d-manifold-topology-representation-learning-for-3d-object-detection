from torch.utils.data import Dataset
import torch
from pathlib import Path
from typing import List, Tuple, Union, Optional
import numpy as np
import json
from tqdm import tqdm
import threading
import enum


class KITTIObjectLabel(enum.Enum):
    """
    KITTI Object Label.
    """

    CAR = "Car"
    VAN = "Van"
    TRUCK = "Truck"
    PEDESTRIAN = "Pedestrian"
    PERSON_SITTING = "Person_sitting"
    CYCLIST = "Cyclist"
    TRAM = "Tram"
    MISC = "Misc"
    DONT_CARE = "DontCare"

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def from_str(label: str) -> int:
        """
        Convert label from string to int.
        """
        if label == "Car":
            return 0
        elif label == "Van":
            return 1
        elif label == "Truck":
            return 2
        elif label == "Pedestrian":
            return 3
        elif label == "Person_sitting":
            return 4
        elif label == "Cyclist":
            return 5
        elif label == "Tram":
            return 6
        else:
            raise ValueError(f"Unknown label: {label}")


class KITTICroppedDataset(Dataset):
    """
    KITTI Cropped Dataset.
    """

    def __init__(self, data_path: str, min_points: Optional[int] = None) -> None:
        super().__init__()
        self.data_path: Path = Path(data_path)
        self.point_cloud_path: Path = self.data_path / "point_clouds"
        self.label_path: Path = self.data_path / "labels"
        self.point_cloud_files: List[Path] = list(self.point_cloud_path.glob("*.npy"))
        self.label_files: List[Path] = list(self.label_path.glob("*.json"))
        self.point_cloud_files.sort()
        self.label_files.sort()
        self.classes = ["Car"]  # , 'Pedestrian', 'Cyclist']
        self.threads: List[threading.Thread] = []
        self.lock: threading.Lock = threading.Lock()
        assert len(self.point_cloud_files) == len(
            self.label_files
        ), f"Number of point clouds ({len(self.point_cloud_files)}) and labels ({len(self.label_files)}) do not match."
        if min_points is not None:
            self.filter_by_min_points_multithreading(min_points)

    def __len__(self) -> int:
        return len(self.point_cloud_files)

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[Tuple[torch.Tensor, int], List[Tuple[torch.Tensor, int]]]:
        if isinstance(index, slice):
            point_cloud_files: List[Path] = self.point_cloud_files[index]
            label_files: List[Path] = self.label_files[index]
            point_clouds: List[np.ndarray] = [
                np.load(point_cloud_file)[:, :3]
                for point_cloud_file in point_cloud_files
            ]
            labels: List[str] = [
                json.load(label_file.resolve().open("r"))["label"]
                for label_file in label_files
            ]
            return [
                (torch.from_numpy(point_cloud), KITTIObjectLabel.from_str(label))
                for point_cloud, label in zip(point_clouds, labels)
            ]
        elif isinstance(index, int):
            point_cloud_file: Path = self.point_cloud_files[index]
            label_file: Path = self.label_files[index]
            point_cloud: np.ndarray = np.load(point_cloud_file)[:, :3]
            label: str = json.load(label_file.resolve().open("r"))["label"]
            return torch.from_numpy(point_cloud), KITTIObjectLabel.from_str(label)
        else:
            raise TypeError(f"Invalid argument type: {type(index)}")

    def get_min_num_points(self) -> int:
        num_labels = 0
        for label in tqdm(self.label_files):
            num_points: int = json.load(label.resolve().open("r"))["num_points"]
            if num_points >= 1024:
                num_labels += 1
        return num_labels

    def filter_by_min_points(
        self, thread_id: int, point_cloud_files, label_files, split_indices: List[int]
    ) -> None:
        tmp_point_cloud_files: List[Path] = []
        tmp_label_files: List[Path] = []
        for i in tqdm(split_indices, desc=f"Thread {thread_id}"):
            data = json.load(self.label_files[i].resolve().open("r"))
            num_points: int = data["num_points"]
            label = data["label"]
            if num_points >= 1024 and label in self.classes:
                point_cloud_files.append(self.point_cloud_files[i])
                label_files.append(self.label_files[i])

        if self.lock.acquire():
            point_cloud_files.extend(tmp_point_cloud_files)
            label_files.extend(tmp_label_files)
            self.lock.release()

    def filter_by_min_points_multithreading(
        self, min_points: int, num_threads: int = 8
    ) -> None:
        split_indices = np.array_split(np.arange(len(self.label_files)), num_threads)
        print(split_indices)
        point_cloud_files: List[Path] = []
        label_files: List[Path] = []
        for idx, split_index in enumerate(split_indices):
            thread = threading.Thread(
                target=self.filter_by_min_points,
                args=(
                    idx,
                    point_cloud_files,
                    label_files,
                    split_index,
                ),
            )
            self.threads.append(thread)
            thread.start()
        # wait for  threads to finish
        for thread in self.threads:
            thread.join()

        self.point_cloud_files = point_cloud_files
        self.label_files = label_files


if __name__ == "__main__":
    kitti = KITTICroppedDataset(data_path="data/kitti_cropped")
    print(kitti.get_min_num_points())
