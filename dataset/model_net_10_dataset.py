from torch.utils.data import Dataset
import torch

from pathlib import Path
from typing import List, Tuple, Union, Optional
import numpy as np
import json
import os


class ModelNet10Dataset(Dataset):

    def __init__(self, data_path: str) -> None:
        super().__init__()
        self.data_path: Path = Path(data_path)
        self.files: List[Path] = list(self.data_path.glob("**/*.off"))
        folders = [folder.name for folder in os.scandir(self.data_path) if folder.is_dir()]
        self.classes = {folder: i for i, folder in enumerate(folders)};

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        point_cloud_file: Path = self.files[index]
        with open(point_cloud_file, 'r') as f:
            if 'OFF' not in f.readline().strip().split(' '):
                raise ValueError('Not a valid OFF header')
            n_verts, n_faces, __ = map(int, f.readline().strip().split(' '))
            verts = [list(map(float, f.readline().strip().split(' '))) for i in range(n_verts)]
            faces = [list(map(int, f.readline().strip().split(' ')[1:])) for i in range(n_faces)]
        
        return np.array(verts), np.array(faces)
    

if __name__ == '__main__':
    dataset = ModelNet10Dataset('/workspace/data/ModelNet10')
    print(dataset.classes)
    print(dataset[0])
    