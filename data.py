import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class SalienceDataset(Dataset):
    """Salience Dataset

    Args:
        Dataset (torch.util.data.Dataset): Dataset
    """
    def __init__(self, data_dir, ratio=1):
        """Init

        Args:
            data_dir (str): Path to the data directory
            ratio (float, optional): Ratio of images to load. If 1, all data is used, if .5 only half of them. Defaults to 1.
        """
        if not isinstance(data_dir, Path):
            data_dir = Path(data_dir)
        assert data_dir.is_dir(), f"{data_dir} not found"
        
        self.ratio = ratio
        self.data_dir = data_dir
        self.data_list = list(data_dir.glob("*.npz"))
    
    def __len__(self):
        return int(len(self.data_list) * self.ratio)
    
    def __getitem__(self, index):
        data = np.load(self.data_list[index])
        return [
            torch.tensor(data["hcqt"], dtype=torch.float32),
            torch.tensor(data["target_salience"], dtype=torch.float32)
        ]
