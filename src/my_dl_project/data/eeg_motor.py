from pathlib import Path

import mne
import torch
from torch.utils.data import Dataset


class EEGMotorDataset(Dataset):
    """
    Simple PyTorch Dataset for the PhysioNet EEG Motor Movement/Imagery dataset.

    For the milestone, each EDF file is loaded as one example.
    Later, this can be improved by extracting event-labeled time windows.
    """

    def __init__(self, data_dir, max_files=None):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("*.edf"))

        if max_files is not None:
            self.files = self.files[:max_files]

        if len(self.files) == 0:
            raise FileNotFoundError(f"No EDF files found in {self.data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]

        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        x = raw.get_data()

        # Convert EEG signal to torch tensor.
        # Shape: [channels, time_points]
        x = torch.tensor(x, dtype=torch.float32)

        # Temporary target for milestone demo.
        # Final project will use EDF annotations/events as labels.
        y = torch.tensor(0, dtype=torch.long)

        return x, y