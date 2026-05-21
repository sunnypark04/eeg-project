from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

DEFUNGI_CLASSES = ["H1", "H2", "H3", "H5", "H6"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class DeFungiDataset(Dataset):
    """Custom PyTorch Dataset for DeFungi.

    The dataset structure is based on images organized in folders by label:

        root_dir/
        ├── H1/
        ├── H2/
        ├── H3/
        ├── H5/
        └── H6/
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        for label_index, class_name in enumerate(DEFUNGI_CLASSES):
            class_dir = self.root_dir / class_name

            if not class_dir.exists():
                continue

            for image_path in class_dir.iterdir():
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((image_path, label_index))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label
