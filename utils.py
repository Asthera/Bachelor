import pandas as pd
import torchvision.transforms as transforms
from typing import Optional, Tuple, Any
from torchvision.io import read_image
from torch.utils.data import Dataset
import os


class CustomDataset(Dataset):
    def __init__(self,
                 dataframe: pd.DataFrame,
                 img_dir: str,
                 transform: Optional[Any] = None,
                 target_transform: Optional[Any] = None,
                 resolution: Tuple[int, int] = (224, 224)) -> None:

        self.data = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img_path = os.path.join(self.img_dir, self.data.iloc[idx]['frame_cropped_path'])

        # read as tensor, torch.uint8
        image = read_image(img_path)

        label = self.data.iloc[idx]['label']

        init_transform = transforms.Compose([
            # MUST BE

            transforms.Resize(self.resolution)
            # transforms.Pad([(pad_left, pad_right), (pad_top, pad_bottom)])

        ])

        image = init_transform(image)
        image = image.float()  # convert to float, was uint8

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        # from 1 channel to 3 channel
        image = image.expand(3, -1, -1)

        return image, label
