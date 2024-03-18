import torchvision.transforms
from torch.utils.data import Dataset
from os.path import join
from torchvision.io import read_image
import json
from sklearn.model_selection import train_test_split
import os


class FramesDataset(Dataset):
    def __init__(self,
                 json_path: str,
                 img_dir: str,
                 init_transform: torchvision.transforms.Compose = None,
                 target_transform: torchvision.transforms.Compose = None,
                 output_transform: torchvision.transforms.Compose = None,
                 subset: str = "train" or "test",
                 experiment_class: str = "b_line" or "a_line",
                 val: float = None
                 ):

        json_data = self.read_json(json_path)
        self.frames_json = self.json_to_dict_of_frames(json_data=json_data, subset=subset)

        if val:

            train, val = train_test_split(self.frames_json, test_size=val, random_state=42)

            if subset == "train":
                self.frames_json = train
            elif subset == "val":
                self.frames_json = val

        # change labels, because we have different classes
        self.frames_json = self.update_labels(experiment_class)

        self.img_dir = img_dir
        self.init_transform = init_transform
        self.transform = target_transform
        self.output_transform = output_transform

    def read_json(self, json_path: str):
        with open(json_path, 'r') as file:
            data = json.load(file)

        return data

    def json_to_dict_of_frames(self, json_data, subset):
        return [frame for video in json_data for frame in video['frames_only_label'] if
                video['subset'] == (subset if subset != "val" else "train")]

    def update_labels(self, experiment_class: str):

        if experiment_class == "b_line":
            return [
                {**frame, 'label': 1 if frame['label'] == 2 else 0 if frame['label'] == 1 else frame['label']}
                for frame in self.frames_json
            ]

        elif experiment_class == "a_line":
            return [
                {**frame, 'label': 0 if frame['label'] == 2 else frame['label']}
                for frame in self.frames_json
            ]

    def __len__(self):
        return len(self.frames_json)

    def __getitem__(self, idx):
        frame = self.frames_json[idx]

        frame_path = join(self.img_dir, frame["frame_cropped_path"])

        # read image  as tensor(torch.uint8) in [0, 255]
        image = read_image(frame_path)

        # convert to float in [0, 1]
        image = image.float()

        # apply transform
        if self.init_transform:
            image = self.init_transform(image)

        if self.transform:
            image = self.transform(image)

        if self.output_transform:
            image = self.output_transform(image)

        # Expand the image to 3 channels, because the model expects 3 channels
        image = image.expand(3, -1, -1)

        return image, frame["label"]
