from torch.utils.data import Dataset
from torchvision import transforms
import os
import torch


class FashionPediaProcessed(Dataset):
    def __init__(self, dataset_path="fashionpedia/processed_dataset.pt") -> None:
        super().__init__()

        if not os.path.isfile(dataset_path):
            raise FileNotFoundError()

        with open(dataset_path, "rb") as file:
            saved_data = torch.load(file)

            self.data = saved_data["dataset"]
            self.means = saved_data["means"]
            self.stds = saved_data["stds"]

        self.inv_normal = transforms.Compose([
            transforms.Normalize(
                mean = torch.zeros_like(self.means),
                std = 1 / self.stds,
            ),
            transforms.Normalize(
                mean = -self.means,
                std = torch.ones_like(self.stds),
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
    def invImg(self, item):
        return self.inv_normal(item['img']).permute(1, 2, 0)
