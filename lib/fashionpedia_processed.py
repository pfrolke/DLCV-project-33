from torch.utils.data import Dataset
import os
import torch


class FashionPediaProcessed(Dataset):
    def __init__(self) -> None:
        super().__init__()

        if not os.path.isfile("fashionpedia/processed_dataset.pt"):
            raise FileNotFoundError()

        with open("fashionpedia/processed_dataset.pt", "rb") as file:
            self.data = torch.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
