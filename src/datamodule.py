# %%writefile src/datamodule.py
import os

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class ImagesDataset(Dataset):
    """Reads in an image, transforms pixel values, and serves
    a dictionary containing the image id, image tensors, and label.
    """

    def __init__(self, data_dir, x_df, y_df=None):
        self.data_dir = data_dir
        self.data = x_df
        self.label = y_df
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def __getitem__(self, index):
        image = Image.open(self.data_dir + "/" + self.data.iloc[index]["filepath"]
                           ).convert("RGB")
        image = self.transform(image)
        image_id = self.data.index[index]
        # if we don't have labels (e.g. for test set) just return the image and image id
        if self.label is None:
            sample = {"image_id": image_id, "image": image}
        else:
            label = torch.tensor(
                self.label.iloc[index].values, dtype=torch.float)
            sample = {"image_id": image_id, "image": image, "label": label}
        return sample

    def __len__(self):
        return len(self.data)


class WLCDataModule:
    def __init__(self, data_dir: str, batch_size: int = 64, num_workers=0, pin_memory=False) -> None:
        """

        :rtype: object
        """
        self.species_labels = None
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_features = os.path.join(data_dir, "train_features.csv")
        self.test_features = os.path.join(data_dir, "test_features.csv")
        self.train_labels = os.path.join(data_dir, "train_labels.csv")

    def setup(self, stage=None):
        train_features = pd.read_csv(self.train_features, index_col="id")
        test_features = pd.read_csv(self.test_features, index_col="id")
        train_labels = pd.read_csv(self.train_labels, index_col="id")
        self.species_labels = sorted(train_labels.columns.unique())

        y = train_labels
        x = train_features.loc[y.index].filepath.to_frame()

        # we set up only relevant datasets when stage is specified
        if stage == "fit" or stage is None:
            self.train_dataset = ImagesDataset(self.data_dir, x, y)
            val_size = int(0.25 * len(self.train_dataset))
            train_size = len(self.train_dataset) - val_size
            self.train_dataset, self.val_dataset = random_split(
                self.train_dataset, [train_size, val_size]
            )
        if stage == "test" or stage is None:
            self.test_dataset = ImagesDataset(
                self.data_dir, test_features.filepath.to_frame()
            )

    # we define a separate DataLoader for each of train/val/test
    def train_dataloader(self) -> object:
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        return train_dataloader

    def val_dataloader(self) -> object:
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * 2,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        return val_dataloader

    def test_dataloader(self) -> object:
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size * 2,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        return test_dataloader

    def __repr__(self) -> object:
        rep = "train_len: {}\nval_len  : {}\ntest_len : {}\n".format(
            len(self.train_dataset), len(self.val_dataset), len(self.test_dataset)
        )
        return rep


def data_loader(data_dir, batch_size, num_workers=0, pin_memory=False):
    _train_features = os.path.join(data_dir, "train_features.csv")
    _test_features = os.path.join(data_dir, "test_features.csv")
    _train_labels = os.path.join(data_dir, "train_labels.csv")

    train_features = pd.read_csv(_train_features, index_col="id")
    test_features = pd.read_csv(_test_features, index_col="id")
    train_labels = pd.read_csv(_train_labels, index_col="id")
    species_labels = sorted(train_labels.columns.unique())

    y = train_labels.sample(frac=1, random_state=1)
    x = train_features.loc[y.index].filepath.to_frame()

    # note that we are casting the species labels to an indicator/dummy matrix
    x_train, x_eval, y_train, y_eval = train_test_split(
        x, y, stratify=y, test_size=0.25)

    train_dataset = ImagesDataset(data_dir, x_train, y_train)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory
                                  )

    eval_dataset = ImagesDataset(data_dir, x_eval, y_eval)
    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=batch_size * 2,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory
                                 )

    test_dataset = ImagesDataset(data_dir, test_features.filepath.to_frame())
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size * 2,
                                 # num_workers=num_workers,
                                 # pin_memory=pin_memory
                                 )
    print("Train data:", len(train_dataset))
    print("  Val data:", len(eval_dataset))
    print(" Test data:", len(test_dataset))
    return train_dataloader, eval_dataloader, test_dataloader, species_labels
