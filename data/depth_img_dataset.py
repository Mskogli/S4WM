import torch
import h5py
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, random_split, DataLoader
from typing import List, Tuple


class DepthImageDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        device: str,
        actions: bool = False,
    ) -> None:
        self.file = h5py.File(file_path, "r")
        self.device = device
        self.actions = actions
        self.num_trajs = 980

    def __len__(self) -> int:
        return self.num_trajs

    def __getitem__(self, idx) -> torch.tensor:
        depth_images = []
        actions = []

        for i in range(100):
            dataset = self.file[f"trajectory_{idx}/image_{i}"]
            img_data = dataset[:]
            depth_images.append(torch.from_numpy(img_data).view(1, 270, 480))
            actions.append(torch.from_numpy(dataset.attrs["actions"]).view(1, 4))

        imgs = torch.cat(depth_images, dim=0)
        acts = torch.cat(actions, dim=0)

        return imgs, acts


def split_dataset(
    dataset: DepthImageDataset, val_split: float
) -> Tuple[DepthImageDataset, ...]:  # 2 tuple
    total_samples = dataset.num_trajs
    train_len = int(total_samples * (1 - val_split))
    val_len = total_samples - train_len

    return random_split(dataset, [train_len, val_len])


if __name__ == "__main__":

    dataset = DepthImageDataset(
        "/home/mathias/dev/quad_depth_imgs",
        "cpu",
        actions=True,
    )

    train_dataset, val_dataset = split_dataset(dataset, 0.1)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    train_batch, _ = next(iter(train_loader))

    for idx, image in enumerate(train_batch[0], 1):
        print(idx)
        plt.imshow(image.view(270, 480))
        plt.show()
