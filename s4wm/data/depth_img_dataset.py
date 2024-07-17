import torch
import h5py

from torch.utils.data import Dataset, random_split
from typing import List, Tuple


class DepthImageDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        device: str,
        num_trajs: int = 16500,
        traj_length: int = 100,
    ) -> None:
        self.file = h5py.File(file_path, "r")
        self.device = device
        self.num_trajs = num_trajs
        self.traj_length = traj_length

        self.max_depth_value = 10
        self.min_depth_value = 0.0

    def __len__(self) -> int:
        return self.num_trajs

    def __getitem__(self, idx: int) -> torch.tensor:
        depth_images = []
        actions = []

        for i in range(self.traj_length):
            dataset = self.file[f"trajectory_{idx}/image_{i}"]
            img_data = dataset[:]

            depth_images.append(
                torch.from_numpy(img_data)
                .view(1, 135, 240, 1)
                .to(torch.device(self.device))
            )
            actions.append(
                torch.from_numpy(dataset.attrs["actions"])
                .view(1, 4)
                .to(torch.device(self.device))
            )

        imgs = torch.cat(depth_images, dim=0)

        # Normalize depth images
        imgs[torch.isinf(imgs)] = self.max_depth_value
        imgs[imgs > self.max_depth_value] = self.max_depth_value
        imgs[imgs < self.min_depth_value] = self.min_depth_value
        imgs = (imgs - self.min_depth_value) / (
            self.max_depth_value - self.min_depth_value
        )

        actions = torch.cat(actions, dim=0)[1:]
        labels = imgs[1:, :].view(-1, 135 * 240)

        return (imgs, actions, labels)


def split_dataset(
    dataset: DepthImageDataset, val_split: float
) -> Tuple[DepthImageDataset, ...]:  # 2 tuple
    total_samples = dataset.num_trajs
    train_len = int(total_samples * (1 - val_split))
    val_len = total_samples - train_len

    return random_split(dataset, [train_len, val_len])
