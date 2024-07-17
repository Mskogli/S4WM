import numpy as np

from jax.tree_util import tree_map
from torch.utils import data
from torch.utils.data import DataLoader
from typing import Tuple

from s4wm.data.depth_img_dataset import DepthImageDataset, split_dataset
from s4wm.utils.dlpack import from_torch_to_jax


def numpy_collate(batch):
    return tree_map(np.asarray, data.default_collate(batch))


class NumpyLoader(data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ) -> None:
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


def create_depth_dataset(
    file_path: str,
    batch_size: int = 128,
    val_fraction: int = 0.1,
    device: str = "cuda:0",
    num_trajs: int = 16500,
    traj_length: int = 100,
) -> Tuple[DataLoader, ...]:  # 2 tuple
    dataset = DepthImageDataset(
        file_path, device, num_trajs=num_trajs, traj_length=traj_length
    )

    train_dataset, val_dataset = split_dataset(dataset, val_fraction)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader


Dataloaders = {
    "depth_dataset": create_depth_dataset,
}

if __name__ == "__main__":
    train_loader, val_loader = create_depth_dataset(
        file_path="/home/mathias/dev/datasets/quad_depth_imgs",
        batch_size=8,
        val_fraction=0.1,
    )

    val_depth_images, val_actions, val_labels = next(iter(val_loader))
    val_depth_images, val_actions, val_labels = map(
        from_torch_to_jax, (val_depth_images, val_actions, val_labels)
    )

    print(val_depth_images.shape, val_actions.shape, val_labels.shape)
