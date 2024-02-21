import numpy as np
from jax.tree_util import tree_map
from torch.utils import data
from .ag_traj_dataset import AerialGymTrajDataset, split_dataset


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
    ):
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


def create_quad_depth_trajectories_datasets(bsz=128):
    print("[*] Generating Aerial Gym Trajectory Dataset")

    N_CLASSES, SEQ_LENGTH, IN_DIM = 128, 44, 132

    dataset = AerialGymTrajDataset(
        "/home/mathias/dev/quad_depth_imgs",
        "cpu",
        actions=True,
    )

    train_dataset, val_dataset = split_dataset(dataset, 0.1)
    trainloader = NumpyLoader(train_dataset, batch_size=bsz, shuffle=True)
    valloader = NumpyLoader(val_dataset, batch_size=bsz, shuffle=True)

    return trainloader, valloader, N_CLASSES, SEQ_LENGTH, IN_DIM


Datasets = {
    "quad_depth_trajectories": create_quad_depth_trajectories_datasets,
}
