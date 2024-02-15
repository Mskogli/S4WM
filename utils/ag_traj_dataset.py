import torch
import ujson
import h5py

from torch.utils.data import Dataset, random_split

# Literal Type hint introduced in python 3.8
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from typing import List, Tuple

States = Literal[
    "full state",
    "position",
    "attitude",
    "linear velocities",
    "angular velocities",
    "actions",
]

# The state is logged as 13 element vector in the dataset
StateIndices = {
    "full state": [0, 16],
    "goal_position": [0, 4],
    "position": [3, 6],
    "attitude": [6, 10],
    "linear velocities": [10, 13],
    "angular velocities": [13, 16],
}


class AerialGymTrajDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        device: str,
        actions: bool = False,
        states: List[States] = [],
    ) -> None:
        self.file = h5py.File(json_path, "r")
        self.device = device
        self.actions = actions
        self.states = states
        self.num_trajs = 989

    def __len__(self) -> int:
        return self.num_trajs

    def __getitem__(self, idx) -> torch.tensor:
        depth_images = []
        actions = []

        traj_grp = self.file[f"trajectory_{idx}"]
        for idx, (_, img_data) in enumerate(traj_grp.items()):
            if idx < 150:
                depth_images.append(img_data[:])
                actions.append(img_data.attrs["actions"])

        imgs = torch.tensor(depth_images, device=self.device)
        acts = torch.tensor(actions, device=self.device)
        return imgs, acts


def split_dataset(
    dataset: AerialGymTrajDataset, val_split: float
) -> Tuple[AerialGymTrajDataset, ...]:  # 2 tuple
    total_samples = dataset.num_trajs
    train_len = int(total_samples * (1 - val_split))
    val_len = total_samples - train_len

    return random_split(dataset, [train_len, val_len])
