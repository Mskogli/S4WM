import torch
import ujson

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
        print("===LOADING DATASET===")
        with open(json_path) as file:
            self.lines = file.readlines()
        print("===DATASET LOADED===")

        self.device = device
        self.actions = actions
        self.states = states

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx) -> torch.tensor:
        json_object = ujson.loads(self.lines[idx])


        latents = [torch.tensor(json_object["latents"], device=self.device)]

        actions = (
            [torch.tensor(json_object["action"], device=self.device)]
            if self.actions
            else []
        )

        states = [
            torch.tensor(
                json_object["states"],
                device=self.device,
            )[:, StateIndices[state][0] : StateIndices[state][1]]
            for state in self.states
        ]

        item = torch.cat(latents + states + actions, 1)

        return item


def split_dataset(
    dataset: AerialGymTrajDataset, val_split: float
) -> Tuple[AerialGymTrajDataset, ...]:  # 2 tuple

    total_samples = len(dataset.lines)
    train_len = int(total_samples * (1 - val_split))
    val_len = total_samples - train_len

    train_data, val_data = random_split(dataset, [train_len, val_len])
    return train_data, val_data
