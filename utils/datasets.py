import torch

from .ag_traj_dataset import AerialGymTrajDataset, split_dataset

def create_quad_depth_trajectories_datasets(bsz=128):
    print("[*] Generating Aerial Gym Trajectory Dataset")

    N_CLASSES, SEQ_LENGTH, IN_DIM = 128, 96, 132

    dataset = AerialGymTrajDataset(
        "/home/mathias/dev/trajectories.jsonl",
        "cpu",
        actions=True,
    )

    train_dataset, val_dataset = split_dataset(dataset, 0.1)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True)

    return trainloader, valloader, N_CLASSES, SEQ_LENGTH, IN_DIM


Datasets = {
    "quad_depth_trajectories": create_quad_depth_trajectories_datasets,
}