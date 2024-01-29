import torch
import torchvision
import torchvision.transforms as transforms

from .ag_traj_dataset import AerialGymTrajDataset

def create_mnist_dataset(bsz=128):
    print("[*] Generating MNIST Sequence Modeling Dataset...")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 784, 256, 1

    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: (x.view(IN_DIM, SEQ_LENGTH).t() * 255).int()
            ),
        ]
    )

    train = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=tf
    )
    test = torchvision.datasets.MNIST(
        "./data", train=False, download=True, transform=tf
    )

    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(
        train,
        batch_size=bsz,
        shuffle=True,
    )
    
    testloader = torch.utils.data.DataLoader(
        test,
        batch_size=bsz,
        shuffle=False,
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM

def create_quad_depth_trajectories_datasets(bsz=128):
    print("[*] Generating Aerial Gym Trajectory Dataset")


    dataset = AerialGymTrajDataset(
        "/home/mathias/dev/trajectories.jsonl",
        "cuda:0",
        actions=True,
    )

    train_dataset, val_dataset = torch.utils.split_dataset(dataset, 0.1)
    trainloader = torch.utils.DataLoader(train_dataset, batch_size=bsz, shuffle=True)
    valloader = torch.utils.DataLoader(val_dataset, batch_size=bsz, shuffle=True)

    return trainloader, valloader

Datasets = {
    "quad_depth_trajectories": create_mnist_dataset,
    "quad_depth_trajectories": create_quad_depth_trajectories_datasets,
}