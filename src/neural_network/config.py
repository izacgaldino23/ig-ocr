import torch

from src.neural_network.args import get_args


def init_config(force_cpu=False):
    # 1. Initialize configuration
    args = get_args()

    # Use the NVIDIA GPU (CUDA) if available; otherwise, use the CPU.
    if force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"1. Running on {device.type}")

    return args, device
