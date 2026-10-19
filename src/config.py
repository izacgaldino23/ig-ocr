import torch

from src.args import get_args


def init_config(force_cpu=False):
    # 1. Configuração inicial
    args = get_args()

    # Verifica se a GPU NVIDIA (CUDA) está disponível; se não, usa a CPU
    if force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"1. Executando em {device.type}")

    return args, device
