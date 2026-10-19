import torch

from src.args import get_args


def init_config():
    # 1. Configuração inicial
    args = get_args()

    # Verifica se a GPU NVIDIA (CUDA) está disponível; se não, usa a CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Executando em {device.type}")

    return args, device
