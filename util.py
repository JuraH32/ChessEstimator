import torch


def get_device():
    # CUDA for PyTorch
    if torch.cuda.is_available():
        return torch.device('cuda')
    # MPS for PyTorch
    # elif torch.backends.mps.is_available():
    #     return torch.device('mps')

    return torch.device('cpu')