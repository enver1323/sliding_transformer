import torch


def compute_path_increments(inputs) -> torch.Tensor:
    path_increments = torch.empty((inputs.size(-2) - 1, inputs.size(-1)), dtype=inputs.dtype, device=inputs.device)

    for i in range(path_increments.size(-2)):
        path_increments[i] = inputs[i + 1] - inputs[i]

    return path_increments
