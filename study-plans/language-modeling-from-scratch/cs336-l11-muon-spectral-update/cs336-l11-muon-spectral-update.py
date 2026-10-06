import torch

def muon_spectral_update(
    parameter: torch.Tensor, gradient: torch.Tensor,
    previous_momentum: torch.Tensor, momentum_coefficient: int | float,
    learning_rate: int | float,
) -> dict:
    """
    Returns a dict of tensors: new_parameter, new_momentum, orthogonalized_update.
    """
    updated_momentum = momentum_coefficient*previous_momentum + gradient
    updated_momentum_fp32 = updated_momentum.float()
    U, S, Vh = torch.linalg.svd(updated_momentum_fp32, full_matrices = False)
    spectral_update = (U@Vh).to(parameter.dtype)
    new_parameter = parameter - learning_rate*spectral_update

    return {"new_parameter":new_parameter,"new_momentum":updated_momentum,"orthogonalized_update":spectral_update}
