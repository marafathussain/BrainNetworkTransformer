import torch

def mse(output: torch.Tensor, target: torch.Tensor):
    # Ensure both tensors have the same shape
    assert output.shape == target.shape, "Tensors must have the same shape"
    
    # Calculate MSE
    mse = torch.mean((output - target)**2)
    
    return mse.item()