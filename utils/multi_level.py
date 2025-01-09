import torch

def torch_unique_with_indices(tensor, dim=0,):
    """Return the unique elements of a tensor and their indices."""
    with torch.no_grad():
        unique, inverse_indices, counts = torch.unique(tensor, return_inverse=True, dim=dim, return_counts=True)
        # indices = torch.scatter_reduce(
        #     torch.zeros_like(unique, dtype=torch.long, device=tensor.device), 
        #     dim=0,
        #     index=inverse_indices,
        #     src=torch.arange(tensor.size(0), device=tensor.device),
        #     reduce="amin",
        #     include_self=False,
        # )
        # indices = torch.scatter_reduce(
        #     input=torch.arange(tensor.size(0), device=tensor.device), 
        #     dim=dim,
        #     index=inverse_indices,
        #     reduce="amin",
        # )
        # indices = torch.zeros_like(inverse_indices)
        # indices.scatter_(dim=dim, index=inverse_indices, src=torch.arange(tensor.size(0), device=tensor.device))[:unique.shape[0]]
        indices = torch.scatter_reduce(
            input=torch.zeros_like(unique[:,0], device=tensor.device, dtype=torch.double),
            src=torch.arange(tensor.size(0), device=tensor.device, dtype=torch.double), 
            dim=dim,
            index=inverse_indices,
            reduce="amin",
            include_self=False,
        ).long()
        return unique, inverse_indices, indices, counts