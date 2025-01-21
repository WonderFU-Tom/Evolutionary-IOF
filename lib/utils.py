# MIT License

# Copyright (c) # Copyright (c) 2024 anonymity

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch

from lib import config



def Ham(spin, J_):
    """Calculate Hamiltonian for spin configurations."""
    # Ensure spin and J_ are PyTorch tensors
    spin = torch.sign(spin)  # Ensure spin values are either -1 or 1
    Hamiltonian = -0.5 * torch.sum(torch.matmul(spin, J_) * spin, dim=1)
#     Hamiltonian = -0.5 * torch.matmul(torch.matmul(spin, J_) , spin.T)
    return Hamiltonian

def Cut(spin, J):
    """
    Calculate the cut value for the spin configurations.

    Parameters
    ----------
    spin : torch.Tensor
        Tensor of shape (batch_size, spins) representing spin configurations.
    J : torch.Tensor
        Tensor of shape (spins, spins) representing the coupling matrix.

    Returns
    -------
    torch.Tensor
        Tensor of shape (batch_size,) containing the cut value for each spin configuration.
    """
    # Calculate Hamiltonian
    H = Ham(spin, J)
    # Calculate cut value
    cut_value = (-0.5 * H) - 0.25 * torch.sum(J).item()
    return cut_value

def Move(spin, flipped_bit_indices, batch_selection_indices):
    """
    Flip the spins at specified indices for each batch.

    Parameters
    ----------
    spin : torch.Tensor
        Tensor of shape (batch_size, spins) representing spin configurations.
    indices : torch.Tensor
        Tensor of shape (batch_size,) representing the indices to be flipped for each batch.

    Returns
    -------
    torch.Tensor
        Updated spin tensor with specified indices flipped.
    """

    # Use advanced indexing to flip the specified indices
    spin[batch_selection_indices, flipped_bit_indices] = -spin[batch_selection_indices, flipped_bit_indices]
    
    return spin


def greedy(J_, batch_size, spin):
        flipped_spins = -spin
        # Find negative indices
        gain_list = (-2 * torch.sign(flipped_spins) * torch.matmul( torch.sign(spin), J_ ))
        negative_indices = torch.nonzero(gain_list < 0, as_tuple=False)
        batch_indices = negative_indices[:,0]
        element_indices = negative_indices[:, 1]

        # Initialize list to store indices for each batch
#         batch_negative_indices = [torch.tensor([], dtype=torch.long) for _ in range(batch_size)]
        batch_negative_indices = [[] for _ in range(batch_size)]

        # Populate the batch_negative_indices list
        for b in range(batch_size):
            # Get element indices for the current batch
            batch_negative_indices[b] = element_indices[batch_indices == b]
        
        batch4greedy  = list(tensor.numel() != 0 for tensor in batch_negative_indices)
        # index4greedy = torch.nonzero(torch.tensor(batch4greedy), as_tuple=False).squeeze()
        not_all_empty = any(batch4greedy)

        return not_all_empty, batch_negative_indices, batch4greedy
