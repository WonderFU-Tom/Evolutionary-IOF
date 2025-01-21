# MIT License

# Copyright (c) 2024 anonymity

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
import numpy as np


def swarm_evolution(alpha, beta, x, J, J_, iterations=200, BC=False, k=1, device='cpu'):
    batch_size, spins = x.shape
    
    s = [x]  # Initialize the list to store spin states at each iteration
    error = torch.ones(spins).to(device)
    target_amp = 0.03
    t_BC = int(iterations/2)
    
    for i in range(iterations):  # Iterative simulation
        alpha0 = np.tanh(i)*alpha
        f = alpha0 * x + beta * error * torch.matmul(x, J)
        error = -(x**2 - target_amp) * error + error
        x = f / torch.cosh(f)

        if BC and i == t_BC:
            # T1 = time.perf_counter()
            inter_var = torch.abs(x) - k * torch.linalg.norm(x, dim=1, keepdim=True) / torch.sqrt(torch.tensor(spins, device=device, dtype=torch.float32))
            b_x = torch.sign(torch.sign(inter_var) + 1) * x
            zero_indices = (b_x == 0).nonzero(as_tuple=True)
            for batch_idx in range(batch_size):
                for zero_idx in zero_indices[1][zero_indices[0] == batch_idx]:
                    b_x[batch_idx, zero_idx] = torch.sign(torch.matmul(J_[:, zero_idx], b_x[batch_idx]))

            for j in range(spins):
                b_x[:, j] = torch.sign(torch.matmul(J_[:, j], b_x.T))
                zero_mask = (b_x[:, j] == 0)
                b_x[zero_mask, j] = torch.randint(low=-1, high=2, size=(torch.sum(zero_mask),), device=device).float()


            x = b_x
            
            
            # T2 = time.perf_counter()
            # print(f"BC time：{1000*(T2-T1)} ms")

        s.append(x)

    # Take the sign bit of simulated spin
    spin_states_all = torch.sign(torch.stack(s))
    s = torch.stack(s)

    Hamiltonian = torch.zeros(len(s), batch_size, device=device)
    for j in range(len(s)):  # Compute system energy
        Hamiltonian[j] = (-0.5) * torch.sum(spin_states_all[j] @ J_ * spin_states_all[j], dim=1)

    min_energy_idx = torch.argmin(Hamiltonian, dim=0)
    best_spins = s[min_energy_idx, torch.arange(batch_size, device=device)]
    best_energy = torch.min(Hamiltonian, dim=0)[0]
    
    return best_spins, best_energy


