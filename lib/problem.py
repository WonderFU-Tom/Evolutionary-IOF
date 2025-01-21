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



def graph(g, device, mu, SD, lattice_size=1024):
    if g[0] == 'G':
        directory = "../G-set/"
        Graph = np.genfromtxt(directory + g + '.txt', skip_header=1, dtype=int)
        spins = np.genfromtxt(directory + g + '.txt', max_rows=1, dtype=int)[0]
        max_cuts = np.genfromtxt(directory + 'max_cuts' + '.txt', dtype=int)

        Graph[:,0:2] = Graph[:,0:2]-1  

        J = np.zeros([spins,spins])

        for i in range(len(Graph)):
            J[Graph[i][0]][Graph[i][1]] = -Graph[i][2]

        J = np.triu(J) + np.triu(J).T
        max_cut = max_cuts[eval(g[1:])-1]
        global_minima = ( max_cut + 0.25*np.sum(J) ) / -0.5


    elif g == 'square lattice':
        spins = lattice_size
        n = int(np.sqrt(spins))
        global_minima = -2 * spins
        max_cut = 2 * spins
        J = np.zeros([spins,spins])

        for i in range(spins):   

            if (i-n) < 0:
                up_spin = i - n + n*n
            else:
                up_spin = i - n
            
            right_spin = (i // n) * n + (i+1) % n
            under_spin = (i+n) % (n*n)
            
            if i % n == 0:
                left_spin = i + n - 1
            else:
                left_spin = i-1
            
            J[i][up_spin] = -1
            J[i][right_spin] = -1
            J[i][under_spin] = -1
            J[i][left_spin] = -1
    
    elif g == 'triangular lattice':
        spins = lattice_size
        n = int(np.sqrt(spins))
        global_minima = -1 * spins
        max_cut = 2 * spins
        J = np.zeros([spins,spins])

        for i in range(spins):   

            if (i-n) < 0:
                up_spin = i - n + n*n
                lu_spin = i - n + n*n - 1
            else:
                up_spin = i - n
                lu_spin = i - n - 1
            
            right_spin = (i // n) * n + (i+1) % n
            under_spin = (i+n) % (n*n)
            ru_spin = (right_spin+n) % (n*n)
            
            if i % n == 0:
                left_spin = i + n - 1
                lu_spin = i- 1
            else:
                left_spin = i-1
                
            
            J[i][up_spin] = -1
            J[i][right_spin] = -1
            J[i][under_spin] = -1
            J[i][left_spin] = -1
            J[i][lu_spin] = -1
            J[i][ru_spin] = -1

    elif g == 'mobius ladder':
        spins = lattice_size
        n = int(np.trunc(spins/2))
        global_minima = - (3 * spins / 2 - 4)
        max_cut = -global_minima + 2
        J = np.zeros([spins,spins])

        for i in range(n):  
            forward_spin = i+1
            back_spin = i-1
            opposite_spin = i + n

            J[i][forward_spin] = -1
            J[i][back_spin] = -1
            J[i][opposite_spin] = -1

        for i in range(n, spins-1):
            forward_spin = i+1
            back_spin = i-1
            opposite_spin = i + n - spins

            J[i][forward_spin] = -1
            J[i][back_spin] = -1
            J[i][opposite_spin] = -1

        J[spins-1][0] = -1
        J[spins-1][spins-2] = -1
        J[spins-1][n-1] = -1



    elif g == 'cubic lattice':
        layers = 4
        spins = int(lattice_size/layers)
        n = int(np.sqrt(spins))
        global_minima = -3*layers * spins
        max_cut = -global_minima
        J = np.zeros([spins,spins])

        for i in range(spins):  

            up_spin = i - n
            right_spin = (i // n) * n + (i+1) % n
            under_spin = (i+n) % (n*n)

            if i % n == 0:
                left_spin = i + n - 1
            else:
                left_spin = i-1

            J[i][up_spin] = -1
            J[i][right_spin] = -1
            J[i][under_spin] = -1
            J[i][left_spin] = -1

        J0 = np.zeros([layers*spins,layers*spins])    
        for i in range(layers):
            J0[i*spins:(i+1)*spins,i*spins:(i+1)*spins] = J

        for i in range(layers*spins):
            
            over_spin = i - spins
            below_spin = (i + spins) % (layers*spins)
            
            J0[i][over_spin] = -1
            J0[i][below_spin] = -1

        J = J0
        spins = lattice_size

    print(g,'  max eigval：', max(np.linalg.eigvals(J)), "  Ground State", global_minima, "  Max Cut：", max_cut)
    J_ = np.copy(J)
    me = max(np.linalg.eigvals(J))

    if SD:
        #******************* Eigenvalue Dropout ***********************
        U,D,V = np.linalg.svd(J)
        drop_bit = np.zeros(spins)
        drop_bit[-mu:] = 100
        D = np.real(np.sqrt(D - drop_bit + 0j)) **2      
        D = np.diag(D)
        J = np.real(np.round(U@D@V, 5))


        J = torch.from_numpy(J)
        J = J.float().to(device)
        J_ = torch.from_numpy(J_)
        J_ = J_.float().to(device)

    return J, J_, me, global_minima


