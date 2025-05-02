import torch
import torch.fft

import numpy as np
import csv

def pft2d_configuration(N, M, mu, p, error, device='cpu'): 
    """
    Configure the 2D Partial Fourier Transform (PFT) by precomputing necessary parameters.

    Parameters:
        N (list of int): Sizes N1 and N2 of the input matrix along each dimension.
        M (list of int): Crop size M1 and M2 for the PFT along each dimension.
        mu (list of int): Centering for the PFT.
        p (list of int): Divisors p1 and p2 of N1 and N2, respectively.
        error (str): Error tolerance level, used to select precomputed values.
        device (str): Device to perform the computations on ('cpu' or 'cuda').

    Returns:
        B (list of torch.Tensor): Precomputed matrices B1 and B2
        p (list of int): Divisors p1 and p2 of N1 and N2, respectively (input as is).
        q (list of int): Quotients of N and p for each dimension.
        r (list of int): Precomputed r values for each dimension, based on error tolerance.
    """
    
    B = [0, 0]
    q = [0, 0]
    r = [0, 0]
    
    for d in range(2):
        q[d] = N[d] // p[d]

        r[d] = 0
        # Load precomputed xi
        csv_file = 'src/precomputed/' + error + ".csv"
        with open(csv_file, 'r') as file:
            XI = [float(val) for val in file.readline().split(',')]

        # Find r
        while XI[r[d]] < (M[d] / p[d]):
            r[d] += 1
            if r[d] == 25:
                break
        r[d] += 2

        # Load precomputed w
        row_number = r[d] - 1
        with open(csv_file, 'r') as file:
            selected_row = list(csv.reader(file))[row_number]

        W = torch.tensor([float(item) for item in selected_row]).to(device)

        # Generate B using precomputed w
        indices_l, indices_j = torch.meshgrid(torch.arange(q[d]).to(device), torch.arange(r[d]).to(device), indexing='ij')  

        exponent_term = torch.exp(-2j * np.pi * mu[d] * (indices_l - q[d] / 2) / N[d])

        # Compute the (1 - 2*l/q) * 1j term
        l_values = torch.arange(q[d]).to(device)
        coefficients = ((1 - 2 * l_values / q[d]) * 1j).unsqueeze(1) ** indices_j
        coefficients[coefficients.isnan()] = 1.0
    
        B_temp = exponent_term * W * coefficients

        B[d] = B_temp
    
    return B, p, q, r

def pft2d_precompute(Z, M, mu, p, q, r, device='cpu'):
    """
    Precompute values needed for the 2D Partial Fourier Transform (PFT) computation.

    This function prepares the input tensor and computes necessary pre-transform values,
    which are used in the actual PFT computation to improve efficiency.

    Parameters:
        Z (torch.Tensor): Input tensor to be transformed, size `N1 x N2`.
        M (list of int): Crop size M1 and M2 for the PFT along each dimension.
        mu (list of int): Centering for the PFT.
        p (list of int): Divisors p1 and p2 of N1 and N2, respectively.
        q (list of int): Quotients of N and p for each dimension.
        r (list of int): Precomputed r values for each dimension, based on error tolerance.
        device (str): Device to perform the computations on ('cpu' or 'cuda').

    Returns:
        Z (torch.Tensor): Reshaped input tensor to be used in the PFT computation.
        m1_mod (torch.Tensor): Modulo values for m1, corresponding to the first dimension.
        m2_mod (torch.Tensor): Modulo values for m2, corresponding to the second dimension.
        precomputed_prod (torch.Tensor): Precomputed product of exponential terms and 
                                         powers of `(m - mu)/p` for both dimensions, to be used in the PFT.
    """
    
    Z = Z.view(p[0], q[0], p[1], q[1]).permute(0, 2, 1, 3).contiguous().view(p[0], p[1], q[0], q[1])  

    # Calculate m1 and m2 ranges
    m1_range = torch.arange(mu[0] - M[0], mu[1] + M[0]).to(device)
    m2_range = torch.arange(mu[0] - M[1], mu[1] + M[1]).to(device)

    # Create grids for m1 and m2
    m1_grid, m2_grid = torch.meshgrid(m1_range, m2_range, indexing='ij')

    # Calculate the modulo values for m1 and m2
    m1_mod = m1_grid % p[0]
    m2_mod = m2_grid % p[1]

    # Compute powers of (m1 - mu[0])/p[0] and (m2 - mu[1])/p[1]
    m1_diff = (m1_grid - mu[0]) / p[0]
    m2_diff = (m2_grid - mu[1]) / p[1]

    # Calculate the exponential terms
    exp_term_m1 = torch.exp(-1j * torch.pi * m1_grid / p[0])
    exp_term_m2 = torch.exp(-1j * torch.pi * m2_grid / p[1])

    # Broadcast the exponential terms to match the shape of Ctil
    exp_term_m1 = exp_term_m1.unsqueeze(-1).unsqueeze(-1)
    exp_term_m2 = exp_term_m2.unsqueeze(-1).unsqueeze(-1)

    # Compute the powers of (m1 - mu[0])/p[0] and (m2 - mu[1])/p[1]
    m1_powers = m1_diff.unsqueeze(-1).unsqueeze(-1) ** torch.arange(r[0]).view(1, 1, r[0], 1).to(device)
    m2_powers = m2_diff.unsqueeze(-1).unsqueeze(-1) ** torch.arange(r[1]).view(1, 1, 1, r[1]).to(device)
    
    precomputed_prod = m1_powers * exp_term_m1 * m2_powers * exp_term_m2
    
    return Z, m1_mod, m2_mod, precomputed_prod

def pft2d_computation(Z, B, m1_mod, m2_mod, precomputed_prod, device='cpu'):
    """
    Perform the 2D Partial Fourier Transform (PFT) using precomputed matrices and values.

    This function carries out the core computation of the 2D PFT, which includes matrix multiplications,
    Fourier transforms, and element-wise operations to produce the final PFT result.

    Parameters:
        Z (torch.Tensor): Input tensor that has been preprocessed and reshaped for the PFT.
        B (list of torch.Tensor): Precomputed B matrices (B1 and B2) used in the PFT.
        m1_mod (torch.Tensor): Modulo values for the first dimension.
        m2_mod (torch.Tensor): Modulo values for the second dimension.
        precomputed_prod (torch.Tensor): Precomputed product of exponential terms and powers for both dimensions.
        device (str): Device to perform the computations on ('cpu' or 'cuda').

    Returns:
        pft_array (torch.Tensor): The resulting 2D Partial Fourier Transform array.
    """
    
    # Compute the matrix multiplication B1^T * Z * B2
    # Let N = N1*N2, M = M1*M2, p = p1*p2, q = q1*q2, r = r1*r2
    # NOTE: Associativity of matmul allows one to choose the parenthesization with lower cost!
    C = torch.matmul(torch.matmul(B[0].t(), Z), B[1]) # Cost: O(r * N)
    
    # Apply 2D FFT on the computed matrix C
    # The 2D FFT is applied across the first two dimensions (0 and 1)
    Ctil = torch.fft.fft2(C, dim=(0, 1)) # Cost: O(r * p log p)
    
    # Perform element-wise multiplication of Ctil with precomputed_prod (i.e. Hadamard product),
    # and sum the results over the last two dimensions (2 and 3)
    pft_array = torch.sum(Ctil[m1_mod, m2_mod] * precomputed_prod, dim=(2, 3)).to(device) 
    
    return pft_array
