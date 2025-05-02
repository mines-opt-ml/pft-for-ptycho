import torch
import torch.fft

import numpy as np
import csv

def pft_configuration(N, M, mu, p, error, device='cpu'):
    """
    Configure the Partial Fourier Transform (PFT) by precomputing necessary parameters.

    Parameters:
        N (int): Size of the input signal or data array.
        M (int): Number of low-frequency components to retain, i.e. crop size.
        mu (float): Center for the PFT.
        p (int): Divisor of N.
        error (str): Error tolerance level, used to select precomputed values.
        device (str): Device to perform the computations on ('cpu' or 'cuda').

    Returns:
        B (torch.Tensor): Precomputed configuration matrix for the PFT.
        p (int): Divisor of N (input as is).
        q (int): Quotient of N divided by p.
        r (int): Precomputed r value, based on the error threshold.
    """
    
    q = N // p
    
    r = 0
    # Load precomputed xi
    csv_file = 'src/precomputed/' + error + ".csv"
    with open(csv_file, 'r') as file:
        XI = [float(val) for val in file.readline().split(',')]

    # Find r
    r = min(25, next((i for i, x in enumerate(XI) if x >= M / p), 25)) + 2
    
    # Load precomputed w
    row_number = r - 1
    with open(csv_file, 'r') as file:
        selected_row = list(csv.reader(file))[row_number]

        W = torch.tensor([float(item) for item in selected_row], device=device)
    
    # Generate B using precomputed w
    indices_l, indices_j = torch.meshgrid(torch.arange(q, device=device), torch.arange(r, device=device), indexing='ij')  

    exponent_term = torch.exp(-2j * np.pi * mu * (indices_l - q / 2) / N)

    # Compute the (1 - 2*l/q) * 1j term
    l_values = torch.arange(q, device=device)
    coefficients = ((1 - 2 * l_values / q) * 1j).unsqueeze(1) ** indices_j
    coefficients[coefficients.isnan()] = 1.0

    B = exponent_term * W * coefficients
    
    return B, p, q, r

def pft_precompute(Z, M, mu, p, q, r, device='cpu'):
    """
    Precompute values needed for the Partial Fourier Transform (PFT) computation.

    This function prepares the input data and computes necessary pre-transform values,
    which are used in the actual PFT computation to improve efficiency.

    Parameters:
        Z (torch.Tensor): Input data array to be transformed, size N.
        M (int): Crop size for the PFT.
        mu (float): Center for the PFT.
        p (int): Divisor of N.
        q (int): Quotient of N divided by p.
        r (int): Precomputed r value, based on error tolerance.

    Returns:
        Z (torch.Tensor): Reshaped input data array to be used in the PFT computation.
        m_mod (torch.Tensor): Modulo values for m, used in the PFT computation.
        precomputed_prod (torch.Tensor): Precomputed product of exponential terms and 
                                         powers of (m - mu)/p to be used in the PFT.
    """
    
    # Reshape data array Z
    Z = Z.view(p, q)
    m_indices = torch.arange(mu - M, mu + M + 1, device=device)
    j_indices = torch.arange(r, device=device)

    # Create a grid of m and j indices using broadcasting
    m_grid, j_grid = torch.meshgrid(m_indices, j_indices, indexing='ij')

    # Calculate the modulo values for m
    m_mod = m_indices % p
    
    # Calculate the powers term ((m - mu)/p) ^ j using broadcasting
    powers_term = ((m_grid - mu) / p) ** j_grid

    # Calculate the exponential term using broadcasting
    exponential_term = torch.exp(-1j * torch.pi * m_grid / p)
    
    precomputed_prod = powers_term * exponential_term
    
    return Z, m_mod, precomputed_prod

def pft_computation(Z, B, m_mod, precomputed_prod, device='cpu'):
    """
    Perform the Partial Fourier Transform (PFT) using precomputed matrices and values.

    This function carries out the core computation of the PFT, which includes a matrix multiplication,
    batch Fourier transforms, and element-wise operations to produce the final PFT result.

    Parameters:
        Z (torch.Tensor): Reshaped input data array to be transformed.
        B (torch.Tensor): Precomputed configuration matrix for the PFT.
        m_mod (torch.Tensor): Modulo values for m, used in the PFT computation.
        precomputed_prod (torch.Tensor): Precomputed product of exponential terms and powers for the PFT.

    Returns:
        pft_array (torch.Tensor): The resulting Partial Fourier Transform array.
    """
    # Compute C = Z x B
    C = torch.matmul(Z, B) # Cost: O(r * N)
    
    # Compute FFT(C[k, j]) with respect to k
    Ctil = torch.fft.fft(C, dim = 0) # O(r * p log p)
    
    # Calculate pft_array using torch.sum along the j-axis
    pft_array = torch.sum(Ctil[m_mod] * precomputed_prod, dim=1)
    
    return pft_array

