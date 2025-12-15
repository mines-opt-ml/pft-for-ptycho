import torch
import torch.fft

import numpy as np
import csv

def pft2d_configuration(N, M, mu, p, error, device='cpu', dtype=torch.complex64): 
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

        B[d] = B_temp.to(dtype)
    
    return B, p, q, r

def pft2d_precompute(Z, M, mu, p, q, r, *, device='cpu', dtype=torch.complex64):
    """
    Precompute values needed for the 2D Partial Fourier Transform (PFT) computation.

    This function prepares the input tensor and computes necessary pre-transform values,
    which are used in the actual PFT computation to improve efficiency.

    Parameters:
        Z (torch.Tensor):
            Input tensor of size (N1, N2), possibly real or complex.
        M (list[int]):
            Half-widths [M1, M2] of the frequency crop in each dimension.
            The resulting frequency ranges have lengths 2*M1+1 and 2*M2+1.
        mu (list[int]):
            Frequency centers [mu1, mu2] for the PFT.
        p (list[int]):
            Divisors [p1, p2] of N1 and N2, respectively.
        q (list[int]):
            Quotients [q1, q2] such that N_d = p_d * q_d.
        r (list[int]):
            Truncation orders [r1, r2] for the Taylor expansions in each dimension.
        device (str or torch.device, optional):
            Device on which to perform computations. Defaults to Z.device.
        dtype (torch.dtype, optional):
            Complex dtype used for returned factors (default: torch.complex64).

    Returns:
        Z (torch.Tensor):
            Reshaped input tensor of shape (p1, p2, q1, q2), to be used in the PFT.
        m1_mod (torch.Tensor):
            1D tensor of length L1 = 2*M1+1 containing modulo-p1 indices
            for the first dimension.
        m2_mod (torch.Tensor):
            1D tensor of length L2 = 2*M2+1 containing modulo-p2 indices
            for the second dimension.
        F1 (torch.Tensor):
            Tensor of shape (L1, r1) containing precomputed weights
            ((m1 - mu1)/p1)^j * exp(-pi i m1 / p1) for the first dimension.
        F2 (torch.Tensor):
            Tensor of shape (L2, r2) containing precomputed weights
            ((m2 - mu2)/p2)^j * exp(-pi i m2 / p2) for the second dimension.
    """
    
    if device is None:
        device = Z.device
    
    # --- reshape Z to (p1, p2, q1, q2) ---
    Z = (
        Z.view(p[0], q[0], p[1], q[1])
         .permute(0, 2, 1, 3)
         .contiguous()
    )

    # real dtype for forming polynomial powers
    real_dtype = Z.real.dtype

    # --- 1-D m ranges: mu_d - M_d ... mu_d + M_d (inclusive) ---
    m1_vals = torch.arange(mu[0] - M[0], mu[0] + M[0], device=device)
    m2_vals = torch.arange(mu[1] - M[1], mu[1] + M[1], device=device)

    # modulo indices into the FFT grid (handle negatives correctly)
    m1_mod = torch.remainder(m1_vals, p[0]).to(torch.long)
    m2_mod = torch.remainder(m2_vals, p[1]).to(torch.long)

    # --- Compact factors per dim: F_d[l, j] = ((m_d - mu_d)/p_d)^j * exp(-pi i m_d/p_d) ---
    a1 = (m1_vals - mu[0]).to(real_dtype) / float(p[0])  # (L1,)
    a2 = (m2_vals - mu[1]).to(real_dtype) / float(p[1])  # (L2,)

    j1 = torch.arange(r[0], device=device, dtype=real_dtype)
    j2 = torch.arange(r[1], device=device, dtype=real_dtype)

    # polynomial terms
    P1 = torch.pow(a1.unsqueeze(1), j1.unsqueeze(0))  # (L1, r1)
    P2 = torch.pow(a2.unsqueeze(1), j2.unsqueeze(0))  # (L2, r2)

    # exponential phase terms
    E1 = torch.exp(-1j * torch.pi * m1_vals / float(p[0])).to(dtype).unsqueeze(1)
    E2 = torch.exp(-1j * torch.pi * m2_vals / float(p[1])).to(dtype).unsqueeze(1)

    # compact per-dimension factors
    F1 = P1.to(dtype) * E1  # (L1, r1)
    F2 = P2.to(dtype) * E2  # (L2, r2)

    return Z, m1_mod, m2_mod, F1, F2

def pft2d_computation(Z, B, m1_mod, m2_mod, F1, F2, device='cpu', dtype=torch.complex64):
    """
    Perform the 2D Partial Fourier Transform (PFT) using precomputed matrices and values.

    This function carries out the core computation of the 2D PFT, which includes matrix multiplications,
    Fourier transforms, and element-wise operations to produce the final PFT result.

    Parameters:
        Z (torch.Tensor):
            Preprocessed input tensor of shape (p1, p2, q1, q2).
        B (list[torch.Tensor]):
            List [B1, B2] of basis matrices with shapes:
              - B1: (q1, r1)
              - B2: (q2, r2)
        m1_mod (torch.Tensor):
            1D tensor of length L1 containing modulo-p1 frequency indices.
        m2_mod (torch.Tensor):
            1D tensor of length L2 containing modulo-p2 frequency indices.
        F1 (torch.Tensor):
            Tensor of shape (L1, r1) containing precomputed factors
            for the first dimension.
        F2 (torch.Tensor):
            Tensor of shape (L2, r2) containing precomputed factors
            for the second dimension.

    Returns:
        pft_array (torch.Tensor):
            The resulting 2D Partial Fourier Transform array of shape (L1, L2).
    """

    # enforce dtype/device
    Z  = Z.to(device=device, dtype=dtype)
    B0 = B[0].to(device=device, dtype=dtype)
    B1 = B[1].to(device=device, dtype=dtype)
    F1 = F1.to(device=device, dtype=dtype)
    F2 = F2.to(device=device, dtype=dtype)
    
    # --- 1) Compute C = B1^T * Z * B2 ---
    # Contract q1 with B[0]: (p1,p2,q1,q2) -> (p1,p2,r1,q2)
    C = torch.tensordot(Z, B[0], dims=([2], [0]))   # (p1,p2,q2,r1)
    C = C.permute(0, 1, 3, 2).contiguous()          # (p1,p2,r1,q2)

    # Contract q2 with B[1]: -> (p1,p2,r1,r2)
    C = torch.tensordot(C, B[1], dims=([3], [0]))   # (p1,p2,r1,r2)

    # --- 2) 2D FFT over the p-dimensions ---
    Ctil = torch.fft.fft2(C, dim=(0, 1))            # (p1,p2,r1,r2)

    # --- 3) Gather desired frequency indices ---
    A = Ctil.index_select(0, m1_mod).index_select(1, m2_mod)
    # A: (L1,L2,r1,r2)

    # --- 4) Contract with compact factors ---
    # Equivalent to sum_{j1,j2} A * F1 * F2, but done in two steps
    T = torch.einsum('abij,ai->abj', A, F1)          # (L1,L2,r2)
    pft_array = torch.einsum('abj,bj->ab', T, F2)   # (L1,L2)

    return pft_array