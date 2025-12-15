import torch
import torch.fft

import numpy as np
import csv

PI = math.pi
ALPHA = 0.28513442812268959006
EPSILON = 1e-7

#################### HELPER FUNCTIONS ####################
def _gather_axis_1d(x: torch.Tensor, axis: int, idx_1d: torch.Tensor) -> torch.Tensor:
    """
    Efficiently gather along a single axis using 1D indices (no broadcasted index builds).
    """
    x = x.movedim(axis, 0).contiguous()   # [axis, ...]
    out = x.index_select(0, idx_1d)       # [2M_d, ...]
    return out.movedim(0, axis)           # restore axis position


def _contract_axis_after_gather(
    sampled: torch.Tensor,
    d: int,
    r_axis_pos: int,
    V_d: torch.Tensor,     # (2M_d, r_d)
    idx_1d: torch.Tensor,  # (2M_d,)
) -> torch.Tensor:
    """
    Fused step for dimension d:
      1) Gather along p-axis d (size -> 2M_d)
      2) Move r_d next to that axis
      3) Multiply by V_d and sum over r_d, keeping the 2M_d axis
    """
    # 1) gather along M-axis d
    sampled = _gather_axis_1d(sampled, d, idx_1d)  # [..., 2M_d, ...]

    # 2) move r_d beside M_d so shape is [..., 2M_d, r_d, ...]
    if r_axis_pos < 0:
        r_axis_pos = sampled.ndim + r_axis_pos
    sampled = sampled.movedim(r_axis_pos, d + 1)

    # 3) broadcast V_d to [..., 2M_d, r_d, ...] and sum over r_d
    view = [1] * sampled.ndim
    view[d] = V_d.shape[0]         # 2M_d
    view[d + 1] = V_d.shape[1]     # r_d
    Vb = V_d.view(*view)

    sampled = (sampled * Vb).sum(dim=d + 1)  # sum r_d; keep 2M_d at dim d
    return sampled.contiguous()

def p_func(r, M):
    rf = math.gamma(r + 1.0)
    term1 = (M * math.pi / 2.0) # (M * PI / 2.0)
    term2 = (ALPHA * EPSILON * rf) ** (-1.0 / r) # (std::pow(ALPHA * EPSILON * rf, -1.0 / r))
    term3 = math.exp(-1.0 / (r * (r + 1)) * (ALPHA * EPSILON * rf) ** (2.0 / r)) # std::exp(-1.0 / (r * (r + 1)) * std::pow(ALPHA * EPSILON * rf, 2.0 / r))

    return term1 * term2 * term3

def step(r, N, M):
    h = 1e-6
    v_p = p_func(r, M)
    forward = p_func(r + h, M)
    backward = p_func(r - h, M)
    
    v_pp = (forward - backward) / (2.0 * h)
    v_pdp = (forward + backward - 2.0 * v_p) / (h * h)
    lv_p = math.log(v_p)

    numerator = (N + 4.0 * v_p * lv_p + 4.0 * M) + 4.0 * r * v_pp * (1.0 + lv_p)
    denominator = 4.0 * (1.0 + lv_p) * (2.0 * v_pp + r * v_pdp) + 4.0 * r * v_pp * v_pp / v_p
    
    return numerator / denominator

# Newton's method
def find_minimizer(N, M):
    r = 10.0
    tolerance = 1e-2
    max_iterations = 1000
    iteration = 0

    for iteration in range(max_iterations):
        delta_r = -step(r, N, M)
        r += delta_r

        # Check convergence
        if abs(delta_r) < tolerance:
            return r

    print(f"Did not converge after {max_iterations} iterations.")
    return float('nan')

def all_divisors_excluding_trivial(N: int):
    """Return all nontrivial positive divisors of N (exclude 1 and N), sorted."""
    divs = []
    limit = int(math.isqrt(N))
    for d in range(1, limit + 1):
        if N % d == 0:
            q = N // d
            if d not in (1, N):
                divs.append(d)
            if q != d and q not in (1, N):
                divs.append(q)
    divs.sort()
    return divs

def nearest_divisor(N: int, x: float, prefer_leq_sqrt: bool = True):
    """
    Return the divisor of N (excluding 1 and N) closest to x.
    Tie-break:
      - if prefer_leq_sqrt is True, prefer the factor <= sqrt(N) when distances tie,
      - otherwise prefer the smaller factor.
    """
    divs = all_divisors_excluding_trivial(N)
    if not divs:  # N prime (shouldn't happen here, but safe-guard)
        return None

    sqrtN = math.sqrt(N)

    def keyfn(d):
        # Sort by distance to x, then tie-break
        # prefer factor <= sqrt(N) if requested; else prefer smaller d
        tie1 = 0 if (prefer_leq_sqrt and d <= sqrtN) else 1
        return (abs(d - x), tie1, d)

    return min(divs, key=keyfn)

#################### MAIN AUTO-MPFT FUNCTIONS ####################
def mpft_configuration(N, M, mu, error, device='cpu'):

    D = len(M)
    
    B = [0] * D
    p = [0] * D
    q = [0] * D
    r = [0] * D

    for d in range(D):
        r[d] = find_minimizer(N[d], M[d])
        optimal_p = p_func(r[d], M[d])
        p[d] = nearest_divisor(N[d], optimal_p)

        q[d] = N[d] // p[d]
        r[d] = math.floor(r[d])

        # Load precomputed w
        row_number = int(r[d] - 1)
        csv_file = 'src/precomputed/' + error + ".csv"
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

        B[d] = B_temp.to(device)

    return B, p, q, r

def mpft_precompute(Z, M, mu, p, q, r, device="cpu"):
    """
    Prepare inputs without materializing any (∏ 2M) × r_d objects.
    
    Returns:
      Zr         : tensor shaped [p1,...,pD, q1,...,qD] on device
      idx_mod_1d : list length D; idx_mod_1d[d] is (2M_d,) int64 in [0, p[d])-1
      V_1d       : list length D; V_1d[d] is (2M_d, r_d) complex64
                   with entries ((m_d - μ_d)/p_d)^{j_d} * exp(-i π m_d / p_d)
    """
    D = len(p)
    assert Z.ndim == D
    
    # Reshape [p1,q1,p2,q2,...] -> [p1,...,pD, q1,...,qD]
    interleaved = []
    for d in range(D):
        interleaved.extend([p[d], q[d]])
    Zr = Z.reshape(*interleaved)
    perm = [2 * k for k in range(D)] + [2 * k + 1 for k in range(D)]
    Zr = Zr.permute(*perm).contiguous().to(device)
    
    idx_mod_1d: List[torch.Tensor] = []
    V_1d: List[torch.Tensor] = []
    
    for d in range(D):
        md = torch.arange(mu[d] - M[d], mu[d] + M[d], device=device, dtype=torch.float32)  # (2M_d,)
        idx_mod_1d.append((md.to(torch.long) % p[d]).to(torch.long))                        # (2M_d,)
    
        diff = (md - float(mu[d])) / float(p[d])                                           # (2M_d,)
        jv = torch.arange(r[d], device=device, dtype=torch.float32)                         # (r_d,)
        pow_d = torch.nan_to_num(diff.unsqueeze(1) ** jv.unsqueeze(0), nan=1.0)            # (2M_d, r_d)
        exp_d = torch.exp(-1j * torch.pi * md / float(p[d]))                                # (2M_d,)
    
        V_1d.append((pow_d.to(torch.complex64) * exp_d.unsqueeze(1)).contiguous())         # (2M_d, r_d)
    
    return Zr, idx_mod_1d, V_1d

def mpft_computation(Z, B, idx_mod_1d, V_1d, device='cpu', order=None, r_contract_order=None):
    """
    Auto-MPFT with **space-lean, fused sampling + r-contraction**.
    
    Inputs:
      Z         : [p1,...,pD, q1,...,qD] (from mpft_precompute)
      B         : list of D matrices B[d] with shape (q_d, r_d)
      idx_mod_1d: per-axis 1D indices into p-axes (length 2M_d)
      V_1d      : per-axis factors (2M_d, r_d)
      order     : (optional) order of applying B^(d) along q-axes; default largest q first
    
    Output:
      Tensor of shape (2M1,...,2MD) [complex64, contiguous]
    """
    D = len(B)
    q_sizes = [int(B[d].shape[0]) for d in range(D)]
    r_sizes = [int(B[d].shape[1]) for d in range(D)]
    
    # 1) Multiply by B^(d) along q-axes (default: largest q first)
    if order is None:
        order = [i for i, _ in sorted(enumerate(q_sizes), key=lambda x: -x[1])]
    
    A = Z.to(device).contiguous()               # [p1,...,pD, q1,...,qD]
    q_axis_pos = list(range(D, 2 * D))
    r_axis_order_by_creation: List[int] = []
    
    for d in order:
        axis = q_axis_pos[d]
        A = A.movedim(axis, -1)                                           # [..., q_d]
        A = torch.tensordot(A, B[d].to(device), dims=([-1], [0]))         # [..., r_d]
        q_axis_pos = [pos - 1 if pos > axis else pos for pos in q_axis_pos]
        q_axis_pos[d] = -1
        r_axis_order_by_creation.append(d)
    
    # 2) FFT over p-axes
    Ctil = torch.fft.fftn(A, dim=tuple(range(D)))                         # [p1,...,pD, r_*]
    
    # 3) Fused sampling + r-contraction per axis
    sampled = Ctil
    current_r_order = r_axis_order_by_creation.copy()
    
    # Contract dimensions in descending (2M_d * r_d) to shrink r-stack early
    fuse_order = sorted(range(D), key=lambda d: -(idx_mod_1d[d].numel() * r_sizes[d]))
    
    for d in fuse_order:
        # locate the r-axis paired with geometric dim d
        try:
            idx_in_stack = current_r_order.index(d)
        except ValueError:
            continue  # already contracted (e.g., r_d == 0)
        r_axis_pos = D + idx_in_stack  # r-axes follow the D M-axes
    
        sampled = _contract_axis_after_gather(
            sampled=sampled,
            d=d,
            r_axis_pos=r_axis_pos,
            V_d=V_1d[d].to(sampled.device),
            idx_1d=idx_mod_1d[d].to(sampled.device),
        )
        current_r_order.pop(idx_in_stack)
    
    # Only M-axes remain: (2M1,...,2MD)
    return sampled.to(torch.complex64).contiguous()