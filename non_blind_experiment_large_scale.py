import torch
import torch.fft as fft
import torch.nn.functional as F
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import numpy as np
import matplotlib.pyplot as plt
import time
import math

torch.set_default_dtype(torch.float64)

from PIL import Image
Image.MAX_IMAGE_PIXELS = 668435456
import csv

# device = 'cuda'
device = 'cpu'
cmap = 'gray'

def resize_image(input_image_path, output_image_path, new_size):
    original_image = Image.open(input_image_path)
    width, height = original_image.size
    resized_image = original_image.resize(new_size)
    resized_image.save(output_image_path)
    
input_path1 = "mandril_gray.tif"
output_path1 = "baboon_resized.png"
input_path2 = "cameraman.tif"
output_path2 = "cameraman_resized.png"

# SET SIZE OF IMAGE HERE
nx = 16384 # 2^{14}
# nx = 512
new_size = (nx, nx)  # New size of the image (width, height)

# PFT PARAMETERS (SET M AND p VALUES HERE)
N = [nx, nx]
M = [64, 64] 
mu = [0, 0]
p = [64, 64]
error = 'e-7'

resize_image(input_path1, output_path1, new_size)
resize_image(input_path2, output_path2, new_size)

im1 = np.array(Image.open('baboon_resized.png').convert("L"))
im2 = np.array(Image.open('cameraman_resized.png').convert("L"))
# im = im.resize((64,64))
r_true = torch.DoubleTensor(im1)
r_true = r_true/torch.max(r_true)
phi_true = torch.DoubleTensor(im2)
phi_true = phi_true/torch.max(phi_true) 
# phi_true = (2 * torch.pi) * phi_true - torch.pi 
phi_true = 0.5*torch.pi*phi_true # 0 to pi/2

z_true = torch.abs(r_true) * torch.exp(1j * phi_true)
z_true = z_true.view(-1, 1).to(device)

n = z_true.shape[0]
m = n


# In[2]:


n_probes = 9
probes = torch.zeros(n_probes, nx, nx, device=device)

probes[0,0:int(nx/2),0:int(nx/2)] = 1.0
probes[1,int(nx/4):int(3*nx/4),0:int(nx/2)] = 1.0
probes[2,int(nx/2):int(nx),0:int(nx/2)] = 1.0

probes[3,0:int(nx/2),int(nx/4):int(3*nx/4)] = 1.0
probes[4,int(nx/4):int(3*nx/4),int(nx/4):int(3*nx/4)] = 1.0
probes[5,int(nx/2):int(nx),int(nx/4):int(3*nx/4)] = 1.0

probes[6,0:int(nx/2),int(nx/2):int(nx)] = 1.0
probes[7,int(nx/4):int(3*nx/4),int(nx/2):int(nx)] = 1.0
probes[8,int(nx/2):int(nx),int(nx/2):int(nx)] = 1.0


# ## PFT Functions

# In[3]:


def pft2d_configuration(N, M, mu, p, error):
    B = [0, 0]
    q = [0, 0]
    r = [0, 0]
    
    for d in range(2):
        q[d] = N[d] // p[d]

        r[d] = 0
        # Load precomputed xi
        csv_file = error + ".csv"
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

        W = torch.tensor([float(item) for item in selected_row], device=device)

        # Generate B using precomputed w
        indices_l, indices_j = torch.meshgrid(torch.arange(q[d], device=device), torch.arange(r[d], device=device), indexing='ij')  

        exponent_term = torch.exp(-2j * np.pi * mu[d] * (indices_l - q[d] / 2) / N[d])

        # Compute the (1 - 2*l/q) * 1j term
        l_values = torch.arange(q[d], device=device)
        coefficients = ((1 - 2 * l_values / q[d]) * 1j).unsqueeze(1) ** indices_j
        coefficients[coefficients.isnan()] = 1.0
    
        B_temp = exponent_term * W * coefficients

        B[d] = B_temp
    
    return B, p, q, r

def pft2d_precompute(M, mu, p, q, r):

    # Calculate m1 and m2 ranges
    m1_range = torch.arange(mu[0] - M[0], mu[1] + M[0], device=device)
    m2_range = torch.arange(mu[0] - M[1], mu[1] + M[1], device=device)

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
    m1_powers = m1_diff.unsqueeze(-1).unsqueeze(-1) ** torch.arange(r[0], device=device).view(1, 1, r[0], 1)
    m2_powers = m2_diff.unsqueeze(-1).unsqueeze(-1) ** torch.arange(r[1], device=device).view(1, 1, 1, r[1])
    
    precomputed_prod = m1_powers * exp_term_m1 * m2_powers * exp_term_m2
    
    return m1_mod, m2_mod, precomputed_prod


def pft2d_computation(X, B, m1_mod, m2_mod, precomputed_prod):
    # Compute matrix multiplication B1^T * X * B2
    # Let N = N1*N2, M = M1*M2, p = p1*p2, q = q1*q2, r = r1*r2
    # NOTE: Associativity of matmul allows us to choose parenthesization with lower cost!
    C = torch.matmul(torch.matmul(B[0].t(), X), B[1]) # O(p1p2q2r1(q1+ r2)) = O(r * N)
    
    # Use torch.fft.fft2 on appropriate dimensions
    Ctil = torch.fft.fft2(C, dim=(0, 1)) # O(r1r2 * p1p2 log p1p2) = O(r * p log p)
    
    # Perform element-wise multiplication and summation
    pft_array = torch.sum(Ctil[m1_mod, m2_mod] * precomputed_prod, dim=(2, 3)).to(device) 
    
    return pft_array

B, p, q, r = pft2d_configuration(N, M, mu, p, error)
m1_mod, m2_mod, precomputed_prod = pft2d_precompute(M, mu, p, q, r)
print(f"N, M = {N[0], N[1]}, {M[0], M[1]} // p, q, r = {p}, {q}, {r} // e = {error}")


def fftshift(x):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    dim = len(x.shape)
    shift = [dim // 2 for dim in x.shape]
    return torch.roll(x, shift, tuple(range(dim)))


# In[9]:


def ifftshift(x):
    """
    Inverse operation of fftshift for PyTorch Tensors
    """
    dim = len(x.shape)
    shift = [-dim // 2 for dim in x.shape]
    return torch.roll(x, shift, tuple(range(dim)))


# Full Data 
b = torch.zeros(n_probes, n, device=device)
for i in range(n_probes):
    z_temp = probes[i,:,:]*z_true.view(nx,nx) #Qz
    # z_temp = z_temp.view(n, 1)
    
    # Add noise
    #eps = 0.01 * torch.randn(b[i, :].shape, device=device) # * torch.abs(probes[i, :, :].view(-1))
    
    # |FQ_iz_true|^2 + noise
    b[i,:] = torch.abs(torch.fft.fft2(z_temp).view(-1)) #+ eps


# Cropped Data
b_crop = torch.zeros(n_probes, int(4*M[0]*M[1]), device=device)
for i in range(n_probes):
    Z = torch.abs(torch.fft.fft2(probes[i,:,:]*z_true.reshape(nx,nx)))

    # Perform shifting to match the FFT output layout
    Z_shifted = fftshift(Z)

    # Determine the indices corresponding to the rectangle [-M1, M1] x [-M2, M2]
    start_idx_1 = N[0] // 2 - M[0]
    end_idx_1 = N[0] // 2 + M[0]
    start_idx_2 = N[1] // 2 - M[1]
    end_idx_2 = N[1] // 2 + M[1]

    # Extract the Fourier coefficients corresponding to the rectangle
    b_crop[i,:] = ifftshift(Z_shifted[start_idx_1:end_idx_1, start_idx_2:end_idx_2]).reshape(-1)


def total_variation(x):
    # Calculate the difference in the horizontal direction
    horizontal_diff = x[:, :-1] - x[:, 1:]
    
    # Calculate the difference in the vertical direction
    vertical_diff = x[:-1, :] - x[1:, :]
    
    # Compute the L1 norm of the gradient (total variation)
    tv = torch.sum(torch.abs(horizontal_diff)) + torch.sum(torch.abs(vertical_diff))
    
    return tv


def f_original(z, b=b, probes = probes, nx=nx, return_gradient=False):
    # evaluates function
    # inputs:
    #   z     = feature vector of size (n_samples x n_features)
    #  indices = indices of size n_probes x n_samples(pixels)
    # outputs:
    #   f_val = scalar function evaluated at input features of size (n_samples x 1)
    device = z.device

    n_samples = z.shape[0]
    n = z.shape[1]

    if return_gradient==True:
        z.requires_grad = True

    # shape is now n by n_samples
    z = z.permute(1,0)

    f_val = 0.0
    
    # loop over every probe to evaluate function
    for i in range(n_probes):
        fft_z = torch.fft.fft2(probes[i] * z.view(nx, nx)) # FQz  
        proj_z = b[i].view(nx, nx) * torch.exp(1j*torch.angle(fft_z))
        
        f_val = f_val + 0.5*torch.norm(torch.fft.ifft2(proj_z - fft_z))**2

    f_val = f_val/n_probes

    if return_gradient==False:
        return f_val
    else:
        grad_f = torch.autograd.grad(f_val, z,
                                     grad_outputs=torch.ones(f_val.shape, device=device), retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False, is_grads_batched=False)[0] 
        grad_f = grad_f.detach()
        f_val = f_val.detach()
        z = z.detach()
    return f_val, grad_f


# In[14]:


def f_single_probe(z, b, probe, nx=nx, return_gradient=False, n_probes=n_probes, lmbda=1e-1):
    # evaluates function
    # inputs:
    #   z     = feature vector of size (n_samples x n_features)
    #  indices = indices of size n_probes x n_samples(pixels)
    # outputs:
    #   f_val = scalar function evaluated at input features of size (n_samples x 1)
    device = z.device

    n_samples = z.view(1,-1).shape[0]
    n = z.view(1,-1).shape[1]

    if return_gradient==True:
        z.requires_grad = True

    # shape is now n by n_samples
    # z = z.permute(1,0)

    f_val = 0.0

    fft_z = torch.fft.fft2(probe * z.view(nx, nx)) # FQz 
    proj_z = b.view(nx, nx) * torch.exp(1j*torch.angle(fft_z))
    
    f_val = f_val + 0.5*torch.norm(torch.fft.ifft2(proj_z - fft_z))**2

    # f_val = f_val/n_probes # + 0.01 * total_variation(z)
    # Add total variation
    f_val_regularized = (f_val + lmbda * total_variation(z))/n_probes
    # f_val_regularized = (f_val + lmbda * total_variation(z)) ################################################################################################################################################################

    if return_gradient==False:
        return f_val
    else:
        grad_f = torch.autograd.grad(f_val_regularized, z,
                                     grad_outputs=torch.ones(f_val_regularized.shape, device=device), retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False, is_grads_batched=False)[0] 
        grad_f = grad_f.detach()
        f_val_regularized = f_val.detach()
        z = z.detach()
    return f_val, grad_f


def f_pft(z, b, probe, lmbda = 0.1, nx=nx, return_gradient=False, B=B, M=M, p=p, q=q,
      m1_mod=m1_mod, m2_mod=m2_mod, precomputed_prod=precomputed_prod, n_probes=n_probes):
    # evaluates function
    # inputs:
    #   z     = feature vector of size (n_samples x n_features)
    #  indices = indices of size n_probes x n_samples(pixels)
    # outputs:
    #   f_val = scalar function evaluated at input features of size (n_samples x 1)
    device = z.device

    if return_gradient==True:
        z.requires_grad = True

    # shape is now n by n_samples
    # z = z.permute(1,0)

    f_val = 0.0

    z_temp = probe.view(nx, nx) * z.view(nx, nx)
    z_temp = z_temp.view(p[0], q[0], p[1], q[1]).permute(0, 2, 1, 3).contiguous().view(p[0], p[1], q[0], q[1])
    pft_z = pft2d_computation(z_temp, B, m1_mod, m2_mod, precomputed_prod)

    proj_z = b.view(2*M[0], 2*M[1]) * torch.exp(1j*torch.angle(ifftshift(pft_z)))
    
    f_val = f_val + 0.5*torch.norm(torch.fft.ifft2(proj_z - fftshift(pft_z)))**2
    
    # Add total variation
    f_val_regularized = (f_val + lmbda * total_variation(z))/n_probes
    # f_val_regularized = (f_val + lmbda * total_variation(z)) ################################################################################################################################################################

    if return_gradient==False:
        return f_val
    else:
        grad_f = torch.autograd.grad(f_val_regularized, z,
                                     grad_outputs=torch.ones(f_val_regularized.shape, device=device), retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False, is_grads_batched=False)[0] 
        grad_f = grad_f.detach()
        f_val_regularized = f_val.detach()
        z = z.detach()
    return f_val, grad_f

# Create an initial image for all algorithms to use
def random_complex_vector(size):
    # Generate magnitude and phase
    magnitude = torch.abs(torch.rand(size)).view(nx, nx)
    phase = (torch.pi / 2) * torch.rand(phi_true.shape)
    # Combine real and imaginary parts to create complex numbers
    complex_vector = magnitude * torch.exp(1j * phase)
    return complex_vector.view(-1)

guess_object = random_complex_vector(n)


# def random_complex_vector(size):
#     # Generate the real part of the complex matrix
#     real_part = torch.rand(nx, nx)
    
#     # Generate the imaginary part of the complex matrix
#     imag_part = torch.rand(nx, nx)
    
#     # Combine the real and imaginary parts into a complex matrix
#     complex_matrix = torch.complex(real_part, imag_part)
    
#     return complex_matrix.view(-1)

# guess_object = random_complex_vector(n).to(device)
# print('guess_object true constant = 1')



def interpolate_image_for_psnr(image):
    # Interpolate the image to [0,1]
    min_val = image.min()
    max_val = image.max()
    interpolated_image = (image - min_val) / (max_val - min_val)

    # Ensure the image values are in the range [0, 1]
    interpolated_image = torch.clamp(interpolated_image, 0, 1)

    return interpolated_image


# ## PIE Algorithms (with and without PFT)


def PIE(zk, alpha = 1, num_iters = 1000, b=b, probes = probes, nx=nx, tol = 1e-1, z_true=z_true, tv_lmbda=1e-1):
    
    device = zk.device
    
    f_val_hist = []
    gradf_val_hist = []
    rel_err_hist = []
    cauchy_err_hist = []
    time_hist = []
    SSIM_phase_hist = []
    SSIM_mag_hist = []
    PSNR_phase_hist = []
    PSNR_mag_hist = []
    mag_err_hist = []
    phase_err_hist = []

    mag_true = torch.Tensor(torch.abs(z_true.view(nx, nx))).cpu()
    phase_true = torch.Tensor(torch.angle(z_true.view(nx, nx))).cpu()

    rel_err = torch.norm(zk.view(n).cpu() - z_true.view(n).cpu()) / torch.norm(z_true.view(n).cpu())
    print('initial rel_err = ', rel_err.cpu())
    
    cauchy_err = 100
    j = 1
    while cauchy_err > tol:
        
        z_old = zk.clone().cpu()
        
        start = time.time()
        for i in range(n_probes):
            
            fval, gradf_val = f_single_probe(zk.view(1,-1), b = b[i], probe = probes[i], return_gradient=True) 
            zk = zk - alpha * gradf_val.view(n)
        end = time.time()
        iter_time = end - start
        time_hist.append(iter_time)

        rel_err = torch.norm(zk.view(n).cpu() - z_true.view(n).cpu()) / torch.norm(z_true.view(n).cpu())

        # start_time = time.time()
        # fval_full, gradf_val_full = f_original(zk.view(1,-1), return_gradient=True) 
        # end_time = time.time()
        # f_val_time = end_time - start_time

        start_time = time.time()
        fval_full = f_original(zk.view(1,-1), return_gradient=False) 
        end_time = time.time()
        f_val_time = end_time - start_time

        cauchy_err = torch.norm(z_old - zk.cpu())/(torch.norm(z_old))
        
        
        f_val_hist.append(fval_full.cpu())
        # gradf_val_hist.append(torch.norm(gradf_val_full).cpu())
        rel_err_hist.append(rel_err)
        cauchy_err_hist.append(cauchy_err.cpu())

        mag_k = torch.abs(zk.view(nx, nx)).cpu().detach()
        phase_k = torch.angle(zk.view(nx, nx)).cpu().detach()

        mag_err = torch.norm(mag_k - mag_true)/torch.norm(mag_true)
        phase_err = torch.norm(phase_k - phase_true)/torch.norm(phase_true)
        mag_err_hist.append(mag_err.cpu())
        phase_err_hist.append(phase_err.cpu())

        start_ssim_psnr_time = time.time()

        mag_ssim = ssim(mag_k.numpy(), r_true.cpu().detach().numpy(), data_range=r_true.cpu().detach().numpy().max()-r_true.cpu().detach().numpy().min())
        phase_ssim = ssim(phase_k.numpy(), phi_true.cpu().detach().numpy(), data_range=phi_true.cpu().detach().numpy().max()-phi_true.cpu().detach().numpy().min())

        mag_psnr = psnr(interpolate_image_for_psnr(mag_k).numpy(), interpolate_image_for_psnr(r_true).numpy())
        phase_psnr = psnr(interpolate_image_for_psnr(phase_k).numpy(), interpolate_image_for_psnr(phi_true).numpy())

        end_ssim_psnr_time = time.time()

        ssim_psnr_time = end_ssim_psnr_time - start_ssim_psnr_time

        SSIM_mag_hist.append(mag_ssim)
        SSIM_phase_hist.append(phase_ssim)
        PSNR_mag_hist.append(mag_psnr)
        PSNR_phase_hist.append(phase_psnr)
        
        print('j: ', j, 
            ' fx: ', "{:5.2e}".format(fval_full.cpu().item()),
        # ' |grad_fx|: ', "{:5.2e}".format(torch.norm(gradf_val_full).cpu().item()),
        ' cauchy_err: ', "{:5.2e}".format(cauchy_err.cpu().item()),
        ' rel_err: ', "{:5.2e}".format(rel_err),
        ' phase_ssim: ', "{:5.2e}".format(phase_ssim),
        ' mag_ssim: ', "{:5.2e}".format(mag_ssim),
        ' phase_psnr: ', "{:5.2e}".format(phase_psnr),
        ' mag_psnr: ', "{:5.2e}".format(mag_psnr),
        ' phase_err: ', "{:5.2e}".format(phase_err),
        ' mag_err: ', "{:5.2e}".format(mag_err),
        ' iter time: ', "{:5.2e}".format(iter_time),
        ' ssim_time : ', "{:5.2e}".format(ssim_psnr_time),
        ' f_full time:', "{:5.2e}".format(f_val_time)
        )

        j += 1

        if cauchy_err < tol or j>=num_iters:
            print("\n Converged at step:", j)
            print('j: ', j, 
            ' fx: ', "{:5.2e}".format(fval_full.cpu().item()),
            # ' |grad_fx|: ', "{:5.2e}".format(torch.norm(gradf_val_full).cpu().item()),
            ' |x-xold|/|xold|: ', "{:5.2e}".format(cauchy_err.cpu().item()),
            ' rel_err: ', "{:5.2e}".format(rel_err),
            ' iter time: ', "{:5.2e}".format(iter_time),
            ' f_full time:', "{:5.2e}".format(f_val_time)
            )
            break
            
    return zk.view(nx, nx), f_val_hist, gradf_val_hist, rel_err_hist, cauchy_err_hist, time_hist, SSIM_mag_hist, SSIM_phase_hist, PSNR_mag_hist, PSNR_phase_hist, phase_err_hist, mag_err_hist

def PIE_PFT(zk, alpha = 1, num_iters = 1000, b=b_crop, probes = probes, nx=nx, tol = 1.10e-1, tv_lmbda = 1e-2, z_true=z_true):
    
    device = zk.device
    
    f_val_hist = []
    gradf_val_hist = []
    rel_err_hist = []
    cauchy_err_hist = []
    time_hist = []
    SSIM_phase_hist = []
    SSIM_mag_hist = []
    PSNR_phase_hist = []
    PSNR_mag_hist = []
    mag_err_hist = []
    phase_err_hist = []
    
    cauchy_err = 100
    j = 1

    mag_true = torch.Tensor(torch.abs(z_true.view(nx, nx))).cpu()
    phase_true = torch.Tensor(torch.angle(z_true.view(nx, nx))).cpu()


    rel_err = torch.norm(zk.view(n).cpu() - z_true.view(n).cpu()) / torch.norm(z_true.view(n).cpu())
    print('initial rel_err = ', rel_err.cpu())

    while cauchy_err > tol:

        start_time_total = time.time()
        
        z_old = zk.clone().cpu()
        
        start = time.time()
        for i in range(n_probes):
            
            fval, gradf_val = f_pft(zk.view(1,-1), b = b[i], probe = probes[i], lmbda=tv_lmbda, return_gradient=True)  
            zk = zk - alpha * gradf_val.view(n)
        end = time.time()
        iter_time = end - start
        time_hist.append(iter_time)

        rel_err = torch.norm(zk.view(n).cpu() - z_true.view(n).cpu()) / torch.norm(z_true.view(n).cpu())

        # start_time = time.time()
        # fval_full, gradf_val_full = f_original(zk.view(1,-1), return_gradient=True) 
        # end_time = time.time()
        # f_val_time = end_time - start_time

        start_time = time.time()
        fval_full = f_original(zk.view(1,-1), return_gradient=False) 
        end_time = time.time()
        f_val_time = end_time - start_time
        
        
        cauchy_err = torch.norm(z_old - zk.cpu())/(torch.norm(z_old))

        f_val_hist.append(fval_full.cpu())
        # gradf_val_hist.append(torch.norm(gradf_val_full).cpu())
        rel_err_hist.append(rel_err)
        cauchy_err_hist.append(cauchy_err.cpu())

        mag_k = torch.abs(zk.view(nx, nx)).cpu().detach()
        # mag_k_torch = torch.abs(zk.view(nx, nx)).cpu()
        phase_k = torch.angle(zk.view(nx, nx)).cpu().detach()
        # phase_k_torch = torch.angle(zk.view(nx, nx)).cpu()

        mag_err = torch.norm(mag_k - mag_true)/torch.norm(mag_true)
        phase_err = torch.norm(phase_k - phase_true)/torch.norm(phase_true)
        mag_err_hist.append(mag_err.cpu())
        phase_err_hist.append(phase_err.cpu())

        start_ssim_psnr_time = time.time()

        # mag_ssim = 0.0
        # phase_ssim = 0.0
        # mag_psnr = 0.0
        # phase_psnr = 0.0

        mag_ssim = ssim(mag_k.numpy(), r_true.cpu().detach().numpy(), data_range=r_true.cpu().detach().numpy().max()-r_true.cpu().detach().numpy().min())
        phase_ssim = ssim(phase_k.numpy(), phi_true.cpu().detach().numpy(), data_range=phi_true.cpu().detach().numpy().max()-phi_true.cpu().detach().numpy().min())

        mag_psnr = psnr(interpolate_image_for_psnr(mag_k).numpy(), interpolate_image_for_psnr(r_true).numpy())
        phase_psnr = psnr(interpolate_image_for_psnr(phase_k).numpy(), interpolate_image_for_psnr(phi_true).numpy())

        end_ssim_psnr_time = time.time()

        ssim_psnr_time = end_ssim_psnr_time - start_ssim_psnr_time

        SSIM_mag_hist.append(mag_ssim)
        SSIM_phase_hist.append(phase_ssim)
        PSNR_mag_hist.append(mag_psnr)
        PSNR_phase_hist.append(phase_psnr)
        
        print('j: ', j, 
            ' fx: ', "{:5.2e}".format(fval_full.cpu().item()),
        # ' |grad_fx|: ', "{:5.2e}".format(torch.norm(gradf_val_full).cpu().item()),
        ' cauchy_err: ', "{:5.2e}".format(cauchy_err.cpu().item()),
        ' rel_err: ', "{:5.2e}".format(rel_err),
        ' phase_ssim: ', "{:5.2e}".format(phase_ssim),
        ' mag_ssim: ', "{:5.2e}".format(mag_ssim),
        ' phase_psnr: ', "{:5.2e}".format(phase_psnr),
        ' mag_psnr: ', "{:5.2e}".format(mag_psnr),
        ' phase_err: ', "{:5.2e}".format(phase_err),
        ' mag_err: ', "{:5.2e}".format(mag_err),
        ' iter time: ', "{:5.2e}".format(iter_time),
        ' ssim_time : ', "{:5.2e}".format(ssim_psnr_time),
        ' f_full time:', "{:5.2e}".format(f_val_time)
        )

        j += 1
        

        if cauchy_err < tol or j>=num_iters:
            print("\n Converged at step:", j)
            print('j: ', j, 
            ' fx: ', "{:5.2e}".format(fval_full.cpu().item()),
            # ' |grad_fx|: ', "{:5.2e}".format(torch.norm(gradf_val_full).cpu().item()),
            ' |x-xold|/|xold|: ', "{:5.2e}".format(cauchy_err.cpu().item()),
            ' rel_err: ', "{:5.2e}".format(rel_err),
            ' iter time: ', "{:5.2e}".format(iter_time),
            ' f_full time:', "{:5.2e}".format(f_val_time)
            )
            break
            
    return zk.view(nx, nx), f_val_hist, gradf_val_hist, rel_err_hist, cauchy_err_hist, time_hist, SSIM_mag_hist, SSIM_phase_hist, PSNR_mag_hist, PSNR_phase_hist, phase_err_hist, mag_err_hist


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 
# EXPERIMENTS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

print('-------------------------------------------------------- STARTING FULL PIE LARGE INIT pi/2 ----------------------------------------------------')

# alpha = 1e-1
alpha = 1.0
num_iters= int(5e2)
tol = 5e-4
tv_lmbda = 1e-1

print('alpha = ', alpha, ', num_iters = ', num_iters, ', tol = ', tol, 'tv_lmbda = ', tv_lmbda)

''' PIE With Full FFT '''
z_guess = guess_object.view(nx, nx).to(device)
z_optPIE, f_val_hist, gradf_val_hist, rel_err_hist, cauchy_err_hist, time_hist, SSIM_mag_hist, SSIM_phase_hist, PSNR_mag_hist, PSNR_phase_hist, phase_err_hist, mag_err_hist  = PIE(z_guess.view(n), alpha = alpha, num_iters = num_iters, tol=tol, tv_lmbda=tv_lmbda) 


fig = plt.figure()
plt.imshow(z_optPIE.imag.cpu(), cmap=cmap)
plt.colorbar()


save_str = 'results/full_reconstruction.png'
fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


fig = plt.figure()
r_optPIE = torch.abs(z_optPIE)
plt.imshow(r_optPIE.cpu(), vmin=0, vmax=1, cmap=cmap)
plt.colorbar()


save_str = 'results/full_magnitude.png'
fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


fig = plt.figure()
phi_optPIE = torch.angle(z_optPIE)
plt.imshow(phi_optPIE.cpu(), cmap=cmap, vmin=0, vmax=np.pi/2)
plt.colorbar()


save_str = 'results/full_phase.png'
fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


file_name = 'results/full_PIE_hist.pt'
state = {
    'z_optPIE': z_optPIE,
    'f_val_hist': f_val_hist,
    'grad_norm_hist': gradf_val_hist,
    'rel_err_hist': rel_err_hist,
    'cauchy_err_hist': cauchy_err_hist,
    'SSIM_phase_hist': SSIM_phase_hist,
    'SSIM_mag_hist': SSIM_mag_hist,
    'PSNR_phase_hist': PSNR_phase_hist,
    'PSNR_mag_hist': PSNR_mag_hist,
    'phase_err_hist': phase_err_hist,
    'mag_err_hist': mag_err_hist,
    'time_hist': time_hist,
    'alpha': alpha,
    'tol': tol,
    'num_iters': num_iters
}
torch.save(state, file_name)
print('files saved to ' + file_name)

del z_optPIE


print(' SAVING TRUE IMAGES')
fig = plt.figure()
plt.imshow(z_true.view(nx, nx).imag.cpu(), cmap=cmap)
plt.colorbar()


save_str = 'results/true.png'
fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


fig = plt.figure()
plt.imshow(r_true.cpu(), vmin=0, vmax=1, cmap=cmap)
plt.colorbar()


save_str = 'results/true_magnitude.png'
fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


fig = plt.figure()
plt.imshow(phi_true.cpu(), cmap=cmap, vmin=0, vmax=np.pi/2)
plt.colorbar()


save_str = 'results/true_phase.png'
fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)



print('-------------------------------------------------------- STARTING PFT PIE ----------------------------------------------------')

# alpha_PFT = 1e-1 # for nx=16k
alpha_PFT = 1e-3
num_iters_PFT= int(2e1)
tol_PFT = 1e-2
tv_lmbda= 100.0

z_guess = guess_object.view(nx, nx).to(device)

print('alpha_PFT = ', alpha_PFT, ', num_iters_PFT = ', num_iters_PFT, ', tol_PFT = ', tol_PFT, ', tv_lmbda = ', tv_lmbda)

z_PFT, f_val_hist_PFT, gradf_val_hist_PFT, rel_err_hist_PFT, cauchy_err_hist_PFT, time_hist_PFT, SSIM_mag_hist_PFT, SSIM_phase_hist_PFT, PSNR_mag_hist_PFT, PSNR_phase_hist_PFT, phase_err_hist_PFT, mag_err_hist_PFT = PIE_PFT(z_guess.view(n), alpha = alpha_PFT, num_iters = num_iters_PFT, tol=tol_PFT, tv_lmbda=tv_lmbda) 


fig = plt.figure()
plt.imshow(z_PFT.imag.cpu(), cmap=cmap)
plt.colorbar()


save_str = 'results/pft_reconstruction.png'
fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


fig = plt.figure()
r_optPIE_PFT = torch.abs(z_PFT)
plt.imshow(r_optPIE_PFT.cpu(), vmin=0, vmax=1, cmap=cmap)
plt.colorbar()


save_str = 'results/pft_magnitude.png'
fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


fig = plt.figure()
phi_optPIE_PFT = torch.angle(z_PFT)
plt.imshow(phi_optPIE_PFT.cpu(), cmap=cmap, vmin=0, vmax=np.pi/2)
plt.colorbar()


save_str = 'results/pft_phase.png'
fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


file_name = 'results/pft_PIE_hist.pt'
state = {
    'z_PFT': z_PFT,
    'f_val_hist_PFT': f_val_hist_PFT,
    'grad_norm_hist_PFT': gradf_val_hist_PFT,
    'rel_err_hist_PFT': rel_err_hist_PFT,
    'cauchy_err_hist_PFT': cauchy_err_hist_PFT,
    'SSIM_phase_hist_PFT': SSIM_phase_hist_PFT,
    'SSIM_mag_hist_PFT': SSIM_mag_hist_PFT,
    'PSNR_phase_hist_PFT': PSNR_phase_hist_PFT,
    'PSNR_mag_hist_PFT': PSNR_mag_hist_PFT,
    'phase_err_hist_PFT': phase_err_hist_PFT,
    'mag_err_hist_PFT': mag_err_hist_PFT,
    'time_hist_PFT': time_hist_PFT,
    'alpha_PFT': alpha_PFT,
    'tol_PFT': tol_PFT,
    'tv_lmbda': tv_lmbda
}
torch.save(state, file_name)
print('files saved to ' + file_name)


print('-------------------------------------------------------- STARTING Warmstarted PIE ----------------------------------------------------')

''' Hybrid PIE '''
# z_guess = guess_object.to(device)

alpha = alpha
num_iters = num_iters
tol = tol
print('alpha = ', alpha, ', num_iters = ', num_iters, ', tol = ', tol)

''' PIE With Full FFT '''
# z_guess = guess_object.view(nx, nx).to(device)
z_optHybrid, f_val_hist, gradf_val_hist, rel_err_hist, cauchy_err_hist, time_hist, SSIM_mag_hist, SSIM_phase_hist, PSNR_mag_hist, PSNR_phase_hist, phase_err_hist, mag_err_hist = PIE(z_PFT.view(n), alpha = alpha, num_iters = num_iters, tol=tol) 

fig = plt.figure()
plt.imshow(z_optHybrid.imag.cpu(), cmap=cmap)
plt.colorbar()


save_str = 'results/hybrid_reconstruction.png'
fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


fig = plt.figure()
r_optHybrid = torch.abs(z_optHybrid)
plt.imshow(r_optHybrid.cpu(), vmin=0, vmax=1, cmap=cmap)
plt.colorbar()


save_str = 'results/hybrid_magnitude.png'
fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


fig = plt.figure()
phi_optHybrid = torch.angle(z_optHybrid)
plt.imshow(phi_optHybrid.cpu(), cmap=cmap, vmin=0, vmax=np.pi/2)
plt.colorbar()


save_str = 'results/hybrid_phase.png'
fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


file_name = 'results/hybrid_hist.pt'
state = {
    'z_optHybrid': z_optHybrid,
    'f_val_hist': f_val_hist,
    'grad_norm_hist': gradf_val_hist,
    'rel_err_hist': rel_err_hist,
    'cauchy_err_hist': cauchy_err_hist,
    'SSIM_phase_hist': SSIM_phase_hist,
    'SSIM_mag_hist': SSIM_mag_hist,
    'PSNR_phase_hist': PSNR_phase_hist,
    'PSNR_mag_hist': PSNR_mag_hist,
    'phase_err_hist': phase_err_hist,
    'mag_err_hist': mag_err_hist,
    'time_hist': time_hist,
    'alpha': alpha,
    'tol': tol,
    'num_iters': num_iters
}
torch.save(state, file_name)
print('files saved to ' + file_name)

