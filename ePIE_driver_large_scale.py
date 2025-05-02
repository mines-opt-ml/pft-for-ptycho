#!/usr/bin/env python
# coding: utf-8

import torch
import torch.fft as fft
from torchvision import transforms

from skimage.feature import match_template
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import numpy as np
import matplotlib.pyplot as plt
import math
import time

import torch.nn.functional as F

torch.set_default_dtype(torch.float64)

# set random seed
torch.manual_seed(42)

import PIL
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = 668435456 
import csv

device = 'cpu'
cmap = 'gray'

def resize_image(input_image_path, output_image_path, new_size):
    original_image = Image.open(input_image_path)
    width, height = original_image.size
    resized_image = original_image.resize(new_size)
    resized_image.save(output_image_path)

# Example usage
input_path1 = "mandril_gray.tif"
output_path1 = "baboon_resized.png"
input_path2 = "cameraman.tif"
output_path2 = "cameraman_resized.png"

# SET SIZE OF IMAGE HERE
# nx = 16384
# nx = int(2**13) ####################
# nx = 512 ###
# nx = 1024
# nx = 4096
nx = 8200 # so that it is divisible by 64
print('padding 1500')
print('nx = ', nx)
# nx = int(2**11) ####################
new_size = (nx, nx)  # New size of the image (width, height)

resize_image(input_path1, output_path1, new_size)
resize_image(input_path2, output_path2, new_size)

# Load and preprocess images
im1 = np.array(Image.open(output_path1).convert("L"))
im2 = np.array(Image.open(output_path2).convert("L"))

print('nx = ', nx)
if nx == 512:
    im1 = np.pad(im1, ((128, 128), (128, 128)), mode='constant', constant_values=0)
    im2 = np.pad(im2, ((128, 128), (128, 128)), mode='constant', constant_values=0) # nx=512
elif nx > 8000:
    im1 = np.pad(im1, ((1500, 1500), (1500, 1500)), mode='constant', constant_values=0)
    im2 = np.pad(im2, ((1500, 1500), (1500, 1500)), mode='constant', constant_values=0)
else: 
    print('ERROR ON PADDING!')

# Convert images to tensors and normalize
r_true = torch.DoubleTensor(im1)
r_true = r_true/torch.max(r_true)
phi_true = torch.DoubleTensor(im2)
phi_true = phi_true/torch.max(phi_true) 
phi_true = (torch.pi / 2) * phi_true  # Adjusted to range between 0 and pi/2 

z_true = torch.abs(r_true) * torch.exp(1j * phi_true)
print('shape of z_true = ', z_true.shape)
nx_padded = z_true.shape[0] # Size of the padded image
z_true = z_true.view(-1, 1).to(device)

# PFT PARAMETERS (SET M AND p VALUES HERE)
# NOTE: The PFT will act on the padded image
N = [nx_padded, nx_padded]
# M = [64, 64] 
# M = [256, 256]
# M = [512, 512]
# M = [1024, 1024]
# M = [16, 16]
M = [64, 64]
mu = [0, 0]
# p = [128,128]
# p = [2048, 2048]
p = [64, 64]
error = 'e-7'



def create_true_probe(imin, radius=50, sigma=1e6, device='cpu'):
    """
    Creates a true probe for an image using Gaussian apertures and a grid pattern
    """
    
    # Check if the image is colored (i.e. 3 channels?) or gray-scale
    if imin.dim() > 2:
        imy, imx, imz = imin.size()
    else:
        imy, imx = imin.size()

    # Find the center of the image
    center_x = imx // 2
    center_y = imy // 2

    rr, cc = torch.meshgrid(torch.arange(1, imy+1), torch.arange(1, imy+1), indexing='ij')
    
    modulus = torch.sqrt((rr - center_y)**2 + (cc - center_x)**2)
    gauss_app = torch.exp(-modulus**2 / (2 * sigma**2))
    gauss_app = gauss_app / torch.max(gauss_app)

    if imin.dim() > 2:
        app_lim = (torch.sqrt((rr - center_y) ** 2 + (cc - center_x) ** 2) <= radius).unsqueeze(2).expand(imy, imy, imz)
        apertures = app_lim * gauss_app.unsqueeze(2).expand(imy, imy, imz)
    else:
        app_lim = (torch.sqrt((rr - center_y) ** 2 + (cc - center_x) ** 2) <= radius)
        apertures = torch.complex(app_lim * gauss_app, torch.zeros_like(app_lim * gauss_app)).to(torch.complex128)

    return apertures.to(device), app_lim.to(device), torch.tensor([center_x]), torch.tensor([center_y]), 1


# Compute radius and spacing based on the dimensions of a 512x512 image
# These values can be adjusted as needed
radius = math.ceil((175 * nx_padded) / 768)
spacing = math.ceil((150 * nx_padded) / 768)

true_probe, _, centerx_true, centery_true, n_probes = create_true_probe(z_true.view(nx_padded, nx_padded), radius=radius, sigma=1e6)

# Display the true probe
# Gaussian with sigma = 1e6 (i.e. all values close to 1)
# plt.imshow(true_probe.real.cpu(), cmap=cmap)
# plt.colorbar()
# plt.show()


def create_probes(imin, spacing=50, radius=50, sigma=1e6, string='grid', dither=3, xedge=120, yedge=120, device=device):
    """
    Creates set of true probes at different positions of the image
    """
    
    # Check if the image is colored (i.e. 3 channels?) or gray-scale
    if imin.dim() > 2:
        imy, imx, imz = imin.size()
    else:
        imy, imx = imin.size()
            
    xfar = (imx - xedge) + 2
    yfar = (imy - yedge) + 2
    
    half_size = math.ceil(imy/2)
    diameter = 2 * radius
    
    # CAUTION: Due to MATLAB indexing, everything may be off by 1 (including count)
    
    if string == 'grid':
        rr, cc = torch.meshgrid(torch.arange(1, imy+1), torch.arange(1, imy+1), indexing='ij') 
        
        r_lin = rr.reshape(1, -1)
        c_lin = cc.reshape(1, -1)
        r_lin[r_lin < xedge] = -1
        r_lin[r_lin > xfar] = -1
        c_lin[c_lin < yedge] = -1
        c_lin[c_lin > yfar] = -1
        
        center_list = torch.cat([r_lin, c_lin], dim = 0)
        count = 0
        
        centerx = torch.empty(0)
        centery = torch.empty(0)
        app_lim_list = []
        apertures_list = []
        
        for i in range(center_list.size(1)):
            
            if(
                center_list[0, i] % spacing == 0
                and center_list[1, i] % spacing == 0
                and center_list[0, i] != -1
                and center_list[1, i] != -1
            ):
                ditherx = torch.round(torch.rand(1) * dither - (dither / 2))
                dithery = torch.round(torch.rand(1) * dither - (dither / 2))
                # ditherx = torch.zeros(1)
                # dithery = torch.zeros(1)
                centerx = torch.cat([centerx, center_list[0, i] + ditherx])
                centery = torch.cat([centery, center_list[1, i] + dithery])
                
                modulus = torch.sqrt((rr - centerx[count])**2 + (cc - centery[count])**2)
                gauss_app = torch.exp(-modulus**2 / (2 * sigma** 2))
                gauss_app = gauss_app / torch.max(gauss_app)
                
                if imin.dim() > 2:
                    app_lim = torch.cat([app_lim, (torch.sqrt((rr - centerx[count]) ** 2 + (cc - centery[count]) ** 2) <= radius).unsqueeze(2).expand(imy, imy, imz)])
                    apertures = torch.cat([apertures, app_lim[count] * gauss_app.unsqueeze(2).expand(imy, imy, imz)])
                else:
                    app_lim = (torch.sqrt((rr - centerx[count]) ** 2 + (cc - centery[count]) ** 2) <= radius)
                    apertures = torch.complex(app_lim * gauss_app, torch.zeros_like(app_lim * gauss_app)).to(torch.complex128)
                
                app_lim_list.append(app_lim)
                apertures_list.append(apertures)
                count += 1
        
        app_lim = torch.stack(app_lim_list)
        apertures = torch.stack(apertures_list)
        
        return apertures.to(device), app_lim.to(device), centerx, centery, count
    
    elif string == 'square':
        rr, cc = torch.meshgrid(torch.arange(1, imy+1), torch.arange(1, imy+1), indexing='ij') 
        
        r_lin = rr.reshape(1, -1)
        c_lin = cc.reshape(1, -1)
        r_lin[r_lin < xedge] = -1
        r_lin[r_lin > xfar] = -1
        c_lin[c_lin < yedge] = -1
        c_lin[c_lin > yfar] = -1
        
        center_list = torch.cat([r_lin, c_lin], dim = 0)
        count = 0
        
        centerx = torch.empty(0)
        centery = torch.empty(0)
        app_lim_list = []
        apertures_list = []
        
        for i in range(center_list.size(1)):
            
            if(
                center_list[0, i] % spacing == 0
                and center_list[1, i] % spacing == 0
                and center_list[0, i] != -1
                and center_list[1, i] != -1
            ):
                ditherx = torch.round(torch.rand(1) * dither - (dither / 2))
                dithery = torch.round(torch.rand(1) * dither - (dither / 2))
                centerx = torch.cat([centerx, center_list[0, i] + ditherx])
                centery = torch.cat([centery, center_list[1, i] + dithery])
                
                modulus = torch.sqrt((rr - centerx[count])**2 + (cc - centery[count])**2)
                gauss_app = torch.exp(-modulus**2 / (2 * sigma**2))
                gauss_app = gauss_app / torch.max(gauss_app)
                
                app_holder = torch.zeros(imx, imy)
                app_holder[
                    centery[count].int() - radius : centery[count].int() + radius,
                    centerx[count].int() - radius : centerx[count].int() + radius
                ] = 1
                
                if imin.dim() > 2:
                    app_lim = torch.stack([app_holder.unsqueeze(2).expand(imy, imy, imz)] * count)
                    apertures = torch.stack([app_lim[count] * gauss_app.unsqueeze(2).expand(imy, imy, imz)] * count)
                else:
                    app_lim = app_holder # may need to change this
                    apertures = app_lim * gauss_app
                
                app_lim_list.append(app_lim)
                apertures_list.append(apertures)
                count += 1
            
        app_lim = torch.stack(app_lim_list)
        apertures = torch.stack(apertures_list)
        
        return apertures.to(device), app_lim.to(device), centerx, centery, count


start_time_create_probes = time.time()

# check if probes file exists
import os

file_name = 'probes_' + str(nx) + '.pt'
if os.path.exists(file_name):

    print('loading probes...')

    probes_dict = torch.load(file_name)
    probes = probes_dict['probes']
    centerx = probes_dict['centerx']
    centery = probes_dict['centery']
    n_probes = probes_dict['n_probes']
else:
    print('creating probes...')

    if nx == 512:
        start_time_create_probes = time.time()
        probes, _, centerx, centery, n_probes = create_probes(z_true.view(nx_padded, nx_padded), spacing=spacing, radius=radius,
                                      sigma=1e6, string='grid', dither=3, xedge=120, yedge=120) # nx=512
        end_time_create_probes = time.time()

        time_create_probes = end_time_create_probes - start_time_create_probes
        print('Time to create probes:', time_create_probes)
    elif nx > 8000:

        start_time_create_probes = time.time()
        probes, _, centerx, centery, n_probes = create_probes(z_true.view(nx_padded, nx_padded), spacing=spacing, radius=radius,
                                      sigma=1e6, string='grid', dither=3, xedge=950, yedge=950)
        end_time_create_probes = time.time()

        time_create_probes = end_time_create_probes - start_time_create_probes
        print('Time to create probes:', time_create_probes)

    else:

        print('ERROR with nx!')

    probes_dict = {'probes': probes, 'centerx': centerx, 'centery': centery, 'n_probes': n_probes}
    torch.save(probes_dict, file_name)


# print('creating probes')
# probes, _, centerx, centery, n_probes = create_probes(z_true.view(nx_padded, nx_padded), spacing=spacing, radius=radius,
#                                       sigma=1e6, string='grid', dither=3, xedge=120, yedge=120) # nx=512
# probes, _, centerx, centery, n_probes = create_probes(z_true.view(nx_padded, nx_padded), spacing=spacing, radius=radius,
#                                       sigma=1e6, string='grid', dither=3, xedge=250, yedge=250) # nx=2**13
# end_time_create_probes = time.time()

# time_create_probes = end_time_create_probes - start_time_create_probes

# There should be 16 probes
print('Number of Probes:', n_probes)

# # Should display probe centers evenly on image
# plt.scatter(centerx, centery)
# plt.imshow(z_true.view(nx_padded, nx_padded).cpu().imag, cmap=cmap)
# plt.show()

# # Display each probe applied to the image
# for i in range(n_probes):
#     fig1 = plt.figure()
#     current_img =  probes[i,:,:]*z_true.view(nx_padded,nx_padded)
#     plt.scatter(centery[i], centerx[i])
#     plt.imshow(current_img.real.cpu(), cmap=cmap)
#     plt.show()

# # Compute percentage of overlap between probes
# def calculate_overlap_percentage(probe1, probe2):
#     overlap = torch.sum(probe1 * probe2)
#     total_area = torch.sum(probe1)
#     return (overlap / total_area).item() * 100.0

# overlap_percentages = torch.zeros(n_probes, device=device)

# for i in range(n_probes - 1):
#     overlap_percentage = calculate_overlap_percentage(probes[i], probes[i + 1])
#     overlap_percentages[i] = overlap_percentage

# # Should be roughly 50% overlap
# for i in range(n_probes - 1):
#     print(f"Overlap percentage between probe {i} and probe {i + 1}: {overlap_percentages[i]:.2f}%")



# ## PFT Functions

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
    """
    Performs 2D partial Fourier transform (PFT) computation on the input matrix X
    """
    # Step 1: Compute matrix multiplication B1^T * X * B2
    # Using associativity to minimize computational cost
    C = torch.matmul(torch.matmul(B[0].t(), X), B[1]) # O(p1p2q2r1(q1+ r2)) = O(r * N)
    
    # Step 2: Compute the 2D FFT on the resulting matrix
    Ctil = torch.fft.fft2(C, dim=(0, 1)) # O(r1r2 * p1p2 log p1p2) = O(r * p log p)
    
    # Step 3: Perform element-wise multiplication and summation
    pft_array = torch.sum(Ctil[m1_mod, m2_mod] * precomputed_prod, dim=(2, 3)).to(device) 
    
    return pft_array


# Perform all precomputes
B, p, q, r = pft2d_configuration(N, M, mu, p, error)
m1_mod, m2_mod, precomputed_prod = pft2d_precompute(M, mu, p, q, r)
print(f"N, M = {N[0], N[1]}, {M[0], M[1]} // p, q, r = {p}, {q}, {r} // e = {error}")



# ## Create Observed Data

def fftshift(x):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    dim = len(x.shape)
    shift = [dim // 2 for dim in x.shape]
    return torch.roll(x, shift, tuple(range(dim)))

def ifftshift(x):
    """
    Inverse operation of fftshift for PyTorch Tensors
    """
    dim = len(x.shape)
    shift = [-dim // 2 for dim in x.shape]
    return torch.roll(x, shift, tuple(range(dim)))

# Full Data
b = torch.zeros(n_probes, nx_padded, nx_padded, device=device)
for i in range(n_probes):
    z_temp = probes[i,:,:]*z_true.view(nx_padded,nx_padded) #Qz
    
    # |FQ_iz_true|^2 + noise
    b[i,:, :] = torch.abs(torch.fft.fft2(z_temp))

    # fig1 = plt.figure()
    # plt.imshow(b[i].cpu(), cmap=cmap)
    # plt.show()
    
    # Ensure there are no negative values in b (just in case)
    # b[i, :] = torch.clamp(b[i, :], min=0)

# Cropped Data
b_crop = torch.zeros(n_probes, int(4*M[0]*M[1]), device=device)
for i in range(n_probes):
    Z = torch.abs(torch.fft.fft2(probes[i,:,:]*z_true.reshape(nx_padded,nx_padded)))

    # Perform shifting to match the FFT output layout
    Z_shifted = fftshift(Z)

    # Determine the indices corresponding to the rectangle [-M1, M1] x [-M2, M2]
    start_idx_1 = N[0] // 2 - M[0]
    end_idx_1 = N[0] // 2 + M[0]
    start_idx_2 = N[1] // 2 - M[1]
    end_idx_2 = N[1] // 2 + M[1]

    # Extract the Fourier coefficients corresponding to the rectangle
    b_crop[i,:] = ifftshift(Z_shifted[start_idx_1:end_idx_1, start_idx_2:end_idx_2]).reshape(-1)
    
    # fig1 = plt.figure()
    # plt.imshow(b_crop[i].view(2*M[0], 2*M[1]).cpu(), cmap=cmap)
    # plt.show()
    
    # Ensure there are no negative values in b (just in case)
    # b[i, :] = torch.clamp(b[i, :], min=0)



# ## Evaluation Functions

def total_variation(x):
    # Calculate the difference in the horizontal direction
    horizontal_diff = x[:, :-1] - x[:, 1:]
    
    # Calculate the difference in the vertical direction
    vertical_diff = x[:-1, :] - x[1:, :]
    
    # Compute the L1 norm of the gradient (total variation)
    tv = torch.sum(torch.abs(horizontal_diff)) + torch.sum(torch.abs(vertical_diff))
    
    return tv


def f_original(z, Q, b=b, nx=nx_padded, return_gradient=False, lmbda=1e-1):
    """
    Compute the full objective function value and its gradients with respect to obj_n (i.e. a section of the image z) and Q
    """
    
    device = z.device

    if return_gradient==True:
        z.requires_grad = True
        Q.requires_grad = True

    f_val = 0.0

    # Iterate over each probe
    for i in range(n_probes):
        obj_n = z[int(Y1[i]) : int(Y2[i]), int(X1[i]) : int(X2[i])].clone()
        fft_zn = torch.fft.fft2(Q*obj_n)
        proj_zn = b[i].view(nx, nx) * torch.exp(1j*torch.angle(fft_zn))
    
        f_val = f_val + 0.5*torch.norm(torch.fft.ifft2(proj_zn - fft_zn))**2

    f_val_regularized = (f_val + lmbda * total_variation(z))/n_probes

    if return_gradient==False:
        z.requires_grad = False
        Q.requires_grad = False
        
        return f_val/n_probes
    else:
        grad_f = torch.autograd.grad(outputs = f_val_regularized, inputs = [obj_n, Q],
                                     grad_outputs=torch.ones(f_val_regularized.shape, device=device), retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False, is_grads_batched=False)
        grad_z = grad_f[0].detach()
        grad_Q = grad_f[1].detach()
        f_val = f_val.detach()

        z.requires_grad = False
        Q.requires_grad = False

        return f_val/n_probes, grad_z, grad_Q


def f_single_probe(z, Q, current_ap, lmbda = 0, b=b, nx=nx_padded, return_gradient=False):
    """
    Compute the objective function value and its gradients for a single probe
    The input z is assumed to be the section of the image we are applying the probe to
    """
    
    device = z.device

    if return_gradient==True:
        z.requires_grad = True
        Q.requires_grad = True

    f_val = 0.0

    fft_z = torch.fft.fft2(Q*z)
    proj_z = b[current_ap, :, :].view(nx, nx) * torch.exp(1j*torch.angle(fft_z))

    f_val = f_val + 0.5*torch.norm(torch.fft.ifft2(proj_z - fft_z))**2

    # if lmbda > 0:
    f_val_regularized = (f_val + lmbda * total_variation(z))/n_probes

    if return_gradient==False:
        z.requires_grad = False
        Q.requires_grad = False
        
        return f_val/n_probes
    else:
        grad_f = torch.autograd.grad(outputs = f_val_regularized, inputs = [z, Q],
                                     grad_outputs=torch.ones(f_val.shape, device=device), retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False, is_grads_batched=False)
        grad_z = grad_f[0].detach()
        grad_Q = grad_f[1].detach()
        f_val = f_val.detach()
        f_val_regularized = f_val_regularized.detach()

        z.requires_grad = False
        Q.requires_grad = False

        return f_val/n_probes, grad_z, grad_Q


def f_pft(z, Q, current_ap, lmbda=0.005, b=b_crop, nx=nx_padded, return_gradient=False, B=B, M=M, p=p, q=q,
      m1_mod=m1_mod, m2_mod=m2_mod, precomputed_prod=precomputed_prod):
    """
    Compute the objective function value and its gradients for a single probe using the PFT
    The input z is assumed to be the section of the image we are applying the probe to
    """
    
    device = z.device

    if return_gradient==True:
        z.requires_grad = True
        Q.requires_grad = True

    f_val = 0.0

    z_temp = Q*z
    z_temp = z_temp.view(p[0], q[0], p[1], q[1]).permute(0, 2, 1, 3).contiguous().view(p[0], p[1], q[0], q[1])
    pft_z = pft2d_computation(z_temp, B, m1_mod, m2_mod, precomputed_prod)

    proj_z = b[current_ap].view(2*M[0], 2*M[1]) * torch.exp(1j*torch.angle(ifftshift(pft_z)))
    
    f_val = f_val + 0.5*torch.norm(torch.fft.ifft2(proj_z - fftshift(pft_z)))**2
    
    # if lmbda > 0:
    f_val_regularized = (f_val + lmbda * total_variation(z))/n_probes

    if return_gradient==False:
        z.requires_grad = False
        Q.requires_grad = False
        
        return f_val/n_probes
    else:
        grad_f = torch.autograd.grad(outputs = f_val_regularized, inputs = [z, Q],
                                     grad_outputs=torch.ones(f_val_regularized.shape, device=device), retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False, is_grads_batched=False)
        grad_z = grad_f[0].detach()
        grad_Q = grad_f[1].detach()
        f_val = f_val.detach()
        f_val_regularized = f_val_regularized.detach()

        z.requires_grad = False
        Q.requires_grad = False

        return f_val/n_probes, grad_z, grad_Q



# ## ePIE Setup

def convert_to_pixel_positions_testing(positions, little_area = nx_padded, pixel_size = 1):
    pixelPositions = positions / pixel_size
    pixelPositions[:, 0] = (pixelPositions[:, 0] - torch.min(pixelPositions[:, 0])) # x goes from 0 to max
    pixelPositions[:, 1] = (pixelPositions[:, 1] - torch.min(pixelPositions[:, 1])) # y goes from 0 to max
    pixelPositions[:, 0] = (pixelPositions[:,0] - torch.round(torch.max(pixelPositions[:,0])/2)) # x is centrosymmetric around 0
    pixelPositions[:, 1] = (pixelPositions[:,1] - torch.round(torch.max(pixelPositions[:,1])/2)) # y is centrosymmetric around 0

    bigx = little_area + torch.round(torch.max(pixelPositions[:]))*2+10 # Field of view for full object
    bigy = little_area + torch.round(torch.max(pixelPositions[:]))*2+10

    big_cent = np.floor(bigx/2) + 1 # Matlab may or may not need integer division // (Same for all other places where floor is used)

    pixelPositions = pixelPositions + big_cent

    return pixelPositions, bigx.item(), bigy.item()

def makeCircleMask(radius, imgSize = nx_padded):
    nc = imgSize//2 + 1
    n2 = nc - 1
    xx, yy = torch.meshgrid(torch.arange(-n2, n2), torch.arange(-n2, n2), indexing='ij') 
    R = torch.sqrt(xx**2 + yy**2)

    mask = (R <= radius).float()
    
    complex_mask = torch.complex(mask, torch.zeros_like(mask)).to(torch.complex128)

    return complex_mask


aperture_radius = radius 
positions = torch.stack((centerx, centery), dim = 1)

# Get center positions for cropping (should be a 2 by n vector)
pixelPositions, bigx, bigy = convert_to_pixel_positions_testing(positions)
centerx = torch.round(pixelPositions[:, 1])
centery = torch.round(pixelPositions[:, 0])

# Calculate crop boundaries for the larger area
Y1 = centery - np.floor(nx_padded/2)
Y2 = Y1 + nx_padded
X1 = centerx - np.floor(nx_padded/2)
X2 = X1 + nx_padded

# Create initial probe and guess object
aperture = 2*makeCircleMask(np.round(aperture_radius))
mask = makeCircleMask(np.round(aperture_radius)).to(device)

rand_mag = torch.abs(torch.rand(int(bigx), int(bigy)))
rand_phase = (torch.pi / 2) * torch.rand(int(bigx), int(bigy))
# rand_phase = torch.rand(int(bigx), int(bigy))
big_obj = rand_mag * torch.exp(1j * rand_phase)

# def random_complex_vector(size):
#     # Generate magnitude and phase
#     magnitude = torch.abs(torch.randn(int(bigx), int(bigx)))
#     phase = phi_true + 2 * torch.randn(phi_true.shape)

#     # Combine real and imaginary parts to create complex numbers
#     complex_vector = magnitude * torch.exp(1j * phase)

#     return complex_vector

# Create large padded images as template for random image
new_size = (nx, nx)  # New size of the image (width, height)

resize_image(input_path1, output_path1, new_size)
resize_image(input_path2, output_path2, new_size)

im1 = np.array(Image.open(output_path1).convert("L"))
im2 = np.array(Image.open(output_path2).convert("L"))

# Adjust padding to ensure the final size is (bigx, bigy)
# Here we assume that the final desired size is exactly (bigx, bigy)
pad_x1 = (int(bigx) - im1.shape[0]) // 2
pad_x2 = int(bigx) - im1.shape[0] - pad_x1
pad_y1 = (int(bigy) - im1.shape[1]) // 2
pad_y2 = int(bigy) - im1.shape[1] - pad_y1

im1 = np.pad(im1, ((pad_x1, pad_x2), (pad_y1, pad_y2)), mode='constant', constant_values=0)
im2 = np.pad(im2, ((pad_x1, pad_x2), (pad_y1, pad_y2)), mode='constant', constant_values=0)

r_true = torch.DoubleTensor(im1)
r_true = r_true / torch.max(r_true)
phi_true = torch.DoubleTensor(im2)
phi_true = phi_true / torch.max(phi_true)
phi_true = (torch.pi / 2) * phi_true  # Adjusted to range between 0 and pi/2

############################## Comment the following line out to have a completely random image ##############################
# big_obj += random_complex_vector(int(bigx))

del r_true
del phi_true


# For PSNR computation
def interpolate_image_for_psnr(image):
    # Interpolate the image to [0,1]
    min_val = image.min()
    max_val = image.max()
    interpolated_image = (image - min_val) / (max_val - min_val)

    # Ensure the image values are in the range [0, 1]
    interpolated_image = torch.clamp(interpolated_image, 0, 1)

    return interpolated_image



# ## Crop Padding and Compute Relative Error Functions

# Create unpadded, true image for cropping and relative errors later
new_size = (nx, nx)  # New size of the image (width, height)

resize_image(input_path1, output_path1, new_size)
resize_image(input_path2, output_path2, new_size)

im1 = np.array(Image.open(output_path1).convert("L"))
im2 = np.array(Image.open(output_path2).convert("L"))

r_true = torch.DoubleTensor(im1)
r_true = r_true/torch.max(r_true)
phi_true = torch.DoubleTensor(im2)
phi_true = phi_true/torch.max(phi_true) 
phi_true = (torch.pi / 2) * phi_true  # Adjusted to range between 0 and pi/2 

z_true_unpadded = torch.abs(r_true) * torch.exp(1j * phi_true)

# # Both functions can also be used for magnitude and phase
# def crop_image(z_opt, z_true=z_true_unpadded, nx=nx):
#     """
#     Crop the optimized image to match the size of the true image
#     """
#     correlation = match_template(torch.abs(z_opt).cpu().numpy(), torch.abs(z_true).cpu().numpy(), pad_input=True)
#     correlation = torch.tensor(correlation)

#     # Extract the region with highest correlation (centered around the peak)
#     h1 = torch.tensor(z_opt.cpu().numpy().shape) // 2 
#     correlation_sub = correlation[h1[0]-int(nx//2) : h1[0]+int(nx//2), h1[1]-int(nx//2) : h1[1]+int(nx//2)]
#     max_val = torch.max(correlation_sub).item()
#     I = torch.nonzero(correlation == max_val, as_tuple=False)

#     # Extract coordinates of the peak
#     I1, I2 = I[0].tolist()

#     # Extract the aligned object from the big object
#     object1 = z_opt[I1-int(nx//2) : I1+int(nx//2), I2-int(nx//2) : I2+int(nx//2)]

#     return object1

# Both functions can also be used for magnitude and phase
def crop_image(z_opt, z_true=z_true_unpadded, nx=nx):
    """
    Crop the optimized image to match the size of the true image
    """
    correlation = match_template(torch.abs(z_opt).cpu().numpy(), torch.abs(z_true).cpu().numpy(), pad_input=True)
    correlation = torch.tensor(correlation)

    # Extract the region with highest correlation (centered around the peak)
    h1 = torch.tensor(z_opt.cpu().numpy().shape) // 2 
    correlation_sub = correlation[h1[0]-int(nx//2) : h1[0]+int(nx//2), h1[1]-int(nx//2) : h1[1]+int(nx//2)]
    max_val = torch.max(correlation_sub).item()
    I = torch.nonzero(correlation == max_val, as_tuple=False)

    # Extract coordinates of the peak
    I1, I2 = I[0].tolist()
    
    # print('I1 = ', I1, ', I2 = ', I2)
    if I1 < 0 or I2 < 0 or I1 >= nx_padded or I2 >= nx_padded:
        print('indices outside image were found in crop_image...')
        object1 = z_opt[h1[0]-int(nx//2) : h1[0]+int(nx//2), h1[1]-int(nx//2) : h1[1]+int(nx//2)] # double check this.
        # object1 = z_opt[I1-int(nx//2) : I1+int(nx//2), I2-int(nx//2) : I2+int(nx//2)]
    else:
        # Extract the aligned object from the big object
        object1 = z_opt[I1-int(nx//2) : I1+int(nx//2), I2-int(nx//2) : I2+int(nx//2)]

    return object1

def rel_error(z_opt, z_true=z_true_unpadded):
    # NOTE: z_opt is assumed to be cropped to the size of z_true_unpadded
    # Compute and return the relative error
    return (torch.norm(z_true.cpu() - z_opt.cpu()) / torch.norm(z_true.cpu())).item()


# print(' SAVING TRUE IMAGES')
# fig = plt.figure()
# plt.imshow(z_true_unpadded.imag.cpu(), cmap=cmap)
# plt.colorbar()
# save_str = './true.png'
# fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


# fig = plt.figure()
# plt.imshow(r_true.cpu(), vmin=0, vmax=1, cmap=cmap)
# plt.colorbar()
# save_str = './true_magnitude.png'
# fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


# fig = plt.figure()
# plt.imshow(phi_true.cpu(), cmap=cmap, vmin=0, vmax=np.pi/2)
# plt.colorbar()
# save_str = './true_phase.png'
# fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)



# ## ePIE Algorithms (with and without PFT)

def ePIE(z, Q, lmbda=0, beta_ap=0.01, beta_obj=1, Y1=Y1, Y2=Y2, X1=X1, X2=X2, num_iters = 200, tol=1e-1, z_true=z_true_unpadded):

    f_val_hist = []
    grad_z_hist = []
    grad_Q_hist = []
    rel_err_hist = []
    cauchy_err_hist = []
    time_hist = []
    SSIM_phase_hist = []
    SSIM_mag_hist = []
    PSNR_phase_hist = []
    PSNR_mag_hist = []
    mag_err_hist = []
    phase_err_hist = []
    probe_err_hist = []

    cauchy_err = 100
    j = 1

    mag_true = torch.Tensor(torch.abs(z_true_unpadded)).cpu()
    phase_true = torch.Tensor(torch.angle(z_true_unpadded)).cpu()
    
    # Main ePIE iteration loop
    while cauchy_err > tol:

        z_old = z.clone().cpu()

        start = time.time()
        for i in torch.randperm(n_probes):

            # Choose region O_{n}^{j}
            obj_n = z[int(Y1[i]) : int(Y2[i]), int(X1[i]) : int(X2[i])].clone()
            
            # Compute max of obj and probe for step size
            object_max = torch.max(torch.abs(obj_n))**2 + 1e-5
            probe_max = torch.max(torch.abs(Q))**2 + 1e-5

            update_factor_obj = beta_obj / probe_max
            update_factor_probe = beta_ap / object_max
            
            fval, grad_zn, grad_Q = f_single_probe(obj_n, Q, lmbda=lmbda, current_ap = i, return_gradient=True) 
            z[int(Y1[i]) : int(Y2[i]), int(X1[i]) : int(X2[i])] = obj_n - (update_factor_obj * grad_zn)
            Q = Q - (update_factor_probe * grad_Q)

            # Comment out following two lines for standard ePIE
            Q.real = Q.real*mask.real 
            Q.imag = Q.imag*mask.imag
        end = time.time()
        iter_time = end - start
        time_hist.append(iter_time)

        # Crop image from padding to compute relative error, ssim, and psnr
        # NOTE: Cropping larger images may be expensive/time consuming
        # print('beginning crop...')
        start_time = time.time()
        z_cropped = crop_image(z)
        end_time = time.time()
        # print('cropping time = ', end_time - start_time, ', z_cropped.shape = ', z_cropped.shape, 'z_true.shape = ', z_true.shape)
        rel_err = rel_error(z_cropped, z_true)
        assert Q.shape == true_probe.shape
        assert Q.type() == true_probe.type()
        probe_err = torch.norm(Q - true_probe)/torch.norm(true_probe)

        start_time = time.time()
        fval_full, grad_z, grad_Q = f_original(z, Q, return_gradient=True)
        end_time = time.time()
        f_val_time = end_time - start_time
        

        cauchy_err = torch.norm(z_old - z.cpu()) # /(torch.norm(z_old)

        f_val_hist.append(fval_full.cpu())
        grad_z_hist.append(torch.norm(grad_z).cpu().item())
        grad_Q_hist.append(torch.norm(grad_Q).cpu().item())
        rel_err_hist.append(rel_err)
        cauchy_err_hist.append(cauchy_err.cpu())

        # Collect Magnitude and Phase (for computing ssim and psnr)
        mag_opt = torch.abs(z_cropped).cpu().detach()
        phase_opt = torch.angle(z_cropped).cpu().detach()

        # Compute and Save mag_err and phase_err
        mag_err = torch.norm(mag_opt - mag_true) / torch.norm(mag_true)
        phase_err = torch.norm(phase_opt - phase_true) / torch.norm(phase_true)
        mag_err_hist.append(mag_err.cpu())
        phase_err_hist.append(phase_err.cpu())

        probe_err_hist.append(probe_err.cpu())

        # mag_ssim = 0.0
        # phase_ssim = 0.0
        # mag_psnr = 0.0
        # phase_psnr = 0.0

        # Compute and Save SSIM
        mag_ssim = ssim(mag_opt.numpy(), r_true.cpu().detach().numpy(), data_range=r_true.cpu().detach().numpy().max()-r_true.cpu().detach().numpy().min())
        phase_ssim = ssim(phase_opt.numpy(), phi_true.cpu().detach().numpy(), data_range=phi_true.cpu().detach().numpy().max()-phi_true.cpu().detach().numpy().min())

        SSIM_mag_hist.append(mag_ssim)
        SSIM_phase_hist.append(phase_ssim)

        # Compute and Save PSNR
        mag_psnr = psnr(interpolate_image_for_psnr(mag_opt).numpy(), interpolate_image_for_psnr(r_true).numpy())
        phase_psnr = psnr(interpolate_image_for_psnr(phase_opt).numpy(), interpolate_image_for_psnr(phi_true).numpy())

        PSNR_mag_hist.append(mag_psnr)
        PSNR_phase_hist.append(phase_psnr)

        print('j: ', j,
                ' fx: ', "{:5.2e}".format(fval_full.item()),
                ' |grad_z|: ', "{:5.2e}".format(torch.norm(grad_z).item()),
                ' |grad_Q|: ', "{:5.2e}".format(torch.norm(grad_Q).item()),
                ' |x-xold|: ', "{:5.2e}".format(cauchy_err.item()),
                ' rel_err: ', "{:5.2e}".format(rel_err),
                ' phase_ssim: ', "{:5.2e}".format(phase_ssim),
                ' mag_ssim: ', "{:5.2e}".format(mag_ssim),
                ' phase_psnr: ', "{:5.2e}".format(phase_psnr),
                ' mag_psnr: ', "{:5.2e}".format(mag_psnr),
                ' phase_err: ', "{:5.2e}".format(phase_err),
                ' mag_err: ', "{:5.2e}".format(mag_err),
                ' probe_err:', "{:5.2e}".format(probe_err),
                ' iter time: ', "{:5.2e}".format(iter_time),
                ' f_full time:', "{:5.2e}".format(f_val_time)
                )

        if cauchy_err < tol or j >= num_iters:
            print('\n Converged at step:', j)
            print('j: ', j,
                 ' fx: ', "{:5.2e}".format(fval_full.item()),
                 ' |grad_z|: ', "{:5.2e}".format(torch.norm(grad_z).item()),
                 ' |grad_Q|: ', "{:5.2e}".format(torch.norm(grad_Q).item()),
                 ' |x-xold|: ', "{:5.2e}".format(cauchy_err.item()),
                 ' rel_err: ', "{:5.2e}".format(rel_err),
                 ' iter time: ', "{:5.2e}".format(iter_time),
                 ' f_full time:', "{:5.2e}".format(f_val_time)
                 )
            break

        j += 1
        
        ''' # Uncomment to display phase every 500 iters
        if j%500 == 0:
            plt.imshow(torch.angle(z).cpu(), cmap=cmap)
            plt.colorbar()
            plt.show()
        '''
    
    return z, Q, f_val_hist, grad_z_hist, grad_Q_hist, rel_err_hist, cauchy_err_hist, time_hist, SSIM_mag_hist, SSIM_phase_hist, PSNR_mag_hist, PSNR_phase_hist, phase_err_hist, mag_err_hist, probe_err_hist


def ePIE_PFT(z, Q, lmbda=0.005, beta_ap=0.01, beta_obj=1, Y1=Y1, Y2=Y2, X1=X1, X2=X2, num_iters = 2001, tol=2.5e-1, z_true=z_true_unpadded):

    f_val_hist = []
    grad_z_hist = []
    grad_Q_hist = []
    rel_err_hist = []
    cauchy_err_hist = []
    time_hist = []
    SSIM_phase_hist = []
    SSIM_mag_hist = []
    PSNR_phase_hist = []
    PSNR_mag_hist = []
    mag_err_hist = []
    phase_err_hist = []
    probe_err_hist = []

    cauchy_err = 100
    j = 1

    mag_true = torch.Tensor(torch.abs(z_true_unpadded)).cpu()
    phase_true = torch.Tensor(torch.angle(z_true_unpadded)).cpu()
    
    # Main ePIE iteration loop
    while cauchy_err > tol:

        z_old = z.clone().cpu()

        start = time.time()
        for i in torch.randperm(n_probes):

            # Choose region O_{n}^{j}
            obj_n = z[int(Y1[i]) : int(Y2[i]), int(X1[i]) : int(X2[i])].clone()
            
            # Compute max of obj and probe for step size
            object_max = torch.max(torch.abs(obj_n))**2
            probe_max = torch.max(torch.abs(Q))**2

            update_factor_obj = beta_obj / probe_max
            update_factor_probe = beta_ap / object_max
            
            fval, grad_z, grad_Q = f_pft(obj_n, Q, lmbda = lmbda, current_ap = i, return_gradient=True) 
            z[int(Y1[i]) : int(Y2[i]), int(X1[i]) : int(X2[i])] = obj_n - (update_factor_obj * grad_z)
            Q = Q - (update_factor_probe * grad_Q)

            # Comment out following two lines for standard ePIE
            Q.real = Q.real*mask.real 
            Q.imag = Q.imag*mask.imag
        end = time.time()
        iter_time = end - start
        time_hist.append(iter_time)

        # Crop image from padding to compute relative error, ssim, and psnr
        # NOTE: Cropping larger images may be expensive/time consuming
        z_cropped = crop_image(z)
        rel_err = rel_error(z_cropped, z_true)
        assert Q.shape == true_probe.shape
        assert Q.type() == true_probe.type()
        probe_err = torch.norm(Q - true_probe)/torch.norm(true_probe)

        start_time = time.time()
        fval_full, grad_z, grad_Q = f_original(z, Q, return_gradient=True)
        end_time = time.time()
        f_val_time = end_time - start_time
        
        
        cauchy_err = torch.norm(z_old - z.cpu()) # /(torch.norm(z_old)

        f_val_hist.append(fval_full.cpu())
        grad_z_hist.append(torch.norm(grad_z).cpu().item())
        grad_Q_hist.append(torch.norm(grad_Q).cpu().item())
        rel_err_hist.append(rel_err)
        cauchy_err_hist.append(cauchy_err.cpu())

        # Collect Magnitude and Phase (for computing ssim and psnr)
        mag_opt = torch.abs(z_cropped).cpu().detach()
        phase_opt = torch.angle(z_cropped).cpu().detach()

        # Compute and Save mag_err and phase_err
        mag_err = torch.norm(mag_opt - mag_true) / torch.norm(mag_true)
        phase_err = torch.norm(phase_opt - phase_true) / torch.norm(phase_true)
        mag_err_hist.append(mag_err.cpu())
        phase_err_hist.append(phase_err.cpu())

        probe_err_hist.append(probe_err.cpu())

        # mag_ssim = 0.0
        # phase_ssim = 0.0
        # mag_psnr = 0.0
        # phase_psnr = 0.0
        ##################################################################################################
        ##################################################################################################
        # Compute and Save SSIM
        mag_ssim = ssim(mag_opt.numpy(), r_true.cpu().detach().numpy(), data_range=r_true.cpu().detach().numpy().max()-r_true.cpu().detach().numpy().min())
        phase_ssim = ssim(phase_opt.numpy(), phi_true.cpu().detach().numpy(), data_range=phi_true.cpu().detach().numpy().max()-phi_true.cpu().detach().numpy().min())

        SSIM_mag_hist.append(mag_ssim)
        SSIM_phase_hist.append(phase_ssim)

        # Compute and Save PSNR
        mag_psnr = psnr(interpolate_image_for_psnr(mag_opt).numpy(), interpolate_image_for_psnr(r_true).numpy())
        phase_psnr = psnr(interpolate_image_for_psnr(phase_opt).numpy(), interpolate_image_for_psnr(phi_true).numpy())

        PSNR_mag_hist.append(mag_psnr)
        PSNR_phase_hist.append(phase_psnr)

        print('j:', j,
                ' fx: ', "{:5.2e}".format(fval_full.item()),
                ' |grad_z|: ', "{:5.2e}".format(torch.norm(grad_z).item()),
                ' |grad_Q|: ', "{:5.2e}".format(torch.norm(grad_Q).item()),
                ' |x-xold|: ', "{:5.2e}".format(cauchy_err.item()),
                ' rel_err: ', "{:5.2e}".format(rel_err),
                ' phase_ssim: ', "{:5.2e}".format(phase_ssim),
                ' mag_ssim: ', "{:5.2e}".format(mag_ssim),
                ' phase_psnr: ', "{:5.2e}".format(phase_psnr),
                ' mag_psnr: ', "{:5.2e}".format(mag_psnr),
                ' phase_err: ', "{:5.2e}".format(phase_err),
                ' mag_err: ', "{:5.2e}".format(mag_err),
                ' probe_err:', "{:5.2e}".format(probe_err),
                ' iter time: ', "{:5.2e}".format(iter_time),
                ' f_full time:', "{:5.2e}".format(f_val_time)
                )

        if cauchy_err < tol or j >= num_iters:
            print('\n Converged at step:', j)
            print('j: ', j,
                 ' fx: ', "{:5.2e}".format(fval_full.item()),
                 ' |grad_z|: ', "{:5.2e}".format(torch.norm(grad_z).item()),
                 ' |grad_Q|: ', "{:5.2e}".format(torch.norm(grad_Q).item()),
                 ' |x-xold|: ', "{:5.2e}".format(cauchy_err.item()),
                 ' rel_err: ', "{:5.2e}".format(rel_err),
                 ' iter time: ', "{:5.2e}".format(iter_time),
                 ' f_full time:', "{:5.2e}".format(f_val_time)
                 )
            break

        j += 1

        ''' # Uncomment to display phase every 500 iters
        if j%500 == 0:
            plt.imshow(torch.angle(z).cpu(), cmap=cmap)
            plt.colorbar()
            plt.show()
        '''
    
    return z, Q, f_val_hist, grad_z_hist, grad_Q_hist, rel_err_hist, cauchy_err_hist, time_hist, SSIM_mag_hist, SSIM_phase_hist, PSNR_mag_hist, PSNR_phase_hist, phase_err_hist, mag_err_hist, probe_err_hist



# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 
# EXPERIMENTS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

print('using pattern_match')
print('-------------------------------------------------------- STARTING FULL ePIE ----------------------------------------------------')
tv_lambda = 1e-6
beta_ap = 1e1
beta_obj = 1e1
num_iters = 50
tol = 1e-2

print('tv_lambda = ', tv_lambda, 'beta_ap = ', beta_ap, 'beta_obj = ', beta_obj, 'num_iters = ', num_iters, 'tol = ', tol)

z_guess = big_obj.to(device)
probe_guess = aperture.to(device)
z_ePIE, probe_ePIE, f_val_hist, grad_z_hist, grad_Q_hist, rel_err_hist, cauchy_err_hist, time_hist, SSIM_mag_hist, SSIM_phase_hist, PSNR_mag_hist, PSNR_phase_hist, phase_err_hist, mag_err_hist, probe_err_hist = ePIE(z_guess.clone(), probe_guess.clone(), lmbda=tv_lambda, beta_ap=beta_ap, beta_obj=beta_obj, num_iters=num_iters, tol=tol)


fig = plt.figure()
plt.imshow(z_ePIE.imag.cpu(), vmin=-2, vmax=1, cmap=cmap)
plt.colorbar()
save_str = './results/full_reconstruction_uncropped.png'
fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


fig = plt.figure()
plt.imshow(crop_image(z_ePIE).imag.cpu(), cmap=cmap)
plt.colorbar()
save_str = './results/full_reconstruction_cropped.png'
fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


fig = plt.figure()
plt.imshow(torch.abs(z_ePIE).cpu(), vmin=0, vmax=1, cmap=cmap)
plt.colorbar()
save_str = './results/full_magnitude_uncropped.png'
fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


fig = plt.figure()
# plt.imshow(torch.abs(crop_image(z_ePIE)).cpu(), vmin=0, vmax=1, cmap=cmap)
plt.imshow(crop_image(torch.abs(z_ePIE), z_true=r_true), vmin=0, vmax=1, cmap=cmap)
plt.colorbar()
save_str = './results/full_magnitude_cropped.png'
fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


fig = plt.figure()
plt.imshow(torch.angle(z_ePIE).cpu(), cmap=cmap, vmin=0.0, vmax=np.pi/2)
plt.colorbar()
save_str = './results/full_phase_uncropped.png'
fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


fig = plt.figure()
# plt.imshow(crop_image(torch.angle(z_ePIE)).cpu(), cmap=cmap, vmin=0.0, vmax=np.pi/2)
plt.imshow(crop_image(torch.angle(z_ePIE), z_true=phi_true), vmin=0.0, vmax=np.pi/2, cmap=cmap)
plt.colorbar()
save_str = './results/full_phase_cropped.png'
fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


fig = plt.figure()
plt.imshow(torch.abs(probe_ePIE).cpu(), cmap=cmap)
plt.colorbar()
save_str = './results/full_probe_mag.png'
fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


fig = plt.figure()
plt.imshow(torch.angle(probe_ePIE).cpu(), cmap=cmap, vmin=0.0, vmax=np.pi/2)
plt.colorbar()
save_str = './results/full_probe_phase.png'
fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


file_name = './results/full_ePIE_hist.pt'
state = {
    'z_ePIE': z_ePIE,
    'probe_ePIE': probe_ePIE,
    'f_val_hist': f_val_hist,
    'grad_z_hist': grad_z_hist,
    'grad_Q_hist': grad_Q_hist,
    'rel_err_hist': rel_err_hist,
    'cauchy_err_hist': cauchy_err_hist,
    'time_hist': time_hist,
    'SSIM_phase_hist': SSIM_phase_hist,
    'SSIM_mag_hist': SSIM_mag_hist,
    'PSNR_phase_hist': PSNR_phase_hist,
    'PSNR_mag_hist': PSNR_mag_hist,
    'mag_err_hist': mag_err_hist,
    'phase_err_hist': phase_err_hist,
    'probe_err_hist': probe_err_hist,
    'beta_ap': beta_ap,
    'beta_obj':beta_obj,
    'tol':tol,
    'tv_lmbda': tv_lambda,
    'num_iters': num_iters
}
torch.save(state, file_name)
print('files saved to ' + file_name)

# # Uncomment to delete results after saving, if needed
del z_ePIE, probe_ePIE

print('\n\n-------------------------------------------------------- STARTING PFT ePIE with probe errs ----------------------------------------------------')
tv_lambda_PFT= 1e3
beta_ap_PFT = 2e-3
beta_obj_PFT = 2e-3
num_iters_PFT = 5
tol_PFT = 1e-2

assert probe_guess.type() == true_probe.type()
assert probe_guess.shape == true_probe.shape
print('probe error = ', torch.norm(probe_guess - true_probe))

print('tv_lambda_PFT = ', tv_lambda_PFT, 'beta_ap_PFT = ', beta_ap_PFT, 'beta_obj_PFT = ', beta_obj_PFT, 'num_iters_PFT = ', num_iters_PFT, 'tol = ', tol_PFT)

z_ePIE_PFT, probe_ePIE_PFT, f_val_hist_PFT, grad_z_hist_PFT, grad_Q_hist_PFT, rel_err_hist_PFT, cauchy_err_hist_PFT, time_hist_PFT, SSIM_mag_hist_PFT, SSIM_phase_hist_PFT, PSNR_mag_hist_PFT, PSNR_phase_hist_PFT, phase_err_hist_PFT, mag_err_hist_PFT, probe_err_hist_PFT = ePIE_PFT(z_guess.clone(), probe_guess.clone(), lmbda=tv_lambda_PFT, beta_ap=beta_ap_PFT, beta_obj=beta_obj_PFT, num_iters=num_iters_PFT, tol=tol_PFT)


# fig = plt.figure()
# plt.imshow(z_ePIE_PFT.imag.cpu(), vmin=-2, vmax=1, cmap=cmap)
# plt.colorbar()
# save_str = './results/PFT_reconstruction_uncropped.png'
# fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


# fig = plt.figure()
# plt.imshow(crop_image(z_ePIE_PFT).imag.cpu(), vmin=-2, vmax=1, cmap=cmap)
# plt.colorbar()
# save_str = './results/PFT_reconstruction_cropped.png'
# fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


fig = plt.figure()
plt.imshow(torch.abs(z_ePIE_PFT).cpu(), vmin=0, vmax=1, cmap=cmap)
plt.colorbar()
save_str = './results/PFT_magnitude_uncropped.png'
fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


fig = plt.figure()
# plt.imshow(torch.abs(crop_image(z_ePIE_PFT)).cpu(), vmin=0, vmax=1, cmap=cmap)
plt.imshow(crop_image(torch.abs(z_ePIE_PFT), z_true=r_true), vmin=0.0, vmax=1.0, cmap=cmap)
plt.colorbar()
save_str = './results/PFT_magnitude_cropped.png'
fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


fig = plt.figure()
plt.imshow(torch.angle(z_ePIE_PFT).cpu(), cmap=cmap, vmin=0.0, vmax=np.pi/2)
plt.colorbar()
save_str = './results/PFT_phase_uncropped.png'
fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


fig = plt.figure()
# plt.imshow(crop_image(torch.angle(z_ePIE_PFT)).cpu(), cmap=cmap, vmin=0.0, vmax=np.pi/2)
plt.imshow(crop_image(torch.angle(z_ePIE_PFT), z_true=phi_true), vmin=0.0, vmax=np.pi/2, cmap=cmap)
plt.colorbar()
save_str = './results/PFT_phase_cropped.png'
fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


fig = plt.figure()
plt.imshow(torch.abs(probe_ePIE_PFT).cpu(), cmap=cmap)
plt.colorbar()
save_str = './results/PFT_probe_mag.png'
fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


fig = plt.figure()
plt.imshow(torch.angle(probe_ePIE_PFT).cpu(), cmap=cmap, vmin=0.0, vmax=np.pi/2)
plt.colorbar()
save_str = './results/PFT_probe_phase.png'
fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


file_name = './results/ePIE_PFT_hist.pt'
state = {
    'z_ePIE_PFT': z_ePIE_PFT,
    'probe_ePIE_PFT': probe_ePIE_PFT,
    'f_val_hist': f_val_hist_PFT,
    'grad_z_hist': grad_z_hist_PFT,
    'grad_Q_hist': grad_Q_hist_PFT,
    'rel_err_hist': rel_err_hist_PFT,
    'cauchy_err_hist': cauchy_err_hist_PFT,
    'time_hist': time_hist_PFT,
    'SSIM_phase_hist': SSIM_phase_hist_PFT,
    'SSIM_mag_hist': SSIM_mag_hist_PFT,
    'PSNR_phase_hist': PSNR_phase_hist_PFT,
    'PSNR_mag_hist': PSNR_mag_hist_PFT,
    'mag_err_hist': mag_err_hist_PFT,
    'phase_err_hist': phase_err_hist_PFT,
    'probe_err_hist': probe_err_hist_PFT,
    'beta_ap_PFT': beta_ap_PFT,
    'beta_obj_PFT':beta_obj_PFT,
    'tol_PFT':tol_PFT,
    'tv_lmbda_PFT': tv_lambda_PFT,
    'num_iters_PFT': num_iters_PFT
}
torch.save(state, file_name)
print('files saved to ' + file_name)


print('\n\n-------------------------------------------------------- STARTING Warmstarted ePIE ----------------------------------------------------')
tv_lambda = tv_lambda
beta_ap = beta_ap
beta_obj = beta_obj
num_iters = num_iters
tol = tol

# load z_ePIE_PFT and probe_ePIE_PFT
pft_dict = torch.load('./results/ePIE_PFT_hist.pt')
z_ePIE_PFT = pft_dict['z_ePIE_PFT']
probe_ePIE_PFT = pft_dict['probe_ePIE_PFT']
print('Loaded z_ePIE_PFT and probe_ePIE_PFT')

print('tv_lambda = ', tv_lambda, 'beta_ap = ', beta_ap, 'beta_obj = ', beta_obj, 'num_iters = ', num_iters, 'tol = ', tol)

z_optHybrid, probe_optHybrid, f_val_hist_hybrid, grad_z_hist_hybrid, grad_Q_hist_hybrid, rel_err_hist_hybrid, cauchy_err_hist_hybrid, time_hist_hybrid, SSIM_mag_hist_hybrid, SSIM_phase_hist_hybrid, PSNR_mag_hist_hybrid, PSNR_phase_hist_hybrid, phase_err_hist_hybrid, mag_err_hist_hybrid, probe_err_hist_hybrid = ePIE(z_ePIE_PFT.clone(), probe_ePIE_PFT, lmbda=tv_lambda, beta_ap=beta_ap, beta_obj=beta_obj, num_iters=num_iters, tol=tol)

fig = plt.figure()
plt.imshow(z_optHybrid.imag.cpu(), vmin=-2, vmax=1, cmap=cmap)
plt.colorbar()
save_str = './results/hybrid_reconstruction_uncropped.png'
fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


fig = plt.figure()
plt.imshow(crop_image(z_optHybrid).imag.cpu(), vmin=-2, vmax=1, cmap=cmap)
plt.colorbar()
save_str = './results/hybrid_reconstruction_cropped.png'
fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


fig = plt.figure()
plt.imshow(torch.abs(z_optHybrid).cpu(), vmin=0, vmax=1, cmap=cmap)
plt.colorbar()
save_str = './results/hybrid_magnitude_uncropped.png'
fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


fig = plt.figure()
# plt.imshow(torch.abs(crop_image(z_optHybrid)).cpu(), vmin=0, vmax=1, cmap=cmap)
plt.imshow(crop_image(torch.abs(z_optHybrid), z_true=r_true), vmin=0.0, vmax=1.0, cmap=cmap)
plt.colorbar()
save_str = './results/hybrid_magnitude_cropped.png'
fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


fig = plt.figure()
plt.imshow(torch.angle(z_optHybrid).cpu(), cmap=cmap, vmin=0.0, vmax=np.pi/2)
plt.colorbar()
save_str = './results/hybrid_phase_uncropped.png'
fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


fig = plt.figure()
# plt.imshow(crop_image(torch.angle(z_optHybrid)).cpu(), cmap=cmap, vmin=0.0, vmax=np.pi/2)
plt.imshow(crop_image(torch.angle(z_optHybrid), z_true=phi_true), vmin=0.0, vmax=np.pi/2, cmap=cmap)
plt.colorbar()
save_str = './results/hybrid_phase_cropped.png'
fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


fig = plt.figure()
plt.imshow(torch.abs(probe_optHybrid).cpu(), cmap=cmap)
plt.colorbar()
save_str = './results/hybrid_probe_mag.png'
fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


fig = plt.figure()
plt.imshow(torch.angle(probe_optHybrid).cpu(), cmap=cmap, vmin=0.0, vmax=np.pi/2)
plt.colorbar()
save_str = './results/hybrid_probe_phase.png'
fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)


file_name = './results/ePIE_hybrid_hist.pt'
state = {
    'z_ePIE_hybrid': z_optHybrid,
    'probe_ePIE_hybrid': probe_optHybrid,
    'f_val_hist': f_val_hist_hybrid,
    'grad_z_hist': grad_z_hist_hybrid,
    'grad_Q_hist': grad_Q_hist_hybrid,
    'rel_err_hist': rel_err_hist_hybrid,
    'cauchy_err_hist': cauchy_err_hist_hybrid,
    'time_hist': time_hist_hybrid,
    'SSIM_phase_hist': SSIM_phase_hist_hybrid,
    'SSIM_mag_hist': SSIM_mag_hist_hybrid,
    'PSNR_phase_hist': PSNR_phase_hist_hybrid,
    'PSNR_mag_hist': PSNR_mag_hist_hybrid,
    'mag_err_hist': mag_err_hist_hybrid,
    'phase_err_hist': phase_err_hist_hybrid,
    'probe_err_hist': probe_err_hist_hybrid
}
torch.save(state, file_name)
print('files saved to ' + file_name)
