from src.create_probes import *
from src.PFT2D import *
from src.create_data import *
from src.nonblind_ptychography_functions import *

import torch
import torch.fft as fft
from torchvision import transforms

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
    
input_path1 = "src/mandril_gray.tif"
output_path1 = "src/baboon_resized.png"
input_path2 = "src/cameraman.tif"
output_path2 = "src/cameraman_resized.png"

# SET SIZE OF IMAGE
nx = 16384
new_size = (nx, nx)  # New size of the image (width, height)

resize_image(input_path1, output_path1, new_size)
resize_image(input_path2, output_path2, new_size)

im1 = np.array(Image.open('src/baboon_resized.png').convert("L"))
im2 = np.array(Image.open('src/cameraman_resized.png').convert("L"))
r_true = torch.DoubleTensor(im1)
r_true = r_true/torch.max(r_true)
phi_true = torch.DoubleTensor(im2)
phi_true = phi_true/torch.max(phi_true) 
phi_true = 0.5* torch.pi * phi_true  # Adjusted to range between 0 and pi/2 

z_true = torch.abs(r_true) * torch.exp(1j * phi_true)
z_true = z_true.view(-1, 1).to(device)

n = z_true.shape[0]
m = n

# PFT Setup Variables
N = [nx, nx]
M = [64, 64]
mu = [0, 0]
p = [64, 64]
error = 'e-7'

# Create Probed Images
probes, n_probes = create_probes_nonblind(nx=nx, ny=nx, device=device)

# Uncomment to see illuminated image
'''
for i in range(n_probes):
    fig1 = plt.figure()
    current_img = probes[i,:,:]*z_true.view(nx,nx)
    plt.imshow(current_img.imag.cpu(), cmap=cmap)
    plt.show()
    # save_str = 'probed_reconstruction' + str(i+1) + '.pdf'
    # fig1.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
'''

# Compute percentage of overlap between consecutive probes
def calculate_overlap_percentage(probe1, probe2):
    overlap = torch.sum(probe1 * probe2)
    total_area = torch.sum(probe1)
    return (overlap / total_area).item() * 100.0

overlap_percentages = torch.zeros(n_probes, device=device)

for i in range(n_probes - 1):
    overlap_percentage = calculate_overlap_percentage(probes[i], probes[i + 1])
    overlap_percentages[i] = overlap_percentage

for i in range(n_probes - 1):
    print(f"Overlap percentage between probe {i} and probe {i + 1}: {overlap_percentages[i]:.2f}%")


#################### Create Observed Data ####################

b = create_full_data(z_true, probes, n_probes, nx, nx, n, isBlind=False, device=device)
b_crop = create_cropped_data(z_true, probes, n_probes, nx, nx, N, M, isBlind=False, device=device)

# Perform PFT precomputes
z_temp = probes[0,:,:]*z_true.view(nx,nx)

B, p, q, r = pft2d_configuration(N, M, mu, p, error, device=device)
z, m1_mod, m2_mod, precomputed_prod = pft2d_precompute(z_temp, M, mu, p, q, r, device=device)
print(f"N, M = {N[0], N[1]}, {M[0], M[1]} // p, q, r = {p}, {q}, {r} // e = {error}")

del z_temp

print('\n')

print('Full Data Shape', b[0].view(-1).shape)
print('Cropped Data Shape:', b_crop[0].shape)

# Store parameters
pft_params = {
    "B": B,
    "M": M,
    "p": p,
    "q": q,
    "m1_mod": m1_mod,
    "m2_mod": m2_mod,
    "precomputed_prod": precomputed_prod
}

PIE_params = {
    'b': b,
    'probes': probes,
    'nx': nx,
    'ny': nx,
    'z_true': z_true,
    'r_true': r_true,
    'phi_true': phi_true
}

PIE_PFT_params = {
    'b': b,
    'b_crop': b_crop,
    'probes': probes,
    'nx': nx,
    'ny': nx,
    'z_true': z_true,
    'r_true': r_true,
    'phi_true': phi_true,
    'pft_params': pft_params
}

if __name__ == '__main__':
    
    #################### Experiments ####################
    
    # Create initial guess object
    guess_object = random_complex_vector(nx, nx).to(device)
    
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 
    # EXPERIMENTS
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    print('-------------------------------------------------------- STARTING FULL PIE LARGE INIT pi/2 ----------------------------------------------------')
    
    # alpha = 1e-1
    alpha = 1.0
    num_iters= int(1e2)
    tol = 5e-4
    tv_lmbda = 1e-1
    
    print('alpha = ', alpha, ', num_iters = ', num_iters, ', tol = ', tol, 'tv_lmbda = ', tv_lmbda)
    
    ''' PIE With Full FFT '''
    z_guess = guess_object.view(nx, nx).clone().to(device)
    z_optPIE, f_val_hist, gradf_val_hist, rel_err_hist, cauchy_err_hist, time_hist, SSIM_mag_hist, SSIM_phase_hist, PSNR_mag_hist, PSNR_phase_hist, phase_err_hist, mag_err_hist  = PIE(z_guess.view(n), alpha = alpha, num_iters = num_iters, tol=tol, tv_lmbda=tv_lmbda, return_all_metrics=True, **PIE_params) 
    
    
    fig = plt.figure()
    plt.imshow(z_optPIE.imag.cpu(), cmap=cmap)
    plt.colorbar()
    save_str = 'results/large_scale_PIE_results/full_reconstruction.png'
    fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
    
    
    fig = plt.figure()
    r_optPIE = torch.abs(z_optPIE)
    plt.imshow(r_optPIE.cpu(), vmin=0, vmax=1, cmap=cmap)
    plt.colorbar()
    save_str = 'results/large_scale_PIE_results/full_magnitude.png'
    fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
    
    
    fig = plt.figure()
    phi_optPIE = torch.angle(z_optPIE)
    plt.imshow(phi_optPIE.cpu(), cmap=cmap, vmin=0, vmax=np.pi/2)
    plt.colorbar()
    save_str = 'results/large_scale_PIE_results/full_phase.png'
    fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
    
    
    file_name = 'results/large_scale_PIE_results/full_PIE_hist.pt'
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
    
    # Uncomment to delete results after saving, if needed
    # del z_optPIE
    
    
    print('-------------------------------------------------------- STARTING PFT PIE ----------------------------------------------------')
    
    # alpha_PFT = 1e-1 # for nx=16k
    alpha_PFT = 1e-3
    num_iters_PFT= int(2e1)
    tol_PFT = 1e-2
    tv_lmbda_PFT= 100.0
    
    print('alpha_PFT = ', alpha_PFT, ', num_iters_PFT = ', num_iters_PFT, ', tol_PFT = ', tol_PFT, ', tv_lmbda = ', tv_lmbda)
    
    ''' PIE With PFT '''
    z_guess = guess_object.view(nx, nx).clone().to(device)
    z_PFT, f_val_hist_PFT, gradf_val_hist_PFT, rel_err_hist_PFT, cauchy_err_hist_PFT, time_hist_PFT, SSIM_mag_hist_PFT, SSIM_phase_hist_PFT, PSNR_mag_hist_PFT, PSNR_phase_hist_PFT, phase_err_hist_PFT, mag_err_hist_PFT = PIE_PFT(z_guess.view(n), alpha = alpha_PFT, num_iters = num_iters_PFT, tol=tol_PFT, tv_lmbda=tv_lmbda_PFT, return_all_metrics=True, **PIE_PFT_params)
    
    
    fig = plt.figure()
    plt.imshow(z_PFT.imag.cpu(), cmap=cmap)
    plt.colorbar()
    save_str = 'results/large_scale_PIE_results/pft_reconstruction.png'
    fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
    
    
    fig = plt.figure()
    r_optPIE_PFT = torch.abs(z_PFT)
    plt.imshow(r_optPIE_PFT.cpu(), vmin=0, vmax=1, cmap=cmap)
    plt.colorbar()
    save_str = 'results/large_scale_PIE_results/pft_magnitude.png'
    fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
    
    
    fig = plt.figure()
    phi_optPIE_PFT = torch.angle(z_PFT)
    plt.imshow(phi_optPIE_PFT.cpu(), cmap=cmap, vmin=0, vmax=np.pi/2)
    plt.colorbar()
    save_str = 'results/large_scale_PIE_results/pft_phase.png'
    fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
    
    
    file_name = 'results/large_scale_PIE_results/PFT_hist.pt'
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
    # Using same parameters as full PIE
    alpha = alpha
    num_iters = num_iters
    tol = tol
    print('alpha = ', alpha, ', num_iters = ', num_iters, ', tol = ', tol)
    
    ''' PIE With Full FFT '''
    z_optHybrid, f_val_hist, gradf_val_hist, rel_err_hist, cauchy_err_hist, time_hist, SSIM_mag_hist, SSIM_phase_hist, PSNR_mag_hist, PSNR_phase_hist, phase_err_hist, mag_err_hist = PIE(z_PFT.view(n), alpha = alpha, num_iters = num_iters, tol=tol, return_all_metrics=True, **PIE_params)
    
    
    fig = plt.figure()
    plt.imshow(z_optHybrid.imag.cpu(), cmap=cmap)
    plt.colorbar()
    save_str = 'results/large_scale_PIE_results/hybrid_reconstruction.png'
    fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
    
    
    fig = plt.figure()
    r_optHybrid = torch.abs(z_optHybrid)
    plt.imshow(r_optHybrid.cpu(), vmin=0, vmax=1, cmap=cmap)
    plt.colorbar()
    save_str = 'results/large_scale_PIE_results/hybrid_magnitude.png'
    fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
    
    
    fig = plt.figure()
    phi_optHybrid = torch.angle(z_optHybrid)
    plt.imshow(phi_optHybrid.cpu(), cmap=cmap, vmin=0, vmax=np.pi/2)
    plt.colorbar()
    save_str = 'results/large_scale_PIE_results/hybrid_phase.png'
    fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
    
    
    file_name = 'results/large_scale_PIE_results/hybrid_hist.pt'
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
    
