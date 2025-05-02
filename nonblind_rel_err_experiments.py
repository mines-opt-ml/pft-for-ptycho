from src.create_probes import *
from src.PFT2D import *
from src.create_data import *
from src.nonblind_ptychography_functions import *

import torch
import torch.fft as fft

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import numpy as np
import matplotlib.pyplot as plt
import time
import math

from IPython.display import clear_output

torch.set_default_dtype(torch.float64)

import PIL
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = 368435456 
import csv

# device = 'cuda'
device = 'cpu'
cmap = 'gray'

def resize_image(input_image_path, output_image_path, new_size):
    original_image = Image.open(input_image_path)
    width, height = original_image.size
    resized_image = original_image.resize(new_size)
    resized_image.save(output_image_path)

# Example usage
input_path1 = "src/mandril_gray.tif"
output_path1 = "src/baboon_resized.png"
input_path2 = "src/cameraman.tif"
output_path2 = "src/cameraman_resized.png"

# SET SIZE OF IMAGE HERE
nx = 512
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
print('z_true.shape = ', z_true.shape)

# x_true = torch.FloatTensor(im)
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
    
    # Number of random images
    num_images = 150  
    
    # Lists to guess images and store relative errors
    guess_images = []
    
    rel_errors_PIE = []
    mag_rel_errors_PIE = []
    phase_rel_errors_PIE = []
    
    mag_ssim_PIE = []
    phase_ssim_PIE = []
    mag_psnr_PIE = []
    phase_psnr_PIE = []
    
    rel_errors_PIE_hybrid = []
    mag_rel_errors_PIE_hybrid = []
    phase_rel_errors_PIE_hybrid = []
    
    mag_ssim_PIE_hybrid = []
    phase_ssim_PIE_hybrid = []
    mag_psnr_PIE_hybrid = []
    phase_psnr_PIE_hybrid = []
    
    for k in range(num_images):
    
        # Generate guess object
        z_guess = random_complex_vector(nx, nx).to(device)
        guess_images.append(z_guess)
    
        print("-------------------- Full PIE --------------------")
        z_optPIE = PIE(z_guess.view(n), alpha = 1.0, num_iters = int(1e2), tol = 5e-4, tv_lmbda = 1e-1, **PIE_params) 
    
        full_PIE_rel_err = rel_error(z_optPIE, z_true, n)
        rel_errors_PIE.append(full_PIE_rel_err)
    
        mag_PIE_rel_err = rel_error(torch.abs(z_optPIE), r_true, n)
        mag_rel_errors_PIE.append(mag_PIE_rel_err)
    
        phase_PIE_rel_err = rel_error(torch.angle(z_optPIE), phi_true, n)
        phase_rel_errors_PIE.append(phase_PIE_rel_err)
    
        # Collect Magnitude and Phase (for computing ssim and psnr)
        mag_optPIE = torch.abs(z_optPIE).cpu().detach()
        phase_optPIE = torch.angle(z_optPIE).cpu().detach()
    
        # Compute and Save SSIM 
        mag_ssim = ssim(mag_optPIE.numpy(), r_true.cpu().detach().numpy(), data_range=r_true.cpu().detach().numpy().max()-r_true.cpu().detach().numpy().min())
        phase_ssim = ssim(phase_optPIE.numpy(), phi_true.cpu().detach().numpy(), data_range=phi_true.cpu().detach().numpy().max()-phi_true.cpu().detach().numpy().min())
        mag_ssim_PIE.append(mag_ssim)
        phase_ssim_PIE.append(phase_ssim)
    
        # Compute and Save PSNR
        mag_psnr = psnr(interpolate_image_for_psnr(mag_optPIE).numpy(), interpolate_image_for_psnr(r_true).numpy())
        phase_psnr = psnr(interpolate_image_for_psnr(phase_optPIE).numpy(), interpolate_image_for_psnr(phi_true).numpy())
        mag_psnr_PIE.append(mag_psnr)
        phase_psnr_PIE.append(phase_psnr)
    
        print('Full Rel Err:', "{:5.2e}".format(full_PIE_rel_err), 
              'Magnitude Rel Err:', "{:5.2e}".format(mag_PIE_rel_err), 'Phase Rel Err:', "{:5.2e}".format(phase_PIE_rel_err))
    
        print('\n')
    
        print("-------------------- Warm Start PIE --------------------")
        print('PIE PFT')
        z_optPIE_PFT = PIE_PFT(z_guess, alpha = 1e-3, num_iters = int(2e1), tol=1e-2, tv_lmbda= 100.0, **PIE_PFT_params) 
        print('\n')
        print('Full PIE')
        z_optHybrid = PIE(z_optPIE_PFT.view(n).clone(), alpha = 1.0, num_iters = int(1e2), tol = 5e-4, tv_lmbda = 1e-1, **PIE_params) 
    
        full_hybrid_rel_err = rel_error(z_optHybrid, z_true, n)
        rel_errors_PIE_hybrid.append(full_hybrid_rel_err)
    
        mag_hybrid_rel_err = rel_error(torch.abs(z_optHybrid), r_true, n)
        mag_rel_errors_PIE_hybrid.append(mag_hybrid_rel_err)
    
        phase_hybrid_rel_err = rel_error(torch.angle(z_optHybrid), phi_true, n)
        phase_rel_errors_PIE_hybrid.append(phase_hybrid_rel_err)
    
        # Collect Magnitude and Phase (for computing ssim and psnr)
        mag_optHybrid = torch.abs(z_optHybrid).cpu().detach()
        phase_optHybrid = torch.angle(z_optHybrid).cpu().detach()
    
        # Compute and Save SSIM
        mag_ssim = ssim(mag_optHybrid.numpy(), r_true.cpu().detach().numpy(), data_range=r_true.cpu().detach().numpy().max()-r_true.cpu().detach().numpy().min())
        phase_ssim = ssim(phase_optHybrid.numpy(), phi_true.cpu().detach().numpy(), data_range=phi_true.cpu().detach().numpy().max()-phi_true.cpu().detach().numpy().min())
        mag_ssim_PIE_hybrid.append(mag_ssim)
        phase_ssim_PIE_hybrid.append(phase_ssim)
    
        # Compute and Save PSNR
        mag_psnr = psnr(interpolate_image_for_psnr(mag_optHybrid).numpy(), interpolate_image_for_psnr(r_true).numpy())
        phase_psnr = psnr(interpolate_image_for_psnr(phase_optHybrid).numpy(), interpolate_image_for_psnr(phi_true).numpy())
        mag_psnr_PIE_hybrid.append(mag_psnr)
        phase_psnr_PIE_hybrid.append(phase_psnr)
    
        print('(Hybrid) Full Rel Err:', "{:5.2e}".format(full_hybrid_rel_err), 
              'Magnitude Rel Err:', "{:5.2e}".format(mag_hybrid_rel_err), 'Phase Rel Err:', "{:5.2e}".format(phase_hybrid_rel_err))
    
        # Clear the output
        clear_output(wait=True)
        time.sleep(3)
        print(f"Processed {k+1}/{num_images} images...")
        print('len(guess_images):', len(guess_images), 'len(rel_errors_PIE):', len(rel_errors_PIE), 'len(rel_errors_PIE_hybrid):', len(rel_errors_PIE_hybrid)) 
        print('\n')
    
    # Save Data
    file_name = 'results/nonblind_rel_error_results/PIE_rel_error_data.pt'
    state = {
        'guess_images': guess_images,
        
        'rel_errors_PIE': rel_errors_PIE,
        'mag_rel_errors_PIE': mag_rel_errors_PIE,
        'phase_rel_errors_PIE': phase_rel_errors_PIE,
    
        'mag_ssim_PIE': mag_ssim_PIE,
        'phase_ssim_PIE': phase_ssim_PIE,
        'mag_psnr_PIE': mag_psnr_PIE,
        'phase_psnr_PIE': phase_psnr_PIE,
        
        'rel_errors_PIE_hybrid': rel_errors_PIE_hybrid,
        'mag_rel_errors_PIE_hybrid': mag_rel_errors_PIE_hybrid,
        'phase_rel_errors_PIE_hybrid': phase_rel_errors_PIE_hybrid,
        
        'mag_ssim_PIE_hybrid': mag_ssim_PIE_hybrid,
        'phase_ssim_PIE_hybrid': phase_ssim_PIE_hybrid,
        'mag_psnr_PIE_hybrid': mag_psnr_PIE_hybrid,
        'phase_psnr_PIE_hybrid': phase_psnr_PIE_hybrid
    }
    torch.save(state, file_name)
    print('files saved to ' + file_name)
    
    
    fig = plt.figure()
    # Plot histograms of full relative errors
    plt.hist(rel_errors_PIE, bins=20, alpha=0.5, label='PIE', edgecolor='black')
    plt.hist(rel_errors_PIE_hybrid, bins=20, alpha=0.5, label='Hybrid PIE', edgecolor='black')
    plt.xlabel('Relative Error')
    plt.ylabel('Frequency')
    plt.title('Full Relative Errors in Reconstructed Images')
    plt.legend(loc='upper right')
    plt.show()
    
    fig = plt.figure()
    # Plot histograms of magnitude relative errors
    plt.hist(mag_rel_errors_PIE, bins=20, alpha=0.5, label='PIE', edgecolor='black')
    plt.hist(mag_rel_errors_PIE_hybrid, bins=20, alpha=0.5, label='Hybrid PIE', edgecolor='black')
    plt.xlabel('Relative Error')
    plt.ylabel('Frequency')
    plt.title('Magnitude Relative Errors in Reconstructed Images')
    plt.legend(loc='upper right')
    plt.show()
    
    fig = plt.figure()
    # Plot histograms of phase relative errors
    plt.hist(phase_rel_errors_PIE, bins=20, alpha=0.5, label='PIE', edgecolor='black')
    plt.hist(phase_rel_errors_PIE_hybrid, bins=20, alpha=0.5, label='Hybrid PIE', edgecolor='black')
    plt.xlabel('Relative Error')
    plt.ylabel('Frequency')
    plt.title('Phase Relative Errors in Reconstructed Images')
    plt.legend(loc='upper right')
    plt.show()
    
    
    fig = plt.figure()
    # Plot histograms of magnitude SSIM
    plt.hist(mag_ssim_PIE, bins=20, alpha=0.5, label='PIE', edgecolor='black')
    plt.hist(mag_ssim_PIE_hybrid, bins=20, alpha=0.5, label='Hybrid PIE', edgecolor='black')
    plt.xlabel('SSIM')
    plt.legend(loc='upper right')
    plt.show()
    
    fig = plt.figure()
    # Plot histograms of phase SSIM
    plt.hist(phase_ssim_PIE, bins=20, alpha=0.5, label='PIE', edgecolor='black')
    plt.hist(phase_ssim_PIE_hybrid, bins=20, alpha=0.5, label='Hybrid PIE', edgecolor='black')
    plt.xlabel('SSIM')
    plt.legend(loc='upper right')
    plt.show()
    
    fig = plt.figure()
    # Plot histograms of magnitude PSNR
    plt.hist(mag_psnr_PIE, bins=20, alpha=0.5, label='PIE', edgecolor='black')
    plt.hist(mag_psnr_PIE_hybrid, bins=20, alpha=0.5, label='Hybrid PIE', edgecolor='black')
    plt.xlabel('PSNR')
    plt.legend(loc='upper right')
    plt.show()
    
    fig = plt.figure()
    # Plot histograms of phase PSNR
    plt.hist(phase_psnr_PIE, bins=20, alpha=0.5, label='PIE', edgecolor='black')
    plt.hist(phase_psnr_PIE_hybrid, bins=20, alpha=0.5, label='Hybrid PIE', edgecolor='black')
    plt.xlabel('PSNR')
    plt.legend(loc='upper right')
    plt.show()
    
