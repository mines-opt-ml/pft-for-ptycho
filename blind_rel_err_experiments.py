from src.create_probes import *
from src.PFT2D import *
from src.create_data import *
from src.blind_ptychography_functions import *

import torch
import torch.fft as fft
from torchvision import transforms

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import numpy as np
import matplotlib.pyplot as plt
import math
import time

from IPython.display import clear_output

torch.set_default_dtype(torch.float64)

# set random seed
# torch.manual_seed(42)

import PIL
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = 668435456 
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

# SET SIZE OF IMAGE HERE
nx = 512
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
else: 
    im1 = np.pad(im1, ((256, 256), (256, 256)), mode='constant', constant_values=0)
    im2 = np.pad(im2, ((256, 256), (256, 256)), mode='constant', constant_values=0)

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

n = z_true.shape[0]
m = n

# PFT Setup Variables
N = [nx_padded, nx_padded]
M = [64, 64]
mu = [0, 0]
p = [64, 64]
error = 'e-7'

# Compute radius and spacing based on the dimensions of a 512x512 image
# These values can be adjusted as needed
radius = math.ceil((175 * nx_padded) / 768)
spacing = math.ceil((150 * nx_padded) / 768)

true_probe, _, centerx_true, centery_true = create_true_probe(z_true.view(nx_padded, nx_padded), radius=radius, sigma=1e6, device=device)

# Uncomment to display true probe
'''
# Gaussian with sigma = 1e6 (i.e. all values close to 1)
plt.imshow(true_probe.real.cpu(), vmin = 0, vmax = 1, cmap=cmap)
plt.colorbar()
plt.show()
'''

probes, _, centerx, centery, n_probes = create_probes(z_true.view(nx_padded, nx_padded), spacing=spacing, radius=radius,
                                      sigma=1e6, string='grid', dither=3, xedge=120, yedge=120, device=device)

# Uncomment to display illuminated image
'''
for i in range(n_probes):
    fig1 = plt.figure()
    current_img =  probes[i,:,:]*z_true.view(nx_padded,nx_padded)
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

b = create_full_data(z_true, probes, n_probes, nx_padded, nx_padded, n, isBlind=True, device=device)
b_crop = create_cropped_data(z_true, probes, n_probes, nx_padded, nx_padded, N, M, isBlind=True, device=device)


# Perform PFT precomputes
z_temp = probes[0,:,:]*z_true.view(nx_padded,nx_padded)

B, p, q, r = pft2d_configuration(N, M, mu, p, error, device=device)
z, m1_mod, m2_mod, precomputed_prod = pft2d_precompute(z_temp, M, mu, p, q, r, device=device)
print(f"N, M = {N[0], N[1]}, {M[0], M[1]} // p, q, r = {p}, {q}, {r} // e = {error}")

del z_temp

print('\n')

print('Full Data Shape', b[0].view(-1).shape)
print('Cropped Data Shape:', b_crop[0].shape)


#################### ePIE Setup ####################

# Set up for creating inital probe and guess object
aperture_radius = radius 
positions = torch.stack((centerx, centery), dim = 1)

# Get center positions for cropping (should be a 2 by n vector)
pixelPositions, bigx, bigy = convert_to_pixel_positions_testing(positions, little_area=nx_padded)
centerx = torch.round(pixelPositions[:, 1])
centery = torch.round(pixelPositions[:, 0])

# Calculate crop boundaries for the larger area
Y1 = centery - np.floor(nx_padded/2)
Y2 = Y1 + nx_padded
X1 = centerx - np.floor(nx_padded/2)
X2 = X1 + nx_padded

# Create initial probe and guess object
aperture = 2*makeCircleMask(np.round(aperture_radius), imgSize=nx_padded)
mask = makeCircleMask(np.round(aperture_radius), imgSize=nx_padded).to(device)

big_obj = random_complex_guess(bigx, bigy)


# Create unpadded, true image
new_size = (nx, nx)  # New size of the image (width, height)

resize_image(input_path1, output_path1, new_size)
resize_image(input_path2, output_path2, new_size)

im1 = np.array(Image.open('src/baboon_resized.png').convert("L"))
im2 = np.array(Image.open('src/cameraman_resized.png').convert("L"))

r_true = torch.DoubleTensor(im1)
r_true = r_true/torch.max(r_true)
phi_true = torch.DoubleTensor(im2)
phi_true = phi_true/torch.max(phi_true) 
phi_true = (torch.pi / 2) * phi_true  # Adjusted to range between 0 and pi/2 

z_true_unpadded = torch.abs(r_true) * torch.exp(1j * phi_true)

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

ePIE_params = {
    "mask": mask,
    "Y1": Y1,
    "Y2": Y2,
    "X1": X1,
    "X2": X2,
    "z_true": z_true_unpadded,
    "r_true": r_true,
    "phi_true": phi_true,
    "true_probe": true_probe
}

ePIE_PFT_params = {
    "b": b,
    "b_crop": b_crop,
    "mask": mask,
    "Y1": Y1,
    "Y2": Y2,
    "X1": X1,
    "X2": X2,
    "z_true": z_true_unpadded,
    "r_true": r_true,
    "phi_true": phi_true,
    "pft_params": pft_params,
    "true_probe": true_probe
}

crop_image_params = {
    "nx": z_true_unpadded.shape[0],
    "ny": z_true_unpadded.shape[1],
    "nx_padded": z_true.view(nx_padded, nx_padded).shape[0],
    "ny_padded": z_true.view(nx_padded, nx_padded).shape[1]
}
    
if __name__ == '__main__':
    
    #################### Experiments ####################
    
    # Number of random images
    num_images = 150
    
    # Lists to guess images and store relative errors
    guess_images = []
    
    rel_errors_ePIE = []
    mag_rel_errors_ePIE = []
    phase_rel_errors_ePIE = []
    
    mag_ssim_ePIE = []
    phase_ssim_ePIE = []
    mag_psnr_ePIE = []
    phase_psnr_ePIE = []
    
    rel_errors_ePIE_hybrid = []
    mag_rel_errors_ePIE_hybrid = []
    phase_rel_errors_ePIE_hybrid = []
    
    mag_ssim_ePIE_hybrid = []
    phase_ssim_ePIE_hybrid = []
    mag_psnr_ePIE_hybrid = []
    phase_psnr_ePIE_hybrid = []
    
    # Initialize initial probe (always the same)
    probe_guess = aperture.to(device)
    
    for k in range(num_images):
    
        # Generate guess object 
        z_guess = random_complex_guess(bigx, bigy)
        guess_images.append(z_guess)
    
        print("-------------------- Full ePIE --------------------")
        z_ePIE, probe_ePIE = ePIE(z_guess.clone().to(device), probe_guess.clone(), b, lmbda=1e-6, beta_ap=1e1, beta_obj=1e1, num_iters=100, tol=1e-2, **ePIE_params)
        
        full_ePIE_rel_err = rel_error(crop_image(z_ePIE, z_true_unpadded, **crop_image_params), z_true_unpadded)
        rel_errors_ePIE.append(full_ePIE_rel_err)
    
        # Collect Magnitude and Phase (for computing relative errors, ssim, and psnr)
        mag_opt_ePIE = crop_image(torch.abs(z_ePIE), r_true, **crop_image_params)
        mag_opt_ePIE = torch.abs(mag_opt_ePIE).cpu().detach()
        phase_opt_ePIE = crop_image(torch.angle(z_ePIE), phi_true, **crop_image_params)
        phase_opt_ePIE = phase_opt_ePIE.cpu().detach()
        
        mag_ePIE_rel_err = rel_error(mag_opt_ePIE, r_true)
        mag_rel_errors_ePIE.append(mag_ePIE_rel_err)
        
        phase_ePIE_rel_err = rel_error(phase_opt_ePIE, phi_true)
        phase_rel_errors_ePIE.append(phase_ePIE_rel_err)
    
        # Compute and Save SSIM 
        mag_ssim = ssim(mag_opt_ePIE.numpy(), r_true.cpu().detach().numpy(), data_range=r_true.cpu().detach().numpy().max()-r_true.cpu().detach().numpy().min())
        phase_ssim = ssim(phase_opt_ePIE.numpy(), phi_true.cpu().detach().numpy(), data_range=phi_true.cpu().detach().numpy().max()-
                          phi_true.cpu().detach().numpy().min())
    
        mag_ssim_ePIE.append(mag_ssim)
        phase_ssim_ePIE.append(phase_ssim)
        
        # Compute and Save PSNR
        mag_psnr = psnr(interpolate_image_for_psnr(mag_opt_ePIE).numpy(), interpolate_image_for_psnr(r_true).numpy())
        phase_psnr = psnr(interpolate_image_for_psnr(phase_opt_ePIE).numpy(), interpolate_image_for_psnr(phi_true).numpy())
    
        mag_psnr_ePIE.append(mag_psnr)
        phase_psnr_ePIE.append(phase_psnr)
        
        print('Full Rel Err:', "{:5.2e}".format(full_ePIE_rel_err), 
              'Magnitude Rel Err:', "{:5.2e}".format(mag_ePIE_rel_err), 'Phase Rel Err:', "{:5.2e}".format(phase_ePIE_rel_err))
    
        print('\n')
        
        print("-------------------- Warm Start ePIE --------------------")
        print('ePIE PFT')
        z_ePIE_PFT, probe_ePIE_PFT = ePIE_PFT(z_guess.clone().to(device), probe_guess.clone(), lmbda=100, beta_ap=2e-2, beta_obj=2e-2, num_iters=10, tol=1.0, 
                                              **ePIE_PFT_params)
        print('\n')
        print('Full ePIE')
        z_optHybrid, probe_optHybrid = ePIE(z_ePIE_PFT.clone(), probe_ePIE_PFT.clone(), b, lmbda=1e-6, beta_ap=2, beta_obj=1e1, num_iters=100, tol=1e-2, **ePIE_params)
        
        full_hybrid_rel_err = rel_error(crop_image(z_optHybrid, z_true_unpadded, **crop_image_params), z_true_unpadded)
        rel_errors_ePIE_hybrid.append(full_hybrid_rel_err)
    
        # Collect (Hybrid) Magnitude and Phase (for computing relative errors, ssim and psnr)
        mag_opt = crop_image(torch.abs(z_optHybrid), r_true, **crop_image_params)
        mag_opt = torch.abs(mag_opt).cpu().detach()
        phase_opt = crop_image(torch.angle(z_optHybrid), phi_true, **crop_image_params)
        phase_opt = phase_opt.cpu().detach()
        
        mag_hybrid_rel_err = rel_error(mag_opt, r_true)
        mag_rel_errors_ePIE_hybrid.append(mag_hybrid_rel_err)
        
        phase_hybrid_rel_err = rel_error(phase_opt, phi_true)
        phase_rel_errors_ePIE_hybrid.append(phase_hybrid_rel_err)
    
        # Compute and Save SSIM 
        mag_ssim = ssim(mag_opt.numpy(), r_true.cpu().detach().numpy(), data_range=r_true.cpu().detach().numpy().max()-r_true.cpu().detach().numpy().min())
        phase_ssim = ssim(phase_opt.numpy(), phi_true.cpu().detach().numpy(), data_range=phi_true.cpu().detach().numpy().max()-phi_true.cpu().detach().numpy().min())
    
        mag_ssim_ePIE_hybrid.append(mag_ssim)
        phase_ssim_ePIE_hybrid.append(phase_ssim)
        
        # Compute and Save PSNR
        mag_psnr = psnr(interpolate_image_for_psnr(mag_opt).numpy(), interpolate_image_for_psnr(r_true).numpy())
        phase_psnr = psnr(interpolate_image_for_psnr(phase_opt).numpy(), interpolate_image_for_psnr(phi_true).numpy())
    
        mag_psnr_ePIE_hybrid.append(mag_psnr)
        phase_psnr_ePIE_hybrid.append(phase_psnr)
        
        print('(Hybrid) Full Rel Err:', "{:5.2e}".format(full_hybrid_rel_err), 
              'Magnitude Rel Err:', "{:5.2e}".format(mag_hybrid_rel_err), 'Phase Rel Err:', "{:5.2e}".format(phase_hybrid_rel_err))
        
        # Clear the output
        clear_output(wait=True)
        time.sleep(3)
        print(f"Processed {k+1}/{num_images} images...")
        print('len(guess_images):', len(guess_images), 'len(rel_errors_ePIE):', len(rel_errors_ePIE), 'len(rel_errors_ePIE_hybrid):', len(rel_errors_ePIE_hybrid)) 
        print('\n')
    
    # Save Data
    file_name = 'results/blind_rel_error_results/ePIE_rel_error_data.pt'
    state = {
        'guess_images': guess_images,
        
        'rel_errors_ePIE': rel_errors_ePIE,
        'mag_rel_errors_ePIE': mag_rel_errors_ePIE,
        'phase_rel_errors_ePIE': phase_rel_errors_ePIE,
    
        'mag_ssim_ePIE': mag_ssim_ePIE,
        'phase_ssim_ePIE': phase_ssim_ePIE,
        'mag_psnr_ePIE': mag_psnr_ePIE,
        'phase_psnr_ePIE': phase_psnr_ePIE,
        
        'rel_errors_ePIE_hybrid': rel_errors_ePIE_hybrid,
        'mag_rel_errors_ePIE_hybrid': mag_rel_errors_ePIE_hybrid,
        'phase_rel_errors_ePIE_hybrid': phase_rel_errors_ePIE_hybrid,
        
        'mag_ssim_ePIE_hybrid': mag_ssim_ePIE_hybrid,
        'phase_ssim_ePIE_hybrid': phase_ssim_ePIE_hybrid,
        'mag_psnr_ePIE_hybrid': mag_psnr_ePIE_hybrid,
        'phase_psnr_ePIE_hybrid': phase_psnr_ePIE_hybrid
    }
    torch.save(state, file_name)
    print('files saved to ' + file_name)
    
    
    fig = plt.figure()
    # Plot histograms of full relative errors
    plt.hist(rel_errors_ePIE, bins=20, alpha=0.5, label='ePIE', edgecolor='black')
    plt.hist(rel_errors_ePIE_hybrid, bins=20, alpha=0.5, label='Hybrid ePIE', edgecolor='black')
    plt.xlabel('Relative Error')
    plt.ylabel('Frequency')
    plt.title('Full Relative Errors in Reconstructed Images')
    plt.legend(loc='upper right')
    plt.show()
    
    fig = plt.figure()
    # Plot histograms of magnitude relative errors
    plt.hist(mag_rel_errors_ePIE, bins=20, alpha=0.5, label='ePIE', edgecolor='black')
    plt.hist(mag_rel_errors_ePIE_hybrid, bins=20, alpha=0.5, label='Hybrid ePIE', edgecolor='black')
    plt.xlabel('Relative Error')
    plt.legend(loc='upper right')
    plt.show()
    
    fig = plt.figure()
    # Plot histograms of phase relative errors
    plt.hist(phase_rel_errors_ePIE, bins=20, alpha=0.5, label='ePIE', edgecolor='black')
    plt.hist(phase_rel_errors_ePIE_hybrid, bins=20, alpha=0.5, label='Hybrid ePIE', edgecolor='black')
    plt.xlabel('Relative Error')
    plt.legend(loc='upper right')
    plt.show()
    
    
    fig = plt.figure()
    # Plot histograms of magnitude SSIM
    plt.hist(mag_ssim_ePIE, bins=20, alpha=0.5, label='ePIE', edgecolor='black')
    plt.hist(mag_ssim_ePIE_hybrid, bins=20, alpha=0.5, label='Hybrid ePIE', edgecolor='black')
    plt.xlabel('SSIM')
    plt.legend(loc='upper right')
    plt.show()
    
    fig = plt.figure()
    # Plot histograms of phase SSIM
    plt.hist(phase_ssim_ePIE, bins=20, alpha=0.5, label='ePIE', edgecolor='black')
    plt.hist(phase_ssim_ePIE_hybrid, bins=20, alpha=0.5, label='Hybrid ePIE', edgecolor='black')
    plt.xlabel('SSIM')
    plt.legend(loc='upper right')
    plt.show()
    
    fig = plt.figure()
    # Plot histograms of magnitude PSNR
    plt.hist(mag_psnr_ePIE, bins=20, alpha=0.5, label='ePIE', edgecolor='black')
    plt.hist(mag_psnr_ePIE_hybrid, bins=20, alpha=0.5, label='Hybrid ePIE', edgecolor='black')
    plt.xlabel('PSNR')
    plt.legend(loc='upper right')
    plt.show()
    
    fig = plt.figure()
    # Plot histograms of phase PSNR
    plt.hist(phase_psnr_ePIE, bins=20, alpha=0.5, label='ePIE', edgecolor='black')
    plt.hist(phase_psnr_ePIE_hybrid, bins=20, alpha=0.5, label='Hybrid ePIE', edgecolor='black')
    plt.xlabel('PSNR')
    plt.legend(loc='upper right')
    plt.show()

