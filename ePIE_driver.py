from src.create_probes import *
from src.PFT2D import *
from src.create_data import *
from src.blind_ptychography_functions import *

import torch
import torch.fft as fft
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
import math
import time

torch.set_default_dtype(torch.float64)

# Set random seed for reproducibility
torch.manual_seed(42)

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

# SET SIZE OF IMAGE
nx = 8200 # so that it is divisible by p = 64
new_size = (nx, nx)  # New size of the image (width, height)

# Resize the images to the specified size
resize_image(input_path1, output_path1, new_size)
resize_image(input_path2, output_path2, new_size)

# Load and convert resized images to grayscale numpy arrays
im1 = np.array(Image.open(output_path1).convert("L"))
im2 = np.array(Image.open(output_path2).convert("L"))

print('nx = ', nx)
if nx == 512:
    im1 = np.pad(im1, ((128, 128), (128, 128)), mode='constant', constant_values=0)
    im2 = np.pad(im2, ((128, 128), (128, 128)), mode='constant', constant_values=0) 
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

# Compute complex-valued image using magnitude and phase
z_true = torch.abs(r_true) * torch.exp(1j * phi_true)
print('shape of z_true = ', z_true.shape)
nx_padded = z_true.shape[0] # Size of the padded image
z_true = z_true.view(-1, 1).to(device)

n = z_true.shape[0]
m = n

# SET PFT PARAMETERS (SET M AND p VALUES HERE)
# NOTE: The PFT will act on the padded image
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

# Uncomment to display each probe applied to the image
'''
for i in range(n_probes):
    fig1 = plt.figure()
    current_img =  probes[i,:,:]*z_true.view(nx_padded,nx_padded)
    # plt.scatter(centery[i], centerx[i], color = 'red')
    plt.imshow(current_img.imag.cpu(), cmap=cmap)
    plt.show()
    # save_str = 'probed_reconstruction' + str(i+1) + '.pdf'
    # fig1.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
'''

# Compute percentage of overlap between probes
def calculate_overlap_percentage(probe1, probe2):
    overlap = torch.sum(probe1 * probe2)
    total_area = torch.sum(probe1)
    return (overlap / total_area).item() * 100.0

overlap_percentages = torch.zeros(n_probes, device=device)

for i in range(n_probes - 1):
    overlap_percentage = calculate_overlap_percentage(probes[i], probes[i + 1])
    overlap_percentages[i] = overlap_percentage

# Should be roughly 50% overlap
for i in range(n_probes - 1):
    print(f"Overlap percentage between probe {i} and probe {i + 1}: {overlap_percentages[i]:.2f}%")


#################### Create Observed Data ####################

b = create_full_data(z_true, probes, n_probes, nx_padded, nx_padded, n, isBlind=True, device=device)
b_crop = create_cropped_data(z_true, probes, n_probes, nx_padded, nx_padded, N, M, isBlind=True, device=device)


# Perform all precomputations
z_temp = probes[0,:,:]*z_true.view(nx_padded,nx_padded)

B, p, q, r = pft2d_configuration(N, M, mu, p, error, device=device)
z, m1_mod, m2_mod, precomputed_prod = pft2d_precompute(z_temp, M, mu, p, q, r, device=device)

# Display PFT configuration details
print(f"N, M = {N[0], N[1]}, {M[0], M[1]} // p, q, r = {p}, {q}, {r} // e = {error}")

del z_temp

print('\n')

print('Full Data Shape', b[0].view(-1).shape)
print('Cropped Data Shape:', b_crop[0].shape)


#################### ePIE Setup ####################

# Set the radius for the aperture used in ePIE
aperture_radius = radius 

# Stack the x and y positions of the probe centers into a single tensor
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
big_obj = random_complex_guess(bigx, bigy)

# Create a mask for the update step of probe (zeros out imaginary part of learned probe)
mask = makeCircleMask(np.round(aperture_radius), imgSize=nx_padded).to(device)

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
    
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 
    # EXPERIMENTS
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    print('using pattern_match')
    print('-------------------------------------------------------- STARTING FULL ePIE ----------------------------------------------------')
    tv_lambda = 0.0
    beta_ap = 1e1
    beta_obj = 1e1
    num_iters = 50
    tol = 1e-2
    
    print('tv_lambda = ', tv_lambda, 'beta_ap = ', beta_ap, 'beta_obj = ', beta_obj, 'num_iters = ', num_iters, 'tol = ', tol)
    
    z_guess = big_obj.to(device)
    probe_guess = aperture.to(device)
    z_ePIE, probe_ePIE, f_val_hist, grad_z_hist, grad_Q_hist, rel_err_hist, cauchy_err_hist, time_hist, SSIM_mag_hist, SSIM_phase_hist, PSNR_mag_hist, PSNR_phase_hist, phase_err_hist, mag_err_hist, probe_err_hist = ePIE(z_guess.clone(), probe_guess.clone(), b=b, lmbda=tv_lambda, beta_ap=beta_ap, beta_obj=beta_obj, num_iters=num_iters, tol=tol, return_all_metrics=True, **ePIE_params)
    
    
    fig = plt.figure()
    plt.imshow(z_ePIE.imag.cpu(), vmin=0, vmax=1, cmap=cmap)
    plt.colorbar()
    plt.show()
    save_str = 'results/large_scale_ePIE_results/full_reconstruction_uncropped.png'
    fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
    
    fig = plt.figure()
    plt.imshow(crop_image(z_ePIE, z_true_unpadded, **crop_image_params).imag.cpu(), cmap=cmap)
    plt.colorbar()
    plt.show()
    save_str = 'results/large_scale_ePIE_results/full_reconstruction_cropped.png'
    fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
    
    
    fig = plt.figure()
    plt.imshow(torch.abs(z_ePIE).cpu(), vmin=0, vmax=1, cmap=cmap)
    plt.colorbar()
    plt.show()
    save_str = 'results/large_scale_ePIE_results/full_magnitude_uncropped.png'
    fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
    
    fig = plt.figure()
    plt.imshow(crop_image(torch.abs(z_ePIE), r_true, **crop_image_params).cpu(), vmin=0, vmax=1, cmap=cmap)
    plt.colorbar()
    plt.show()
    save_str = 'results/large_scale_ePIE_results/full_magnitude_cropped.png'
    fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
    
    
    fig = plt.figure()
    plt.imshow(torch.angle(z_ePIE).cpu(), vmin=0, vmax=torch.pi/2, cmap=cmap)
    plt.colorbar()
    plt.show()
    save_str = 'results/large_scale_ePIE_results/full_phase_uncropped.png'
    fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
    
    fig = plt.figure()
    plt.imshow(crop_image(torch.angle(z_ePIE), phi_true, **crop_image_params).cpu(), vmin=0, vmax=torch.pi/2, cmap=cmap)
    plt.colorbar()
    plt.show()
    save_str = 'results/large_scale_ePIE_results/full_phase_cropped.png'
    fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
    
    
    fig = plt.figure()
    plt.imshow(torch.abs(probe_ePIE).cpu(), cmap=cmap)
    plt.colorbar()
    plt.show()
    save_str = 'results/large_scale_ePIE_results/full_probe_mag.png'
    fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
    
    fig = plt.figure()
    plt.imshow(torch.angle(probe_ePIE).cpu(), cmap=cmap)
    plt.colorbar()
    plt.show()
    save_str = 'results/large_scale_ePIE_results/full_probe_phase.png'
    fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
    
    
    file_name = 'results/large_scale_ePIE_results/full_ePIE_hist.pt'
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
        'phase_err_hist': phase_err_hist
    }
    torch.save(state, file_name)
    print('files saved to ' + file_name)
    
    # Uncomment to delete results after saving, if needed
    # del z_ePIE, probe_ePIE
    
    
    print('\n\n-------------------------------------------------------- STARTING PFT ePIE with probe errs ----------------------------------------------------')
    tv_lambda_PFT= 1e3
    beta_ap_PFT = 2e-3
    beta_obj_PFT = 2e-3
    num_iters_PFT = 10
    tol_PFT = 1.0
    
    assert probe_guess.type() == true_probe.type()
    assert probe_guess.shape == true_probe.shape
    print('probe error = ', torch.norm(probe_guess - true_probe))
    
    print('tv_lambda_PFT = ', tv_lambda_PFT, 'beta_ap_PFT = ', beta_ap_PFT, 'beta_obj_PFT = ', beta_obj_PFT, 'num_iters_PFT = ', num_iters_PFT, 'tol = ', tol_PFT)
    
    z_ePIE_PFT, probe_ePIE_PFT, f_val_hist_PFT, grad_z_hist_PFT, grad_Q_hist_PFT, rel_err_hist_PFT, cauchy_err_hist_PFT, time_hist_PFT, SSIM_mag_hist_PFT, SSIM_phase_hist_PFT, PSNR_mag_hist_PFT, PSNR_phase_hist_PFT, phase_err_hist_PFT, mag_err_hist_PFT, probe_err_hist_PFT = ePIE_PFT(z_guess.clone(), probe_guess.clone(), lmbda=tv_lambda_PFT, beta_ap=beta_ap_PFT, beta_obj=beta_obj_PFT, num_iters=num_iters_PFT, tol=tol_PFT, return_all_metrics=True, **ePIE_PFT_params)
    
    
    fig = plt.figure()
    plt.imshow(z_ePIE_PFT.imag.cpu(), vmin=0, vmax=1, cmap=cmap)
    plt.colorbar()
    plt.show()
    save_str = 'results/large_scale_ePIE_results/PFT_reconstruction_uncropped.png'
    fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
    
    fig = plt.figure()
    plt.imshow(crop_image(z_ePIE_PFT, z_true_unpadded, **crop_image_params).imag.cpu(), vmin=0, vmax=1, cmap=cmap)
    plt.colorbar()
    plt.show()
    save_str = 'results/large_scale_ePIE_results/PFT_reconstruction_cropped.png'
    fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
    
    
    fig = plt.figure()
    plt.imshow(torch.abs(z_ePIE_PFT).cpu(), vmin=0, vmax=1, cmap=cmap)
    plt.colorbar()
    plt.show()
    save_str = 'results/large_scale_ePIE_results/PFT_magnitude_uncropped.png'
    fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
    
    fig = plt.figure()
    plt.imshow(crop_image(torch.abs(z_ePIE_PFT), r_true, **crop_image_params).cpu(), vmin=0, vmax=1, cmap=cmap)
    plt.colorbar()
    plt.show()
    save_str = 'results/large_scale_ePIE_results/PFT_magnitude_cropped.png'
    fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
    
    
    fig = plt.figure()
    plt.imshow(torch.angle(z_ePIE_PFT).cpu(), cmap=cmap)
    plt.colorbar()
    plt.show()
    save_str = 'results/large_scale_ePIE_results/PFT_phase_uncropped.png'
    fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
    
    fig = plt.figure()
    plt.imshow(crop_image(torch.angle(z_ePIE_PFT), phi_true, **crop_image_params).cpu(), cmap=cmap)
    plt.colorbar()
    plt.show()
    save_str = 'results/large_scale_ePIE_results/PFT_phase_cropped.png'
    fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
    
    
    fig = plt.figure()
    plt.imshow(torch.abs(probe_ePIE_PFT).cpu(), cmap=cmap)
    plt.colorbar()
    plt.show()
    save_str = 'results/large_scale_ePIE_results/PFT_probe_mag.png'
    fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
    
    fig = plt.figure()
    plt.imshow(torch.angle(probe_ePIE_PFT).cpu(), cmap=cmap)
    plt.colorbar()
    plt.show()
    save_str = 'results/large_scale_ePIE_results/PFT_probe_phase.png'
    fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
    
    
    file_name = 'results/large_scale_ePIE_results/ePIE_PFT_hist.pt'
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
        'phase_err_hist': phase_err_hist_PFT
    }
    torch.save(state, file_name)
    print('files saved to ' + file_name)
    
    
    print('\n\n-------------------------------------------------------- STARTING Warmstarted ePIE ----------------------------------------------------')
    # Using same parameters as full ePIE
    tv_lambda = tv_lambda
    beta_ap = beta_ap
    beta_obj = beta_obj
    num_iters = num_iters
    tol = tol
    
    print('tv_lambda = ', tv_lambda, 'beta_ap = ', beta_ap, 'beta_obj = ', beta_obj, 'num_iters = ', num_iters, 'tol = ', tol)
    
    z_optHybrid, probe_optHybrid, f_val_hist_hybrid, grad_z_hist_hybrid, grad_Q_hist_hybrid, rel_err_hist_hybrid, cauchy_err_hist_hybrid, time_hist_hybrid, SSIM_mag_hist_hybrid, SSIM_phase_hist_hybrid, PSNR_mag_hist_hybrid, PSNR_phase_hist_hybrid, phase_err_hist_hybrid, mag_err_hist_hybrid, probe_err_hist_hybrid = ePIE(z_ePIE_PFT.clone(), probe_ePIE_PFT, b=b, lmbda=tv_lambda, beta_ap=beta_ap, beta_obj=beta_obj, num_iters=num_iters, tol=tol, return_all_metrics=True, **ePIE_params)
    
    
    fig = plt.figure()
    plt.imshow(z_optHybrid.imag.cpu(), vmin=0, vmax=1, cmap=cmap)
    plt.colorbar()
    plt.show()
    save_str = 'results/large_scale_ePIE_results/hybrid_reconstruction_uncropped.png'
    fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
    
    fig = plt.figure()
    plt.imshow(crop_image(z_optHybrid, z_true_unpadded, **crop_image_params).imag.cpu(), vmin=0, vmax=1, cmap=cmap)
    plt.colorbar()
    plt.show()
    save_str = 'results/large_scale_ePIE_results/hybrid_reconstruction_cropped.png'
    fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
    
    
    fig = plt.figure()
    plt.imshow(torch.abs(z_optHybrid).cpu(), vmin=0, vmax=1, cmap=cmap)
    plt.colorbar()
    plt.show()
    save_str = 'results/large_scale_ePIE_results/hybrid_magnitude_uncropped.png'
    fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
    
    fig = plt.figure()
    plt.imshow(crop_image(torch.abs(z_optHybrid), r_true, **crop_image_params).cpu(), vmin=0, vmax=1, cmap=cmap)
    plt.colorbar()
    plt.show()
    save_str = 'results/large_scale_ePIE_results/hybrid_magnitude_cropped.png'
    fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
    
    
    fig = plt.figure()
    plt.imshow(torch.angle(z_optHybrid).cpu(), vmin=0, vmax=torch.pi/2, cmap=cmap)
    plt.colorbar()
    plt.show()
    save_str = 'results/large_scale_ePIE_results/hybrid_phase_uncropped.png'
    fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
    
    fig = plt.figure()
    plt.imshow(crop_image(torch.angle(z_optHybrid), phi_true, **crop_image_params).cpu(), vmin=0, vmax=torch.pi/2, cmap=cmap)
    plt.colorbar()
    plt.show()
    save_str = 'results/large_scale_ePIE_results/hybrid_phase_cropped.png'
    fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
    
    
    fig = plt.figure()
    plt.imshow(torch.abs(probe_optHybrid).cpu(), cmap=cmap)
    plt.colorbar()
    plt.show()
    save_str = 'results/large_scale_ePIE_results/hybrid_probe_mag.png'
    fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
    
    fig = plt.figure()
    plt.imshow(torch.angle(probe_optHybrid).cpu(), cmap=cmap)
    plt.colorbar()
    plt.show()
    save_str = 'results/large_scale_ePIE_results/hybrid_probe_phase.png'
    fig.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
    
    
    file_name = 'results/large_scale_ePIE_results/ePIE_hybrid_hist.pt'
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
        'phase_err_hist': phase_err_hist_hybrid
    }
    torch.save(state, file_name)
    print('files saved to ' + file_name)

