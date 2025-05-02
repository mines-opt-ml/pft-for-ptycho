import torch
import torch.fft as fft

from skimage.feature import match_template
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import numpy as np
import matplotlib.pyplot as plt
import time

from src.PFT2D import *

# For plotting images
cmap='gray'

#################### FFT SHIFT FUNCTIONS ####################
def fftshift(x):
    """
    Shifts the zero-frequency component to the center of a PyTorch Tensor.
    
    Parameters:
        x (Tensor): Input tensor to be shifted.
    
    Returns:
        Tensor: The shifted tensor with zero-frequency component centered.
    """
    dim = len(x.shape)
    shift = [dim // 2 for dim in x.shape]
    return torch.roll(x, shift, tuple(range(dim)))

def ifftshift(x):
    """
    Inversely shifts the zero-frequency component to the original position for PyTorch Tensors.
    
    Parameters:
        x (Tensor): Input tensor.
    
    Returns:
        Tensor: The tensor with zero-frequency component shifted back to the original position.
    """
    dim = len(x.shape)
    shift = [-dim // 2 for dim in x.shape]
    return torch.roll(x, shift, tuple(range(dim)))


#################### SET UP FUNCTIONS ####################
def random_complex_vector(nx, ny):
    # Generate magnitude and phase
    magnitude = torch.abs(torch.rand(nx, ny))
    phase = (torch.pi / 2) * torch.rand(nx, ny)
    # Combine real and imaginary parts to create complex numbers
    complex_vector = magnitude * torch.exp(1j * phase)
    return complex_vector.view(-1)
    
def rel_error(z_opt, z_true, n):
    return (torch.norm(z_true.view(n, 1).cpu() - z_opt.view(n, 1).cpu()) / torch.norm(z_true.view(n, 1).cpu())).item()

def interpolate_image_for_psnr(image):
    # Interpolate the image to [0,1]
    min_val = image.min()
    max_val = image.max()
    interpolated_image = (image - min_val) / (max_val - min_val)

    # Ensure the image values are in the range [0, 1]
    interpolated_image = torch.clamp(interpolated_image, 0, 1)

    return interpolated_image

#################### EVALUATION FUNCTIONS ####################

def total_variation(x):
    """
    Computes the total variation (TV) of a 2D tensor, i.e. the sum of absolute differences in the horizontal and vertical directions.

    Parameters:
        x (torch.Tensor): A 2D tensor representing the image or data for which the total variation is calculated.

    Returns:
        tv (torch.Tensor): The total variation value, representing the L1 norm of the gradient of the tensor.
    """
    
    # Calculate the difference in the horizontal direction
    horizontal_diff = x[:, :-1] - x[:, 1:]
    
    # Calculate the difference in the vertical direction
    vertical_diff = x[:-1, :] - x[1:, :]
    
    # Compute the L1 norm of the gradient (total variation)
    tv = torch.sum(torch.abs(horizontal_diff)) + torch.sum(torch.abs(vertical_diff))
    
    return tv
    

def f_original(z, b, probes, nx, ny, return_gradient=False):
    """
    Computes the objective function value and optionally its gradient with respect to the feature vector `z` for the Ptychographic Iterative Engine (PIE).

    Parameters:
        z (torch.Tensor): The full image or object being reconstructed.
        b (torch.Tensor): The measured diffraction intensities for each probe position.
        probes (torch.Tensor): The probe functions used to scan the sample, with shape (n_probes, nx, ny).
        nx (int): The number of pixels in the x-dimension.
        ny (int): The number of pixels in the y-dimension.
        return_gradient (bool, optional): If True, also returns the gradient with respect to `z`. Defaults to False.

    Returns:
        torch.Tensor: The average objective function value over all probes.
        If `return_gradient` is True, also returns:
            - grad_f (torch.Tensor): The gradient of the objective function with respect to `z`.
    """
    
    device = z.device

    n_probes = b.shape[0]
    n = z.shape[1]

    if return_gradient==True:
        z.requires_grad = True

    # shape is now n by n_samples
    z = z.permute(1,0)

    f_val = 0.0
    
    # loop over every probe to evaluate function
    for i in range(n_probes):
        fft_z = torch.fft.fft2(probes[i] * z.view(nx, ny)) # FQz  
        proj_z = b[i].view(nx, ny) * torch.exp(1j*torch.angle(fft_z))
        
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


def f_single_probe(z, b, probe, nx, ny, n_probes, return_gradient=False, lmbda=1e-1):
    """
    Computes the objective function value and optionally its gradient with respect to the feature vector `z` for a single probe in the Ptychographic Iterative
    Engine (PIE).

    Parameters:
        z (torch.Tensor): The full image or object being reconstructed.
        b (torch.Tensor): The measured diffraction intensities for each probe position.
        probe (torch.Tensor): The probe function used to scan the sample, with shape (nx, ny).
        nx (int): The number of pixels in the x-dimension.
        ny (int): The number of pixels in the y-dimension.
        n_probes (int): The number of probes used.
        return_gradient (bool, optional): If True, also returns the gradient with respect to `z`. Defaults to False.
        lmbda (float, optional): The regularization parameter for total variation. Defaults to 1e-1.

    Returns:
        torch.Tensor: The average (regularized) objective function value for the single probe application.
        If `return_gradient` is True, also returns:
            - grad_f (torch.Tensor): The gradient of the regularized objective function with respect to `z`.
    """
    
    device = z.device

    n_samples = z.view(1,-1).shape[0]
    n = z.view(1,-1).shape[1]

    if return_gradient==True:
        z.requires_grad = True

    # shape is now n by n_samples
    # z = z.permute(1,0)

    f_val = 0.0

    fft_z = torch.fft.fft2(probe * z.view(nx, ny)) # FQz 
    proj_z = b.view(nx, ny) * torch.exp(1j*torch.angle(fft_z))
    
    f_val = f_val + 0.5*torch.norm(torch.fft.ifft2(proj_z - fft_z))**2

    # Add total variation
    f_val_regularized = (f_val + lmbda * total_variation(z))/n_probes

    if return_gradient==False:
        return f_val
    else:
        grad_f = torch.autograd.grad(f_val_regularized, z,
                                     grad_outputs=torch.ones(f_val_regularized.shape, device=device), retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False, is_grads_batched=False)[0] 
        grad_f = grad_f.detach()
        f_val_regularized = f_val.detach()
        z = z.detach()
    return f_val, grad_f


def f_pft(z, b, probe, nx, ny, n_probes, lmbda = 0.1, return_gradient=False, B=None, M=None, p=None, q=None,
      m1_mod=None, m2_mod=None, precomputed_prod=None):
    """
    Computes the objective function value using the Partial Fourier Transform (PFT) and optionally its gradient with respect to the feature vector `z`.

    Parameters:
        z (torch.Tensor): The full image or object being reconstructed.
        b (torch.Tensor): The measured diffraction intensities for each probe position.
        probe (torch.Tensor): The probe function used to scan the sample, with shape (nx, ny).
        nx (int): The number of pixels in the x-dimension.
        ny (int): The number of pixels in the y-dimension.
        n_probes (int): The number of probes used.
        lmbda (float, optional): The regularization parameter for total variation. Defaults to 0.1.
        return_gradient (bool, optional): If True, also returns the gradient with respect to `z`. Defaults to False.
        B (list of torch.Tensor): Precomputed B matrices (B1 and B2) used in the PFT.
        M (optional): Parameters for PFT computation (not detailed in the function).
        p (list of int): Divisors p1 and p2 of N1 and N2 (size of image), respectively.
        q (list of int): Quotients of N and p for each dimension.
        m1_mod (torch.Tensor): Modulo values for the first dimension.
        m2_mod (torch.Tensor): Modulo values for the second dimension.
        precomputed_prod (torch.Tensor): Precomputed product of exponential terms and powers for both dimensions.

    Returns:
        torch.Tensor: The regularized objective function value computed using PFT.
        If `return_gradient` is True, also returns:
            - grad_f (torch.Tensor): The gradient of the regularized objective function with respect to `z`.
    """
    
    device = z.device

    if return_gradient==True:
        z.requires_grad = True

    # shape is now n by n_samples
    # z = z.permute(1,0)

    f_val = 0.0

    z_temp = probe.view(nx, ny) * z.view(nx, ny)
    z_temp = z_temp.view(p[0], q[0], p[1], q[1]).permute(0, 2, 1, 3).contiguous().view(p[0], p[1], q[0], q[1])
    pft_z = pft2d_computation(z_temp, B, m1_mod, m2_mod, precomputed_prod, device=device)

    proj_z = b.view(2*M[0], 2*M[1]) * torch.exp(1j*torch.angle(ifftshift(pft_z)))
    
    f_val = f_val + 0.5*torch.norm(torch.fft.ifft2(proj_z - fftshift(pft_z)))**2
    
    # Add total variation
    f_val_regularized = (f_val + lmbda * total_variation(z))/n_probes
    
    if return_gradient==False:
        return f_val
    else:
        grad_f = torch.autograd.grad(f_val_regularized, z,
                                     grad_outputs=torch.ones(f_val_regularized.shape, device=device), retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False, is_grads_batched=False)[0] 
        grad_f = grad_f.detach()
        f_val_regularized = f_val.detach()
        z = z.detach()
    return f_val, grad_f


#################### PIE FUNCTIONS ####################
def PIE(zk, alpha = 1, num_iters = 1000, tol = 1e-1, tv_lmbda=1e-1, return_all_metrics=False,
        b=None, probes=None, nx=None, ny=None, z_true=None, r_true=None, phi_true=None):
    """
    Performs the Ptychographic Iterative Engine (PIE) algorithm to reconstruct an image.

    Parameters:
        zk (torch.Tensor): The initial estimate of the image to be reconstructed.
        alpha (float, optional): The step size for gradient descent. Defaults to 1.
        num_iters (int, optional): The maximum number of iterations. Defaults to 1000.
        tol (float, optional): The tolerance for convergence based on the Cauchy error. Defaults to 1e-1.
        tv_lmbda (float, optional): The regularization parameter for total variation. Defaults to 1e-1.
        return_all_metrics (bool, optional): If True, returns additional metrics collected during the iteration. Defaults to False.
        b (torch.Tensor, optional): The measured diffraction intensities for each probe position of size (n_probes, nx * ny).
        probes (torch.Tensor): The probe functions used to scan the sample, each of size (nx, ny).
        nx (int, optional): The number of pixels in the x-dimension.
        ny (int, optional): The number of pixels in the y-dimension.
        z_true (torch.Tensor, optional): The ground truth image for evaluation purposes.
        r_true (torch.Tensor, optional): The ground truth magnitude for evaluation purposes.
        phi_true (torch.Tensor, optional): The ground truth probe function for evaluation purposes.

    Returns:
        torch.Tensor: The reconstructed image of size (nx, nx).
        If `return_all_metrics` is True, also returns:
            - f_val_hist (list of torch.Tensor): History of objective function values.
            - gradf_val_hist (list of float): History of gradient norms.
            - rel_err_hist (list of float): History of relative errors.
            - cauchy_err_hist (list of float): History of Cauchy errors.
            - time_hist (list of float): History of iteration times.
            - SSIM_mag_hist (list of float): History of SSIM values for magnitude.
            - SSIM_phase_hist (list of float): History of SSIM values for phase.
            - PSNR_mag_hist (list of float): History of PSNR values for magnitude.
            - PSNR_phase_hist (list of float): History of PSNR values for phase.
            - mag_err_hist (list of float): History of magnitude errors.
            - phase_err_hist (list of float): History of phase errors.
    """
    
    device = zk.device
    n = zk.shape[0]
    n_probes = b.shape[0]
    
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

            # Update the object
            fval, gradf_val = f_single_probe(zk.view(1,-1), b = b[i], probe = probes[i], nx=nx, ny=nx, n_probes=n_probes, return_gradient=True) 
            zk = zk - alpha * gradf_val.view(n)
            
        end = time.time()
        iter_time = end - start
        time_hist.append(iter_time)

        rel_err = torch.norm(zk.view(n).cpu() - z_true.view(n).cpu()) / torch.norm(z_true.view(n).cpu())

        # start_time = time.time()
        # fval_full, gradf_val_full = f_original(zk.view(1,-1), b=b, probes=probes, nx=nx, ny=nx, return_gradient=False)
        # end_time = time.time()
        # f_val_time = end_time - start_time

        start_time = time.time()
        fval_full = f_original(zk.view(1,-1), b=b, probes=probes, nx=nx, ny=nx, return_gradient=False) 
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

        if j%10 == 0: 
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

        if j%25 == 0:
            plt.imshow(torch.angle(zk).view(nx, nx).cpu(), vmin=0, vmax=torch.pi/2, cmap=cmap)
            plt.colorbar()
            plt.show()
        
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

    if return_all_metrics:
        return zk.view(nx, nx), f_val_hist, gradf_val_hist, rel_err_hist, cauchy_err_hist, time_hist, SSIM_mag_hist, SSIM_phase_hist, PSNR_mag_hist, PSNR_phase_hist, phase_err_hist, mag_err_hist
            
    return zk.view(nx, nx)


def PIE_PFT(zk, alpha = 1, num_iters = 1000, tol = 1.10e-1, tv_lmbda = 1e-2, return_all_metrics=False,
            b=None, b_crop=None, probes=None, nx=None, ny=None, z_true=None, r_true=None, phi_true=None, pft_params=None):
    """
    Performs the Ptychographic Iterative Engine (PIE) algorithm using the Partial Fourier Transform (PFT) to reconstruct an image.

    Parameters:
        zk (torch.Tensor): The initial estimate of the image to be reconstructed.
        alpha (float, optional): The step size for gradient descent. Defaults to 1.
        num_iters (int, optional): The maximum number of iterations. Defaults to 1000.
        tol (float, optional): The tolerance for convergence based on the Cauchy error. Defaults to 1.10e-1.
        tv_lmbda (float, optional): The regularization parameter for total variation. Defaults to 1e-2.
        return_all_metrics (bool, optional): If True, returns additional metrics collected during the iteration. Defaults to False.
        b (torch.Tensor, optional): The measured diffraction intensities for each probe position of size (n_probes, nx * ny).
        b_crop (torch.Tensor, optional): The cropped measured diffraction intensities for PFT of size (n_probes, 4 * M1 * M2).
        probes (list of torch.Tensor, optional): The probe functions used to scan the sample, each of size (nx, ny).
        nx (int, optional): The number of pixels in the x-dimension.
        ny (int, optional): The number of pixels in the y-dimension.
        z_true (torch.Tensor, optional): The ground truth image for evaluation purposes.
        r_true (torch.Tensor, optional): The ground truth magnitude for evaluation purposes.
        phi_true (torch.Tensor, optional): The ground truth probe function for evaluation purposes.
        pft_params (dict, optional): Parameters for the Partial Fourier Transform function.

    Returns:
        torch.Tensor: The reconstructed image of size (nx, nx).
        If `return_all_metrics` is True, also returns:
            - f_val_hist (list of torch.Tensor): History of objective function values.
            - gradf_val_hist (list of float): History of gradient norms.
            - rel_err_hist (list of float): History of relative errors.
            - cauchy_err_hist (list of float): History of Cauchy errors.
            - time_hist (list of float): History of iteration times.
            - SSIM_mag_hist (list of float): History of SSIM values for magnitude.
            - SSIM_phase_hist (list of float): History of SSIM values for phase.
            - PSNR_mag_hist (list of float): History of PSNR values for magnitude.
            - PSNR_phase_hist (list of float): History of PSNR values for phase.
            - mag_err_hist (list of float): History of magnitude errors.
            - phase_err_hist (list of float): History of phase errors.
    """
    
    device = zk.device
    n = zk.shape[0]
    n_probes = b.shape[0]
    
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

    mag_true = torch.Tensor(torch.abs(z_true.view(nx, ny))).cpu()
    phase_true = torch.Tensor(torch.angle(z_true.view(nx, ny))).cpu()


    rel_err = torch.norm(zk.view(n).cpu() - z_true.view(n).cpu()) / torch.norm(z_true.view(n).cpu())
    print('initial rel_err = ', rel_err.cpu())

    while cauchy_err > tol:

        start_time_total = time.time()
        
        z_old = zk.clone().cpu()
        
        start = time.time()
        for i in range(n_probes):

            # Update the object
            fval, gradf_val = f_pft(zk.view(1,-1), b = b_crop[i], probe = probes[i], nx=nx, ny=nx, n_probes=n_probes, lmbda=tv_lmbda, return_gradient=True, **pft_params)  
            zk = zk - alpha * gradf_val.view(n)
            
        end = time.time()
        iter_time = end - start
        time_hist.append(iter_time)

        rel_err = torch.norm(zk.view(n).cpu() - z_true.view(n).cpu()) / torch.norm(z_true.view(n).cpu())

        # start_time = time.time()
        # fval_full, gradf_val_full = f_original(zk.view(1,-1), b=b, probes=probes, nx=nx, ny=nx, return_gradient=False)
        # end_time = time.time()
        # f_val_time = end_time - start_time

        start_time = time.time()
        fval_full = f_original(zk.view(1,-1), b=b, probes=probes, nx=nx, ny=nx, return_gradient=False) 
        end_time = time.time()
        f_val_time = end_time - start_time
        
        
        cauchy_err = torch.norm(z_old - zk.cpu())/(torch.norm(z_old))

        f_val_hist.append(fval_full.cpu())
        # gradf_val_hist.append(torch.norm(gradf_val_full).cpu())
        rel_err_hist.append(rel_err)
        cauchy_err_hist.append(cauchy_err.cpu())

        mag_k = torch.abs(zk.view(nx, ny)).cpu().detach()
        # mag_k_torch = torch.abs(zk.view(nx, ny)).cpu()
        phase_k = torch.angle(zk.view(nx, ny)).cpu().detach()
        # phase_k_torch = torch.angle(zk.view(nx, ny)).cpu()

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

        if j%10 == 0:
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

        if j%25 == 0:
            plt.imshow(torch.angle(zk).view(nx, ny).cpu(), vmin=0, vmax=torch.pi/2, cmap=cmap)
            plt.colorbar()
            plt.show()
        
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


    if return_all_metrics:
        return zk.view(nx, nx), f_val_hist, gradf_val_hist, rel_err_hist, cauchy_err_hist, time_hist, SSIM_mag_hist, SSIM_phase_hist, PSNR_mag_hist, PSNR_phase_hist, phase_err_hist, mag_err_hist
    
    return zk.view(nx, ny)

