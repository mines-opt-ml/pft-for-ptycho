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
def convert_to_pixel_positions_testing(positions, little_area, pixel_size = 1):
    """
    Converts positions from experimental geometry to pixel geometry and adjusts them to be centrosymmetric.

    Parameters:
        positions (torch.Tensor): The real-space positions as a tensor of shape (N, 2), where N is the number of positions.
        little_area (int): The size of the smaller area within the full field of view.
        pixel_size (float, optional): The size of one pixel in real-space units. Defaults to 1.

    Returns:
        tuple: A tuple containing:
            - pixelPositions (torch.Tensor): The converted pixel positions adjusted to be centrosymmetric.
            - bigx (int): The width of the full field of view in pixels.
            - bigy (int): The height of the full field of view in pixels.
    """
    
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

def makeCircleMask(radius, imgSize):
    """
    Create a circular mask for a given radius and image size.

    This function generates a circular mask with a specified radius, centered in an image of
    a given size. The mask is returned as a complex tensor, where the real part is the 
    circular mask, and the imaginary part is zero.

    Parameters:
        radius (int or float): The radius of the circle to be masked.
        imgSize (int): The size of the image (assumed to be square) in which the circle will be centered.

    Returns:
        complex_mask (torch.Tensor): A complex tensor of type `torch.complex128`, where the real part is a circular 
                                     mask of ones (inside the circle) and zeros (outside), and the imaginary part is zero.
    """
    
    nc = imgSize//2 + 1
    n2 = nc - 1
    xx, yy = torch.meshgrid(torch.arange(-n2, n2), torch.arange(-n2, n2), indexing='ij') 
    R = torch.sqrt(xx**2 + yy**2)

    mask = (R <= radius).float()
    
    complex_mask = torch.complex(mask, torch.zeros_like(mask)).to(torch.complex128)

    return complex_mask

def random_complex_guess(bigx, bigy):
    # Generate magnitude and phase
    rand_mag = torch.abs(torch.rand(int(bigx), int(bigy)))
    rand_phase = (torch.pi / 2) * torch.rand(int(bigx), int(bigy)) 
    # Combine real and imaginary parts to create complex numbers
    big_obj = rand_mag * torch.exp(1j * rand_phase)
    
    return big_obj

def rel_error(z_opt, z_true):
    # NOTE: z_opt is assumed to be cropped to the size of z_true_unpadded
    # Compute and return the relative error
    return (torch.norm(z_true.cpu() - z_opt.cpu()) / torch.norm(z_true.cpu())).item()

def crop_image(z_opt, z_true=None, nx=None, ny=None, nx_padded=None, ny_padded=None):
    """
    Crop the reconstructed image to match the size of the true image.

    This function crops the reconstructed object (`z_opt`) to match the size of the 
    true object (`z_true`). The cropping is performed by finding the region in `z_opt` 
    that has the highest correlation with `z_true` and centering the crop around this region.

    Parameters:
        z_opt (torch.Tensor): The reconstructed object to be cropped.
        z_true (torch.Tensor, optional): The true object used as a reference for cropping. 
                                         Defaults to None.
        nx (int, optional): The width of the true image (in pixels). Defaults to None.
        ny (int, optional): The height of the true image (in pixels). Defaults to None.
        nx_padded (int, optional): The width of the padded image (in pixels). Defaults to None.
        ny_padded (int, optional): The height of the padded image (in pixels). Defaults to None.

    Returns:
        object1 (torch.Tensor): The cropped version of `z_opt` that matches the size of `z_true`.
    """
    correlation = match_template(torch.abs(z_opt).cpu().numpy(), torch.abs(z_true).cpu().numpy(), pad_input=True)
    correlation = torch.tensor(correlation)

    # Extract the region with highest correlation (centered around the peak)
    h1 = torch.tensor(z_opt.cpu().numpy().shape) // 2 
    correlation_sub = correlation[h1[0]-int(nx//2) : h1[0]+int(nx//2), h1[1]-int(ny//2) : h1[1]+int(ny//2)]
    max_val = torch.max(correlation_sub).item()
    I = torch.nonzero(correlation == max_val, as_tuple=False)

    # Extract coordinates of the peak
    I1, I2 = I[0].tolist()
    
    # print('I1 = ', I1, ', I2 = ', I2)
    if I1 < 0 or I2 < 0 or I1 >= nx_padded or I2 >= ny_padded:
        print('indices outside image were found in crop_image...')
        object1 = z_opt[h1[0]-int(nx//2) : h1[0]+int(nx//2), h1[1]-int(ny//2) : h1[1]+int(ny//2)] # double check this.
        # object1 = z_opt[I1-int(nx//2) : I1+int(nx//2), I2-int(nx//2) : I2+int(nx//2)]
    else:
        # Extract the aligned object from the big object
        object1 = z_opt[I1-int(nx//2) : I1+int(nx//2), I2-int(ny//2) : I2+int(ny//2)]

    return object1

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
    

def f_original(z, Q, b, Y1, Y2, X1, X2, return_gradient=False, lmbda=1e-1):
    """
    Computes the objective function value and optionally its gradients with respect to a section of the image `z` and the probe `Q`
    for the extended Ptychographic Iterative Engine (ePIE).

    Parameters:
        z (torch.Tensor): The full image or object being reconstructed.
        Q (torch.Tensor): The probe function, which is applied to sections of the image.
        b (torch.Tensor): The measured diffraction intensities for each probe position.
        Y1, Y2 (torch.Tensor): The vertical start and end indices for each probe position.
        X1, X2 (torch.Tensor): The horizontal start and end indices for each probe position.
        return_gradient (bool, optional): If True, also returns the gradients with respect to `z` and `Q`. Defaults to False.
        lmbda (float, optional): The regularization parameter for the total variation. Defaults to 1e-1.

    Returns:
        torch.Tensor: The average objective function value over all probes.
        If `return_gradient` is True, also returns:
            - grad_z (torch.Tensor): The gradient of the objective function with respect to the selected section of `z`.
            - grad_Q (torch.Tensor): The gradient of the objective function with respect to `Q`.
    """
    
    device = z.device

    n_probes = b.shape[0]
    nx = b.shape[1]
    ny = b.shape[2]

    if return_gradient==True:
        z.requires_grad = True
        Q.requires_grad = True

    f_val = 0.0

    # Iterate over each probe
    for i in range(n_probes):
        obj_n = z[int(Y1[i]) : int(Y2[i]), int(X1[i]) : int(X2[i])].clone()
        fft_zn = torch.fft.fft2(Q*obj_n)
        proj_zn = b[i].view(nx, ny) * torch.exp(1j*torch.angle(fft_zn))
    
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


def f_single_probe(z, Q, b, current_ap, lmbda = 0, return_gradient=False):
    """
    Computes the objective function value and optionally its gradients for a single probe application.

    Parameters:
        z (torch.Tensor): The section of the image being probed.
        Q (torch.Tensor): The probe function applied to the section of the image.
        b (torch.Tensor): The measured diffraction intensities for each probe position.
        current_ap (int): The index of the current probe application.
        lmbda (float, optional): The regularization parameter for the total variation. Defaults to 0.
        return_gradient (bool, optional): If True, also returns the gradients with respect to `z` and `Q`. Defaults to False.

    Returns:
        torch.Tensor: The average objective function value for the single probe application.
        If `return_gradient` is True, also returns:
            - grad_z (torch.Tensor): The gradient of the objective function with respect to `z`.
            - grad_Q (torch.Tensor): The gradient of the objective function with respect to `Q`.
    """
    
    device = z.device

    n_probes = b.shape[0]
    nx = b.shape[1]
    ny = b.shape[2]

    if return_gradient==True:
        z.requires_grad = True
        Q.requires_grad = True

    f_val = 0.0

    fft_z = torch.fft.fft2(Q*z)
    proj_z = b[current_ap, :, :].view(nx, ny) * torch.exp(1j*torch.angle(fft_z))

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


def f_pft(z, Q, b, current_ap, lmbda=0.005, return_gradient=False, B=None, M=None, p=None, q=None,
          m1_mod=None, m2_mod=None, precomputed_prod=None):
    """
    Computes the objective function value and optionally its gradients for a single probe using the Partial Fourier Transform (PFT).

    Parameters:
        z (torch.Tensor): The section of the image being probed.
        Q (torch.Tensor): The probe function applied to the section of the image.
        b (torch.Tensor): The measured diffraction intensities for each probe position.
        current_ap (int): The index of the current probe application.
        lmbda (float, optional): The regularization parameter for the total variation. Defaults to 0.005.
        return_gradient (bool, optional): If True, also returns the gradients with respect to `z` and `Q`. Defaults to False.
        B (list of torch.Tensor): Precomputed B matrices (B1 and B2) used in the PFT.
        M (optional): Parameters for PFT computation (not detailed in the function).
        p (list of int): Divisors p1 and p2 of N1 and N2 (size of image), respectively.
        q (list of int): Quotients of N and p for each dimension.
        m1_mod (torch.Tensor): Modulo values for the first dimension.
        m2_mod (torch.Tensor): Modulo values for the second dimension.
        precomputed_prod (torch.Tensor): Precomputed product of exponential terms and powers for both dimensions.

    Returns:
        torch.Tensor: The average objective function value for the single probe application.
        If `return_gradient` is True, also returns:
            - grad_z (torch.Tensor): The gradient of the objective function with respect to `z`.
            - grad_Q (torch.Tensor): The gradient of the objective function with respect to `Q`.
    """
    
    device = z.device

    n_probes = b.shape[0]

    if return_gradient==True:
        z.requires_grad = True
        Q.requires_grad = True

    f_val = 0.0

    z_temp = Q*z
    z_temp = z_temp.view(p[0], q[0], p[1], q[1]).permute(0, 2, 1, 3).contiguous().view(p[0], p[1], q[0], q[1])
    pft_z = pft2d_computation(z_temp, B, m1_mod, m2_mod, precomputed_prod, device=device)

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


#################### ePIE FUNCTIONS ####################
def ePIE(z, Q, b, lmbda=0, beta_ap=0.01, beta_obj=1, num_iters = 200, tol=1e-1, return_all_metrics=False,
         mask=None, Y1=None, Y2=None, X1=None, X2=None, z_true=None, r_true=None, phi_true=None, true_probe=None):
    """
    Performs the extended Ptychographic Iterative Engine (ePIE) algorithm to reconstruct an image and update the probe function.

    Parameters:
        z (torch.Tensor): The initial estimate of the image to be reconstructed.
        Q (torch.Tensor): The initial probe function.
        b (torch.Tensor): The measured diffraction intensities for each probe position.
        lmbda (float, optional): The regularization parameter for total variation. Defaults to 0.
        beta_ap (float, optional): The step size for updating the probe function. Defaults to 0.01.
        beta_obj (float, optional): The step size for updating the object. Defaults to 1.
        num_iters (int, optional): The maximum number of iterations. Defaults to 200.
        tol (float, optional): The tolerance for convergence based on the Cauchy error. Defaults to 1e-1.
        return_all_metrics (bool, optional): If True, returns additional metrics collected during the iteration. Defaults to False.
        mask (torch.Tensor, optional): A mask to apply to the probe function (if any).
        Y1, Y2 (torch.Tensor, optional): Vertical start and end indices for each probe position.
        X1, X2 (torch.Tensor, optional): Horizontal start and end indices for each probe position.
        z_true (torch.Tensor, optional): The ground truth image for evaluation purposes.
        r_true (torch.Tensor, optional): The ground truth magnitude for evaluation purposes.
        phi_true (torch.Tensor, optional): The ground truth phase for evaluation purposes.
        true_probe (torch.Tensor, optional): The ground truth probe function for evaluation purposes.

    Returns:
        tuple: A tuple containing:
            - z (torch.Tensor): The reconstructed image after iteration.
            - Q (torch.Tensor): The updated probe function after iteration.
            If `return_all_metrics` is True, also returns:
                - f_val_hist (list): History of objective function values.
                - grad_z_hist (list): History of gradients with respect to the image.
                - grad_Q_hist (list): History of gradients with respect to the probe.
                - rel_err_hist (list): History of relative errors.
                - cauchy_err_hist (list): History of Cauchy errors.
                - time_hist (list): History of iteration times.
                - SSIM_mag_hist (list): History of SSIM metrics for magnitude.
                - SSIM_phase_hist (list): History of SSIM metrics for phase.
                - PSNR_mag_hist (list): History of PSNR metrics for magnitude.
                - PSNR_phase_hist (list): History of PSNR metrics for phase.
                - mag_err_hist (list): History of magnitude errors.
                - phase_err_hist (list): History of phase errors.
                - probe_err_hist (list): History of probe errors.
    """

    n_probes = b.shape[0]
    nx_padded = b.shape[1]
    ny_padded = b.shape[2]
    nx = z_true.shape[0]
    ny = z_true.shape[1]
    
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

    mag_true = torch.Tensor(torch.abs(z_true)).cpu()
    phase_true = torch.Tensor(torch.angle(z_true)).cpu()
    
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

            # Update the object and probe
            fval, grad_zn, grad_Q = f_single_probe(obj_n, Q, b, lmbda=lmbda, current_ap = i, return_gradient=True) 
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
        z_cropped = crop_image(z, z_true, nx, ny, nx_padded, ny_padded)
        end_time = time.time()
        # print('cropping time = ', end_time - start_time, ', z_cropped.shape = ', z_cropped.shape, 'z_true.shape = ', z_true.shape)
        rel_err = rel_error(z_cropped, z_true)
        assert Q.shape == true_probe.shape
        assert Q.type() == true_probe.type()
        probe_err = torch.norm(Q - true_probe)/torch.norm(true_probe)
        probe_err_hist.append(probe_err)

        start_time = time.time()
        fval_full, grad_z, grad_Q = f_original(z, Q, b, Y1, Y2, X1, X2, return_gradient=True)
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

            plt.imshow(torch.angle(z).cpu(), vmin=0, vmax=torch.pi/2, cmap=cmap)
            plt.colorbar()
            plt.show()
            
            break

        if j%10 == 0 or j == 1:
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

        if j%25 == 0:
            plt.imshow(torch.angle(z).cpu(), vmin=0, vmax=torch.pi/2, cmap=cmap)
            plt.colorbar()
            plt.show()

        j += 1
        
        ''' # Uncomment to display phase every 500 iters
        if j%500 == 0:
            plt.imshow(torch.angle(z).cpu(), cmap=cmap)
            plt.colorbar()
            plt.show()
        '''

    if return_all_metrics:
        return z, Q, f_val_hist, grad_z_hist, grad_Q_hist, rel_err_hist, cauchy_err_hist, time_hist, SSIM_mag_hist, SSIM_phase_hist, PSNR_mag_hist, PSNR_phase_hist, phase_err_hist, mag_err_hist, probe_err_hist
        
    return z, Q


def ePIE_PFT(z, Q, lmbda=5e-3, beta_ap=0.01, beta_obj=1, num_iters = 2001, tol=2.5e-1, return_all_metrics=False, b=None, b_crop=None,
             mask=None, Y1=None, Y2=None, X1=None, X2=None, z_true=None, r_true=None, phi_true=None, pft_params=None, true_probe=None):
    """
    Performs the extended Ptychographic Iterative Engine (ePIE) algorithm using the Partial Fourier Transform (PFT) to reconstruct an image and update the probe 
    function.

    Parameters:
        z (torch.Tensor): The initial estimate of the image to be reconstructed.
        Q (torch.Tensor): The initial probe function.
        lmbda (float, optional): The regularization parameter for total variation. Defaults to 5e-3.
        beta_ap (float, optional): The step size for updating the probe function. Defaults to 0.01.
        beta_obj (float, optional): The step size for updating the object. Defaults to 1.
        num_iters (int, optional): The maximum number of iterations. Defaults to 2001.
        tol (float, optional): The tolerance for convergence based on the error between iterations. Defaults to 2.5e-1.
        return_all_metrics (bool, optional): If True, returns additional metrics collected during the iteration. Defaults to False.
        b (torch.Tensor, optional): The measured diffraction intensities for each probe position.
        b_crop (torch.Tensor, optional): Cropped diffraction intensities for the PFT.
        mask (torch.Tensor, optional): A mask to apply to the probe function (if any).
        Y1, Y2 (torch.Tensor, optional): Vertical start and end indices for each probe position.
        X1, X2 (torch.Tensor, optional): Horizontal start and end indices for each probe position.
        z_true (torch.Tensor, optional): The ground truth image for evaluation purposes.
        r_true (torch.Tensor, optional): The ground truth magnitude for evaluation purposes.
        phi_true (torch.Tensor, optional): The ground truth phase for evaluation purposes.
        pft_params (dict, optional): Parameters for the Partial Fourier Transform function.
        true_probe (torch.Tensor, optional): The ground truth probe function for evaluation purposes.

    Returns:
        tuple: A tuple containing:
            - z (torch.Tensor): The reconstructed image after iteration.
            - Q (torch.Tensor): The updated probe function after iteration.
            If `return_all_metrics` is True, also returns:
                - f_val_hist (list): History of objective function values.
                - grad_z_hist (list): History of gradients with respect to the image.
                - grad_Q_hist (list): History of gradients with respect to the probe.
                - rel_err_hist (list): History of relative errors.
                - cauchy_err_hist (list): History of errors between iterations.
                - time_hist (list): History of iteration times.
                - SSIM_mag_hist (list): History of SSIM metrics for magnitude.
                - SSIM_phase_hist (list): History of SSIM metrics for phase.
                - PSNR_mag_hist (list): History of PSNR metrics for magnitude.
                - PSNR_phase_hist (list): History of PSNR metrics for phase.
                - mag_err_hist (list): History of magnitude errors.
                - phase_err_hist (list): History of phase errors.
                - probe_err_hist (list): History of probe errors.
    """
    
    n_probes = b.shape[0]
    nx_padded = b.shape[1]
    ny_padded = b.shape[2]
    nx = z_true.shape[0]
    ny = z_true.shape[1]
    
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

    mag_true = torch.Tensor(torch.abs(z_true)).cpu()
    phase_true = torch.Tensor(torch.angle(z_true)).cpu()
    
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

            # Update the object and probe
            fval, grad_z, grad_Q = f_pft(obj_n, Q, b_crop, lmbda = lmbda, current_ap = i, return_gradient=True, **pft_params) 
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
        z_cropped = crop_image(z, z_true, nx, ny, nx_padded, ny_padded)
        rel_err = rel_error(z_cropped, z_true)
        assert Q.shape == true_probe.shape
        assert Q.type() == true_probe.type()
        probe_err = torch.norm(Q - true_probe)/torch.norm(true_probe)

        start_time = time.time()
        fval_full, grad_z, grad_Q = f_original(z, Q, b, Y1, Y2, X1, X2, return_gradient=True)
        end_time = time.time()
        f_val_time = end_time - start_time
        
        
        cauchy_err = torch.norm(z_old - z.cpu()) # /(torch.norm(z_old)

        f_val_hist.append(fval_full.cpu())
        grad_z_hist.append(torch.norm(grad_z).cpu().item())
        grad_Q_hist.append(torch.norm(grad_Q).cpu().item())
        rel_err_hist.append(rel_err)
        cauchy_err_hist.append(cauchy_err.cpu())
        probe_err_hist.append(probe_err.cpu())

        # Collect Magnitude and Phase (for computing ssim and psnr)
        mag_opt = torch.abs(z_cropped).cpu().detach()
        phase_opt = torch.angle(z_cropped).cpu().detach()

        # Compute and Save mag_err and phase_err
        mag_err = torch.norm(mag_opt - mag_true) / torch.norm(mag_true)
        phase_err = torch.norm(phase_opt - phase_true) / torch.norm(phase_true)
        mag_err_hist.append(mag_err.cpu())
        phase_err_hist.append(phase_err.cpu())

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

        if j%10 == 0 or j == 1:
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

        j += 1

        ''' # Uncomment to display phase every 500 iters
        if j%500 == 0:
            plt.imshow(torch.angle(z).cpu(), cmap=cmap)
            plt.colorbar()
            plt.show()
        '''

    if return_all_metrics:
        return z, Q, f_val_hist, grad_z_hist, grad_Q_hist, rel_err_hist, cauchy_err_hist, time_hist, SSIM_mag_hist, SSIM_phase_hist, PSNR_mag_hist, PSNR_phase_hist, phase_err_hist, mag_err_hist, probe_err_hist 
    
    return z, Q
