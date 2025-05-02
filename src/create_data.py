import torch
import torch.fft as fft

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

def create_full_data(z_true, probes, n_probes, nx, ny, n, isBlind=True, device='cpu'):
    """
    Generates the Fourier magnitude data for the ptychography problem.

    Parameters:
        z_true (torch.Tensor): The true object tensor.
        probes (torch.Tensor): The probe functions used to scan the sample, with shape (n_probes, nx, ny).
        n_probes (int): The number of probe positions.
        nx (int): The number of pixels in the x-dimension.
        ny (int): The number of pixels in the y-dimension.
        n (int): The total number of pixels (nx * ny).
        isBlind (bool, optional): Determines the format of the output data. 
            If True, the function returns data in a format suitable for blind ptychography, with shape (n_probes, nx, ny). 
            If False, the data is returned in a format suitable for nonblind ptychography, with shape (n_probes, n). Defaults to True.
        device (str, optional): Device to perform the computations on ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        b (torch.Tensor): The generated Fourier magnitude data, representing the diffraction intensity measurements at each probe position.
                          The output shape is (n_probes, nx, ny) if isBlind=True, otherwise (n_probes, n).
    """

    if isBlind:
        
        b = torch.zeros(n_probes, nx, ny).to(device)
        for i in range(n_probes):
            z_temp = probes[i,:,:]*z_true.view(nx,ny) #Qz
            
            # |FQ_iz_true|^2 + noise
            b[i,:, :] = torch.abs(torch.fft.fft2(z_temp))
        
            # fig1 = plt.figure()
            # plt.imshow(b[i].cpu(), cmap=cmap)
            # plt.show()
            
            # Ensures there are no negative values in b (just in case)
            # b[i, :, :] = torch.clamp(b[i, :], min=0)

    else:

        b = torch.zeros(n_probes, n).to(device)
        for i in range(n_probes):
            z_temp = probes[i,:,:]*z_true.view(nx,ny) #Qz
            
            # |FQ_iz_true|^2 + noise
            b[i,:] = torch.abs(torch.fft.fft2(z_temp).view(-1)) 
            
            # fig1 = plt.figure()
            # plt.imshow(torch.fft.fftshift(b[i,:].view(nx, ny)).cpu(), cmap=cmap)
            # plt.show()
            
            # Ensure there are no negative values in b (just in case)
            # b[i, :] = torch.clamp(b[i, :], min=0)

    return b

def create_cropped_data(z_true, probes, n_probes, nx, ny, N, M, isBlind=True, device='cpu'):
    """
    Generates cropped Fourier magnitude data for the ptychography problem.

    Parameters:
        z_true (torch.Tensor): The true object tensor.
        probes (torch.Tensor): The probe functions used to scan the sample, with shape (n_probes, nx, ny).
        n_probes (int): The number of probe positions.
        nx (int): The number of pixels in the x-dimension.
        ny (int): The number of pixels in the y-dimension.
        N (tuple): The size of the Fourier transform grid in the x and y dimensions (N[0], N[1]).
        M (tuple): The half-width of the rectangle to crop from the Fourier transform in the x and y dimensions (M[0], M[1]).
        isBlind (bool, optional): Determines the format of the output data.
            If True, the function returns data in a format suitable for blind ptychography. 
            If False, the data is returned in a format suitable for nonblind ptychography, with shape (n_probes, n). Defaults to True.
        device (str, optional): The device to perform the computations on ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        b_crop (torch.Tensor): The cropped Fourier magnitude data, representing the diffraction intensity measurements at each probe position.
                               The output shape is (n_probes, 4*M[0]*M[1]), where each row contains the Fourier coefficients within the specified rectangle.
    """

    if isBlind:
        
        b_crop = torch.zeros(n_probes, int(4*M[0]*M[1])).to(device)
        for i in range(n_probes):
            Z = torch.abs(torch.fft.fft2(probes[i,:,:]*z_true.reshape(nx,ny)))
        
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

    else:

        b_crop = torch.zeros(n_probes, int(4*M[0]*M[1])).to(device)
        for i in range(n_probes):
            Z = torch.abs(torch.fft.fft2(probes[i,:,:]*z_true.reshape(nx,ny)))
        
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
            
            # ZERO OUT NEGATIVE VALUES IN b JUST IN CASE
            # b[i, :] = torch.clamp(b[i, :], min=0)

    return b_crop
