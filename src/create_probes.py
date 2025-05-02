import torch
import torch.fft as fft

import math

########## NON-BLIND PTYCHOGRAPHY ##########

def create_probes_nonblind(nx, ny, device='cpu'):
    """
    Generates a set of non-blind probe patterns.

    Parameters:
        nx (int): The number of pixels in the x-dimension of each probe.
        ny (int): The number of pixels in the y-dimension of each probe.
        device (str, optional): Device to perform computations on ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: A tensor of shape (9, nx, ny) representing the set of non-blind probes. Each probe is a binary mask with a different region of the 
                            image illuminated.
            - int: The number of probes, which is 9 in this case.
    """
    
    n_probes = 9
    probes = torch.zeros(n_probes, nx, ny).to(device)
    
    probes[0,0:int(nx/2), 0:int(ny/2)] = 1.0
    probes[1,int(nx/4):int(3*nx/4), 0:int(ny/2)] = 1.0
    probes[2,int(nx/2):int(nx), 0:int(ny/2)] = 1.0
    
    probes[3,0:int(nx/2), int(ny/4):int(3*ny/4)] = 1.0
    probes[4,int(nx/4):int(3*nx/4), int(ny/4):int(3*ny/4)] = 1.0
    probes[5,int(nx/2):int(nx), int(ny/4):int(3*ny/4)] = 1.0
    
    probes[6,0:int(nx/2), int(ny/2):int(ny)] = 1.0
    probes[7,int(nx/4):int(3*nx/4), int(ny/2):int(ny)] = 1.0
    probes[8,int(nx/2):int(nx), int(ny/2):int(ny)] = 1.0

    return probes, n_probes


########## BLIND PTYCHOGRAPHY ##########

def create_true_probe(imin, radius=50, sigma=1e6, device='cpu'):
    """
    Creates a true probe pattern for an image using circular (Gaussian) apertures.

    Parameters:
        imin (torch.Tensor): The input image tensor, which can be grayscale or RGB.
        radius (int, optional): The radius of the aperture in pixels. Defaults to 50.
        sigma (float, optional): The standard deviation of the Gaussian function, controlling the aperture's spread. Defaults to 1e6.
        device (str, optional): Device to perform computations on ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: The generated probe pattern, with shape matching the input image dimensions. The tensor is in complex format if the image is 
                            grayscale, otherwise it's a 3-channel tensor.
            - torch.Tensor: A binary mask indicating the region within the aperture radius.
            - torch.Tensor: The x-coordinate of the image center.
            - torch.Tensor: The y-coordinate of the image center.
    """
    
    # Check if the image is colored (i.e. 3 channels) or gray-scale
    if imin.dim() > 2:
        imy, imx, imz = imin.size()
    else:
        imy, imx = imin.size()

    # Find the center of the image
    center_x = imx // 2
    center_y = imy // 2

    # Create a grid of coordinates
    rr, cc = torch.meshgrid(torch.arange(1, imy+1), torch.arange(1, imy+1), indexing='ij')

    # Compute the distance from the center
    modulus = torch.sqrt((rr - center_y)**2 + (cc - center_x)**2)

    # Generate the Gaussian aperture function
    gauss_app = torch.exp(-modulus**2 / (2 * sigma**2))
    gauss_app = gauss_app / torch.max(gauss_app)

    # Create the aperture mask and apply Gaussian function
    if imin.dim() > 2: # For RGB images
        app_lim = (torch.sqrt((rr - center_y) ** 2 + (cc - center_x) ** 2) <= radius).unsqueeze(2).expand(imy, imy, imz)
        apertures = app_lim * gauss_app.unsqueeze(2).expand(imy, imy, imz)
    else: # For grayscale images
        app_lim = (torch.sqrt((rr - center_y) ** 2 + (cc - center_x) ** 2) <= radius)
        apertures = torch.complex(app_lim * gauss_app, torch.zeros_like(app_lim * gauss_app)).to(torch.complex128)

    return apertures.to(device), app_lim.to(device), torch.tensor([center_x]), torch.tensor([center_y])


def create_probes(imin, spacing=50, radius=50, sigma=1e6, string='grid', dither=3, xedge=120, yedge=120, device='cpu'):
    """
    Generates a set of probe functions for ptychography using either a grid or square pattern.

    Parameters:
        imin (torch.Tensor): The input image tensor, which can be grayscale or RGB.
        spacing (int, optional): The spacing between the centers of adjacent probes in pixels. Defaults to 50.
        radius (int, optional): The radius of the probes in pixels. Defaults to 50.
        sigma (float, optional): The standard deviation of the Gaussian function used to shape the probes. Defaults to 1e6.
        string (str, optional): The pattern type for probe placement, either 'grid' or 'square'. Defaults to 'grid'.
        dither (int, optional): The amount of random displacement (dithering) applied to the probe centers. Defaults to 3.
        xedge (int, optional): The minimum x-coordinate for probe placement to avoid edges. Defaults to 120.
        yedge (int, optional): The minimum y-coordinate for probe placement to avoid edges. Defaults to 120.
        device (str, optional): The device on which to perform the computations, such as 'cpu' or 'cuda'. Defaults to 'cpu'.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: A tensor containing all generated probe functions.
            - torch.Tensor: A binary mask indicating the regions covered by the probes.
            - torch.Tensor: The x-coordinates of the probe centers.
            - torch.Tensor: The y-coordinates of the probe centers.
            - int: The total number of probes generated.
    """
    
    # Check if the image is colored (i.e. 3 channels) or gray-scale
    if imin.dim() > 2:
        imy, imx, imz = imin.size()
    else:
        imy, imx = imin.size()

    # Calculate the far edges of the image for probe placement boundaries
    xfar = (imx - xedge) + 2
    yfar = (imy - yedge) + 2
    
    half_size = math.ceil(imy/2)
    diameter = 2 * radius
    
    # DUE TO MATLAB INDEXING, EVERYTHING MAY BE OFF BY 1 (INCLUDING count)

    # Generate probes based on the specified pattern
    if string == 'grid':
        rr, cc = torch.meshgrid(torch.arange(1, imy+1), torch.arange(1, imy+1), indexing='ij') 
        
        r_lin = rr.reshape(1, -1)
        c_lin = cc.reshape(1, -1)

        # Exclude coordinates near the image edges
        r_lin[r_lin < xedge] = -1
        r_lin[r_lin > xfar] = -1
        c_lin[c_lin < yedge] = -1
        c_lin[c_lin > yfar] = -1
        
        center_list = torch.cat([r_lin, c_lin], dim = 0)
        count = 0

        # Initialize variables to store probe information
        centerx = torch.empty(0)
        centery = torch.empty(0)
        app_lim_list = []
        apertures_list = []

        # Iterate through potential probe positions and generate probes
        for i in range(center_list.size(1)):
            
            if(
                center_list[0, i] % spacing == 0
                and center_list[1, i] % spacing == 0
                and center_list[0, i] != -1
                and center_list[1, i] != -1
            ):
                # Apply dithering to the probe center coordinates
                ditherx = torch.round(torch.rand(1) * dither - (dither / 2))
                dithery = torch.round(torch.rand(1) * dither - (dither / 2))
                centerx = torch.cat([centerx, center_list[0, i] + ditherx])
                centery = torch.cat([centery, center_list[1, i] + dithery])

                # Calculate the modulus for the Gaussian aperture
                modulus = torch.sqrt((rr - centerx[count])**2 + (cc - centery[count])**2)
                gauss_app = torch.exp(-modulus**2 / (2 * sigma** 2))
                gauss_app = gauss_app / torch.max(gauss_app) # Normalize the Gaussian

                # Create probe masks and apply the Gaussian aperture
                if imin.dim() > 2: # For RGB images
                    app_lim = torch.cat([app_lim, (torch.sqrt((rr - centerx[count]) ** 2 + (cc - centery[count]) ** 2) <= radius).unsqueeze(2).expand(imy, imy, imz)])
                    apertures = torch.cat([apertures, app_lim[count] * gauss_app.unsqueeze(2).expand(imy, imy, imz)])
                else: # For grayscale images
                    app_lim = (torch.sqrt((rr - centerx[count]) ** 2 + (cc - centery[count]) ** 2) <= radius)
                    apertures = torch.complex(app_lim * gauss_app, torch.zeros_like(app_lim * gauss_app)).to(torch.complex128)

                # Store the generated probe and mask
                app_lim_list.append(app_lim)
                apertures_list.append(apertures)
                count += 1

        # Stack all generated probes and masks into tensors
        app_lim = torch.stack(app_lim_list)
        apertures = torch.stack(apertures_list)
        
        return apertures.to(device), app_lim.to(device), centerx, centery, count
    
    elif string == 'square':
        rr, cc = torch.meshgrid(torch.arange(1, imy+1), torch.arange(1, imy+1), indexing='ij') 
        
        r_lin = rr.reshape(1, -1)
        c_lin = cc.reshape(1, -1)

        # Exclude coordinates near the image edges
        r_lin[r_lin < xedge] = -1
        r_lin[r_lin > xfar] = -1
        c_lin[c_lin < yedge] = -1
        c_lin[c_lin > yfar] = -1
        
        center_list = torch.cat([r_lin, c_lin], dim = 0)
        count = 0

        # Initialize variables to store probe information
        centerx = torch.empty(0)
        centery = torch.empty(0)
        app_lim_list = []
        apertures_list = []

        # Iterate through potential probe positions and generate probes
        for i in range(center_list.size(1)):
            
            if(
                center_list[0, i] % spacing == 0
                and center_list[1, i] % spacing == 0
                and center_list[0, i] != -1
                and center_list[1, i] != -1
            ):
                # Apply dithering to the probe center coordinates
                ditherx = torch.round(torch.rand(1) * dither - (dither / 2))
                dithery = torch.round(torch.rand(1) * dither - (dither / 2))
                centerx = torch.cat([centerx, center_list[0, i] + ditherx])
                centery = torch.cat([centery, center_list[1, i] + dithery])

                # Calculate the modulus for the Gaussian aperture
                modulus = torch.sqrt((rr - centerx[count])**2 + (cc - centery[count])**2)
                gauss_app = torch.exp(-modulus**2 / (2 * sigma**2))
                gauss_app = gauss_app / torch.max(gauss_app)

                # Create a square aperture mask
                app_holder = torch.zeros(imx, imy)
                app_holder[
                    centery[count].int() - radius : centery[count].int() + radius,
                    centerx[count].int() - radius : centerx[count].int() + radius
                ] = 1

                # Apply the Gaussian aperture to the square mask
                if imin.dim() > 2: # For RGB images
                    app_lim = torch.stack([app_holder.unsqueeze(2).expand(imy, imy, imz)] * count)
                    apertures = torch.stack([app_lim[count] * gauss_app.unsqueeze(2).expand(imy, imy, imz)] * count)
                else: # For grayscale images
                    app_lim = app_holder 
                    apertures = app_lim * gauss_app

                # Store the generated probe and mask
                app_lim_list.append(app_lim)
                apertures_list.append(apertures)
                count += 1

        # Stack all generated probes and masks into tensors
        app_lim = torch.stack(app_lim_list)
        apertures = torch.stack(apertures_list)
        
        return apertures.to(device), app_lim.to(device), centerx, centery, count

