import numpy as np
import torch 
import sigpy.mri as mr
from sense_estimation import create_basis_functions, sense_estimation_ls, pytorch_sense_estimation_ls
from utils import generate_US_pattern, rmse
from common_utils import loadmat, extract_accleration_factor, extract_file_paths


def IFFT(x):
    return torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(x, dim=(-2, -1)), dim=(-2,-1)), dim=(-2, -1))
def FFT(x):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), dim=(-2, -1)), dim=(-2, -1))

def JSENSE(kspace, mask, est_sensemap, num_iter, max_basis_order, DEVICE):
    """
    JSENSE reconstruction
    """
    # Create undersampled kspace data
    _, rows, cols = kspace.shape 

    # Initializing image reconstruction
    recs = torch.zeros((num_iter, rows, cols), dtype=torch.complex64, device=DEVICE)
    img_sli_rec = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(kspace, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
    
    recs[0] = torch.from_numpy(np.sum(img_sli_rec * np.conjugate(est_sensemap), axis=0) / np.sum(est_sensemap * np.conjugate(est_sensemap), axis=0)).to(DEVICE)

    
    # Convert to pytorch and put on GPU
    kspace = torch.from_numpy(kspace).to(DEVICE)
    mask = torch.from_numpy(mask).to(DEVICE)
    # Initializing sense reconstruction
    basis_funct = torch.from_numpy(create_basis_functions(rows, cols, max_basis_order, show_plot=False)).to(DEVICE)
    coeffs_array = pytorch_sense_estimation_ls(kspace, recs[0], basis_funct, mask, DEVICE)

    est_sensemap = torch.sum(coeffs_array[:, :, None, None] * basis_funct[None], 1)

    # First step of image reconstruction
    img_sli_rec = IFFT(kspace)
    recs[1] = torch.sum(img_sli_rec * torch.conj(est_sensemap), dim=0) / torch.sum(est_sensemap * torch.conj(est_sensemap), dim=0) 

    for i in range(1, num_iter-1):
        # Sense reconstruction 
        # ===============================================
        coeffs_array = pytorch_sense_estimation_ls(kspace, recs[0], basis_funct, mask, DEVICE)

        # Data consistency projection
        # ===============================================
        rec_ksp = (1 - mask) * FFT(est_sensemap * recs[i, None, :, :])
        rec_usksp = rec_ksp + kspace

        # Update sensemap  
        est_sensemap = torch.sum(coeffs_array[:, :, None, None] * basis_funct[None], 1)

        # Create next reconstruction
        img_sli_rec = IFFT(rec_usksp)
        recs[i+1] = (torch.sum(img_sli_rec * torch.conj(est_sensemap), dim=0) / torch.sum(est_sensemap * torch.conj(est_sensemap), dim=0))
        
    return recs[-1]

# import argparse

# import numpy as np
# import h5py

# from sense_estimation import create_basis_functions, sense_estimation_ls
# import pickle
# import matplotlib.pyplot as plt
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(prog='PROG')
#     parser.add_argument('--path', type=str, default='data/testdata.h5', help='Full path to h5 file for reconstruction') 
#     parser.add_argument('--slice', type=int, default=0, help='Slice to reconstruct, 0 = all slices (default: 0)')
#     parser.add_argument('--usfact', type=int, default=4, help='Undersampling factor (default: 4)')
#     parser.add_argument('--basis_order', type=int, default=15, help='Polynomial basis max order (default: 8)')
#     parser.add_argument('--num_iter', type=int, default=50, help='Number of iterations (default: 50)')

#     args = parser.parse_args()
#     path = '/mnt/remote/CMRxRecon/Data_test/MultiCoil/Cine/TrainingSet/AccFactor04/P101/cine_lax.mat'
#     path_mask = '/mnt/remote/CMRxRecon/Data_test/MultiCoil/Cine/TrainingSet/AccFactor04/P101/cine_lax_mask.mat'
#     sli = args.slice
#     R = args.usfact
#     max_basis_order = args.basis_order
#     num_iter = args.num_iter

#     print('______JSENSE RECONSTRUCTION______')
#     print('PATH: ', path, '   SLICE: ', sli, '    Under sample factor: ', R, '    Iterations: ', num_iter)

#     ## OPEN DATA 
#    # with h5py.File(path, 'r') as fdset:
#     ksp_sli = loadmat(path)['kspace_sub04'][:][0,1].transpose(0, 2, 1)  # The shape of kspace tensor is (number of slices, number of coils, height, width)
#     mask = loadmat(path_mask)['mask04'][:].transpose(1,0)[np.newaxis].repeat(10, axis=0)  # The shape of mask tensor is (number of slices, height, width

#     img_sli = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(ksp_sli, axes=(1,2)), axes=(1,2)), axes=(1,2)) # Change to normal view
#     gt_img = np.sqrt(np.sum(np.square(np.abs(img_sli.copy())), axis=0))  # Save rss ground truth 
#     ksp_sli = np.fft.fftshift(np.fft.fft2(img_sli, axes=(1,2)), axes=(1,2)) # Create kspace data
#     num_coils, rows, cols = ksp_sli.shape 

#     # Create and apply US pattern
#     # us_pat, num_low_freqs = generate_US_pattern(ksp_sli.shape, R=R) 
#     # us_ksp = us_pat * ksp_sli
    
#     # Estimate initial sensemaps using Espirit 
#     est_sensemap = np.fft.fftshift(mr.app.EspiritCalib(ksp_sli, calib_width=24, thresh=0.02, kernel_width=6, crop=0.01, max_iter=50, show_pbar=False).run(),axes=(1, 2))
#     us_image, reocon = JSENSE(ksp_sli, mask, est_sensemap, num_iter, max_basis_order, 'cuda:3')
    
#     plt.imsave('./JSENSE_gt.png', np.abs(gt_img), cmap='gray')
#     #plt.imsave('./JSENSE_us.png', np.abs(us_image.detach().cpu().numpy()), cmap='gray')
#     plt.imsave('./JSENSE_recon.png', np.abs(reocon.detach().cpu().numpy()), cmap='gray')
