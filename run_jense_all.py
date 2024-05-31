import os
import numpy as np

from tqdm import trange
from common_utils import loadmat, extract_accleration_factor, extract_file_paths
from jense import JSENSE
from torch import fft
import time
import scipy.io as sio
import sigpy.mri as mr
DEVICE = "cuda:3"

def run_jense(file_structure, output_path):
    for i in trange(len(file_structure[0])):
        filename = [sublist[i] for sublist in file_structure if sublist]
        acc_factor = extract_accleration_factor(filename[0])
        data = loadmat(filename[0])[f"kspace_sub{acc_factor}"].transpose(0, 1, 2, 4, 3).astype(np.complex64)
        SamMask = loadmat(filename[2])[f"mask{acc_factor}"].transpose(1, 0).astype(np.float32)

        frames, slices, coils, x, y = data.shape

        SamMask = SamMask[np.newaxis].repeat(coils, axis=0)

        recon_image = np.zeros((x, y, slices, frames), np.complex64)
        
        time_all_start = time.time()
        for _slice in range(slices):
            est_sensemap = (mr.app.EspiritCalib(data[0,_slice], calib_width=24, thresh=0.02, kernel_width=6, crop=0.01, max_iter=50, show_pbar=False).run()).astype(np.complex64)
            for frame in range(frames):
                tstKsp = data[frame, _slice]

                #tstKsp = data_cpl#.transpose(1, 2, 0)

                # %% normalize the undersampled k-space
                zf_coil_img = np.fft.ifftshift(
                    np.fft.ifft2(np.fft.fftshift(tstKsp, axes=(-2, -1)), axes=(-2,-1)),
                    axes=(-2, -1),
                )

                NormFactor = np.max(np.sqrt(np.sum(np.abs(zf_coil_img) ** 2, axis=2)))
                tstDsKsp = tstKsp / NormFactor

                time_start = time.time()
                # pre_img: complex image outputted directly by the MLPs
                # pre_tstCsm: predicted sensitivity maps
                # pre_img_dc: complex image reconstructed after the k-space data consistency step
                # pre_ksp: predicted k-space
                jsense_recon = JSENSE(
                    tstDsKsp,
                    SamMask,
                    est_sensemap,
                    num_iter=30,
                    max_basis_order=15,
                    DEVICE=DEVICE,
                )

                recon_image[..., _slice, frame] = jsense_recon.detach().cpu().numpy() * NormFactor
                time_end = time.time()
                print(
                    f"Reconstruction process 2D took {((time_end - time_start) / 60):.2f} mins"
                )

        time_all_end = time.time()
        print(
            f"Reconstruction process 4D took {((time_all_end - time_all_start) / 60):.2f} mins"
        )

        kspace_sub = {f"kspace_sub{acc_factor}": np.abs(recon_image)}

        if not os.path.exists(
            os.path.join(output_path, filename[0].split("/Data_test/")[-1].split("cine")[0])
        ):
            os.makedirs(
                os.path.join(
                    output_path, filename[0].split("/Data_test/")[-1].split("cine")[0]
                )
            )

        sio.savemat(
            os.path.join(output_path, filename[0].split("/Data_test/")[-1]),
            kspace_sub,
        )

file_structure = extract_file_paths(
    "/mnt/remote/CMRxRecon/Data_test/",
    "MultiCoil",
    "TrainingSet",
    "AccFactor04",
    cardiac_view="cine_lax",
)

output_path = "/mnt/remote/CMRxRecon/Experiments/JENSE/"

run_jense(file_structure, output_path)


file_structure = extract_file_paths(
    "/mnt/remote/CMRxRecon/Data_test/",
    "MultiCoil",
    "TrainingSet",
    "AccFactor08",
    cardiac_view="cine_lax",
)

run_jense(file_structure, output_path)

file_structure = extract_file_paths(
    "/mnt/remote/CMRxRecon/Data_test/",
    "MultiCoil",
    "TrainingSet",
    "AccFactor10",
    cardiac_view="cine_lax",
)

run_jense(file_structure, output_path)

file_structure = extract_file_paths(
    "/mnt/remote/CMRxRecon/Data_test/",
    "MultiCoil",
    "TrainingSet",
    "AccFactor04",
    cardiac_view="cine_sax",
)

run_jense(file_structure, output_path)

file_structure = extract_file_paths(
    "/mnt/remote/CMRxRecon/Data_test/",
    "MultiCoil",
    "TrainingSet",
    "AccFactor08",
    cardiac_view="cine_sax",
)

run_jense(file_structure, output_path)

file_structure = extract_file_paths(
    "/mnt/remote/CMRxRecon/Data_test/",
    "MultiCoil",
    "TrainingSet",
    "AccFactor10",
    cardiac_view="cine_sax",
)

run_jense(file_structure, output_path)

