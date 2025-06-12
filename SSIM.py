import torch
import numpy as np
from mesh import read_mesh, pixelize_mesh
from skimage.metrics import structural_similarity as ssim


def ssim_3d(img1, img2, window_size=10, epsilon=1e-6):
    # img1 and img2 should be 3D numpy arrays (D, H, W)
    D, H, W = img1.shape
    ssim_map = np.zeros_like(img1)

    # Iterate over each voxel in the image (ignoring borders for simplicity)
    pad_size = window_size // 2
    img1_padded = np.pad(img1, pad_size, mode='constant')
    img2_padded = np.pad(img2, pad_size, mode='constant')

    for d in range(pad_size, D + pad_size):
        for h in range(pad_size, H + pad_size):
            for w in range(pad_size, W + pad_size):
                # Extract the local window centered at (d, h, w)
                window1 = img1_padded[d - pad_size: d + pad_size + 1, h - pad_size: h + pad_size + 1,
                          w - pad_size: w + pad_size + 1]
                window2 = img2_padded[d - pad_size: d + pad_size + 1, h - pad_size: h + pad_size + 1,
                          w - pad_size: w + pad_size + 1]

                # Check for constant windows (variance = 0)
                if np.var(window1) == 0 or np.var(window2) == 0:
                    ssim_value = 1.0  # Perfect similarity if both are constant
                else:
                    # Compute SSIM for the local window
                    try:
                        ssim_value = ssim(window1, window2, data_range=window1.max() - window1.min())
                    except:
                        ssim_value = 1.0  # If SSIM fails (e.g., due to division by zero), assume perfect similarity

                ssim_map[d - pad_size, h - pad_size, w - pad_size] = ssim_value

    # Calculate the mean SSIM for the entire image
    return np.mean(ssim_map)

# Import the mesh file and the AE model
file_path = "ah93w145.msh"
points = read_mesh(file_path)
zero_indices1 = np.where(points[:, 1] == 1500)[0]  # 这个命令表示找到y=1500的点，然后返回他们的行索引
x_min, x_max = np.min(points[:, 0][zero_indices1]), np.max(points[:, 0][zero_indices1])
z_min, z_max = np.min(points[:, 2][zero_indices1]), np.max(points[:, 2][zero_indices1])
model = torch.load("model_AE.pth")

# Rasterize the nodes into a grid and obtain the output values from the AE model
grid = torch.tensor(pixelize_mesh(points, (80, 40, 80)), device="cuda", dtype=torch.float32)
grid = grid.unsqueeze(0).unsqueeze(0)
output_p, latent = model(grid)

# Dimensionality reduction
input_p = grid.squeeze(0).squeeze(0)
output_p = output_p.squeeze(0).squeeze(0)
input_p = input_p.cpu().detach().numpy()
output_p = output_p.cpu().detach().numpy()

# Compute the MSSIM result
result = ssim_3d(input_p, output_p)
print("3D SSIM:", result)