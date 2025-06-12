import numpy as np
import matplotlib.pyplot as plt

# 读取数据
data = np.loadtxt("data_contour.txt")
print(data.shape)

# The data file contains three columns: x, z, flow field data
x = data[:, 0]
y = data[:, 1]
flow = data[:, 2]

# Assume that the x and y coordinates are uniformly distributed
xi = np.linspace(min(x), max(x), len(np.unique(x)))
yi = np.linspace(min(y), max(y), len(np.unique(y)))
xi, yi = np.meshgrid(xi, yi)

# Fill the flow field data into the grid
zi = np.full_like(xi, np.nan)  # Initialize with NaN
zi[np.searchsorted(np.unique(y), y), np.searchsorted(np.unique(x), x)] = flow

# Load the ground truth; 0.6 represents the cross-section at y = 0.6
fluent_data = np.load("naca1410_fluent_0.6.npy")

error = (fluent_data[2] - zi)
error = np.abs(error)

# Plot
plt.figure(figsize=(9, 4))
plt.rcParams['font.family'] = 'Times New Roman'
plt.xticks(np.linspace(0.25, 1.25, num=5))
plt.yticks(np.linspace(-0.16, 0.16, num=6))
levels = np.linspace(0, 0.12, 44)
contour = plt.contourf(xi, yi, error, cmap="coolwarm", levels = levels)
cbar_v = plt.colorbar()
cbar_v = cbar_v.set_ticks(np.linspace(0, 0.12, 20))
plt.xlabel('x', fontstyle='italic', fontsize=15)
plt.ylabel('z', fontstyle='italic', fontsize=15)
plt.title('RAMLP(Error)', fontfamily='Times New Roman', fontsize=15, fontweight='bold', fontstyle='italic')
# plt.savefig('error_ramlp_0.6.jpeg', format='jpeg', dpi=300)
# plt.savefig('error_ramlp_0.6.tiff', format='tiff', dpi=300)
plt.show()

