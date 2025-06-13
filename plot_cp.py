import os
import torch
import numpy as np
from torch import nn
from mesh import read_mesh
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from coord2CST import coord_2_CST
from CST2coord import CST_2_coord
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class MLP(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=256, output_dim=4, num_hidden_layers=5):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())

        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MHP(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=256, output_dim=1, num_hidden_layers=5):
        super(MHP, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())

        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, hidden_dim, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // reduction),
            nn.ReLU(),
            nn.Linear(hidden_dim // reduction, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels = x.shape
        x_se = self.global_avg_pool(x.unsqueeze(-1)).view(batch_size, channels)
        attention_weights = self.fc(x_se)
        return x * attention_weights

class RAMLP(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=256, output_dim=4, num_layers=4):
        super(RAMLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)])
        self.attention_layers = nn.ModuleList([SEBlock(hidden_dim) for _ in range(num_layers - 1)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer, attn in zip(self.hidden_layers, self.attention_layers):
            residual = x
            x = self.activation(layer(x))
            x = attn(x)
            x = x + residual
        x = self.output_layer(x)
        return x

path_modol_mlp = "model_MLP.pth"
path_modol_mhp = "model_MHP_U.pth"
path_modol_ramlp = "model_RAMLP.pth"
path_latent_train_6 = "latent_train_relu_6.pt"

model_MLP = torch.load(path_modol_mlp)
model_MHP = torch.load(path_modol_mhp)
model_RAMLP = torch.load(path_modol_ramlp)
latent_relu_6 = torch.load(path_latent_train_6)

def normalize_tensor(tensor, max_vals, min_vals):

    normalized_arr = 2 * (tensor - min_vals) / (max_vals - min_vals) - 1

    return normalized_arr


# The airfoil coordinate data downloaded from UIUC.
path_line = "naca1410.dat"

def get_coords(path):

    coords = np.loadtxt(path, skiprows=1)

    for i in range(5):
        if coords[i, 0] > 2:
            coords = np.delete(coords, i, axis=0)

    if coords[0, 0] < 0.5:
        for i in range(1, len(coords)-1):
            if coords[i + 1, 0] < coords[i, 0]:
                mid_index = i
                break
        first_half_reversed = coords[:mid_index + 1][::-1]
        second_half = coords[mid_index + 1:]
        coords = np.vstack((first_half_reversed, second_half))
        coords = np.delete(coords, mid_index, 0)

    if np.any(coords[:, 0] == 0):
        indices_le = np.where(coords[:, 0] == 0)[0].item()
    else:
        indices_le = np.argmin(coords[:, 0])
        coords[indices_le, 0] = 0

    if coords[0, 1] == coords[-1, 1]:
        coords[0, 1] = coords[1, 1] / 2
        coords[-1, 1] = coords[-2, 1] / 2
    else:
        pass

    dzu = coords[0, 1]
    dzl = coords[-1, 1]

    if coords.shape[0] % 2 == 0:
        coords = np.delete(coords, indices_le-1, 0)
        CST_parameter, error = coord_2_CST(coords, dzu, dzl, order=9)
    else:
        CST_parameter, error = coord_2_CST(coords, dzu, dzl, order=9)

    coords_new = CST_2_coord(CST_parameter, dzu=dzu, dzl=dzl)

    x_root = coords_new[:, 0].reshape(-1, 1)
    z_root = coords_new[:, 1].reshape(-1, 1)
    y_root = np.zeros((coords_new.shape[0], 1))
    coords_root = np.hstack((x_root, y_root, z_root))

    coords_root = torch.from_numpy(coords_root).float()

    return coords_root

y_loc, Ma, AOA, num_wing = 0.3, 0.4, -4, 1
data = get_coords(path_line)
data = data.to(device) * (1 - y_loc / 3)
data_x = data[:,0] + np.tan(np.pi/6) * y_loc
data_y = (torch.ones((data.shape[0], 1)) * y_loc).to(device)
data_ma = (torch.ones((data.shape[0], 1)) * Ma).to(device)
data_aoa = (torch.ones((data.shape[0], 1)) * AOA).to(device)
data_latent_6 = latent_relu_6[num_wing].unsqueeze(0).expand(data_y.size(0), -1).to(device)
data_input = torch.cat((data_ma, data_aoa, data_x.reshape(-1,1), data_y, data[:,2].reshape(-1,1), data_latent_6), 1)


input_max = torch.tensor([[0.6000, 5.0000, 1.45, 1.5, 0.2, 24.7941, 13.7290, 18.5239, 39.1042, 60.7303, 16.5290]], device=device)
input_min = torch.tensor([[2.0000e-01, -5.0000e+00, -0.1, -3.4106e-16, -0.2, 9.1243,  2.2724, 11.8411,  6.5119, 23.0061,  6.7900]], device=device)
output_max = torch.tensor([[308.11, 238.98, 241.22, 1.0684]], device=device)
output_min = torch.tensor([[-141.6986,  -160.26, -277.70, -3.0824]], device=device)

data_input_ = normalize_tensor(data_input, input_max, input_min)
data_output_mlp = model_MLP(data_input_)
data_output_mhp = model_MHP(data_input_)
data_output_ramlp = model_RAMLP(data_input_)
data_output_mlp = (data_output_mlp + 1) / 2 * (output_max - output_min) + output_min
data_output_mhp = (data_output_mhp + 1) / 2 * (output_max - output_min) + output_min
data_output_ramlp = (data_output_ramlp + 1) / 2 * (output_max - output_min) + output_min

data = []
path_fluent_data = "naca1410_CP_0.3.txt"
with open(path_fluent_data, 'r') as file:
    for line in file:

        values = list(map(float, line.strip().split()))
        data.append(values)

fluent_array = np.array(data)
fluent_data = torch.tensor(fluent_array)

fig, ax = plt.subplots(figsize=(10, 5))
plt.rcParams['font.family'] = 'Times New Roman'
plt.xlabel("x", fontstyle='italic', fontsize=14)
plt.ylabel("$C_{p}$", fontstyle='italic', fontsize=14)
plt.scatter(data_input[:,2].cpu().detach().numpy(), data_output_mlp[:,3].cpu().detach().numpy(), s=40, label="Pre_MLP", facecolors='brown', edgecolors="red", marker="^")
plt.scatter(data_input[:,2].cpu().detach().numpy(), data_output_mhp[:,3].cpu().detach().numpy(), s=40, label="Pre_MHP", facecolors='peru', edgecolors="darkorange", marker="s")
plt.scatter(data_input[:,2].cpu().detach().numpy(), data_output_ramlp[:,3].cpu().detach().numpy(), s=40, label="Pre_RAMLP", facecolors='lightblue', edgecolors="blue")
if y_loc == 0.3:
    plt.xlim(0.1, 1.2)
    plt.xticks(np.arange(0.1, 1.2, 0.2))
    plt.ylim(-1.5, 1)
elif y_loc == 0.6:
    plt.xlim(0.3, 1.2)
    plt.ylim(-1.75, 1.0)
elif y_loc == 0.9:
    plt.xlim(0.5, 1.3)
    plt.ylim(-1.75, 1.0)
elif y_loc == 1.2:
    plt.xlim(0.6, 1.4)
    plt.ylim(-1.8, 1)
plt.plot(fluent_data[:,0], fluent_data[:,1], '--', linewidth=2, label="Ref", color="m")
plt.legend(fontsize=16)
# plt.savefig('C:\\Users\86188\Desktop\de_wing\\test\curves\cp\\naca4421_0.4_-4/naca4421_0.4_-4_0.3.jpeg', format='jpeg', dpi=300)
# plt.savefig('C:\\Users\86188\Desktop\de_wing\\test\curves\cp\\naca4421_0.4_-4/naca4421_0.4_-4_0.3.tiff', format='tiff', dpi=300)
plt.show()
