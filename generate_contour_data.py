import torch
import numpy as np
from torch import nn
from matplotlib.path import Path
from coord2CST import coord_2_CST
from CST2coord import CST_2_coord

torch.cuda.empty_cache()

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
        self.layers.append(nn.SiLU())

        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.SiLU())

        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Squeeze-and-Excitation (SE) block
class SEBlock(nn.Module):
    def __init__(self, hidden_dim, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // reduction),
            nn.SiLU(),
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
        self.activation = nn.SiLU()

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer, attn in zip(self.hidden_layers, self.attention_layers):
            residual = x  
            x = self.activation(layer(x))
            x = attn(x)
            x = x + residual
        x = self.output_layer(x)
        return x

path_modol = "model_MLP.pth"
path_latent_relu_6 = "latent_train_6.pt"

#load model
model = torch.load(path_modol).to(torch.float32)
latent_relu_6 = torch.load(path_latent_relu_6)

def normalize_tensor_1(tensor):

    max_vals = torch.max(tensor, 0, keepdim=True)[0]
    min_vals = torch.min(tensor, 0, keepdim=True)[0]

    normalized_arr = 2 * (tensor - min_vals) / (max_vals - min_vals) - 1

    return normalized_arr

def normalize_tensor(tensor, max_vals, min_vals):

    normalized_arr = 2 * (tensor - min_vals) / (max_vals - min_vals) - 1

    return normalized_arr

latent_max = torch.tensor([[24.7941, 13.7290, 18.5239, 39.1042, 60.7303, 16.5290]],
       device='cuda:0')
latent_min = torch.tensor([[ 9.1243,  2.2724, 11.8411,  6.5119, 23.0061,  6.7900]],
       device='cuda:0')

# Normalize the latent variables
latent_relu_6 = normalize_tensor(latent_relu_6, latent_max, latent_min)

# Define the directory for the root airfoil coordinates
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

# Obtain airfoil cross-section coordinates
y_loc, Ma, AOA, num_wing = 0.6, 0.5, 5, 1  #num_wing indicates the index of the airfoil; the index for NACA1410 is 1, and the index for NACA4421 is 3
data_coords = get_coords(path_line)
data_coords = data_coords.to(device) * (1 - y_loc * 0.5 / 1.5)
data_coords[:,0] = data_coords[:,0] + np.tan(np.pi/6) * y_loc
data_x = data_coords[:,0].reshape(-1, 1)
data_z = data_coords[:,2].reshape(-1, 1)

# Obtain the maximum and minimum values of the inputs and outputs for normalization
input_max = torch.tensor([[0.6000, 5.0000, 1.45, 1.5, 0.2]], device=device)
input_min = torch.tensor([[2.0000e-01, -5.0000e+00, -0.1, -3.4106e-16, -0.2]], device=device)
output_max = torch.tensor([[308.1072, 238.9804, 241.2216, 1.0684]], device=device)
output_min = torch.tensor([[-141.6986, -160.2610, -277.6957, -3.0824]], device=device)

# Set the range of the x-axis and y-axis for the output contour plot
if y_loc == 0.3:
    rang_x, rang_z, num_mesh = [0.05, 1.2], [-0.18, 0.18], 400
elif y_loc == 0.6:
    rang_x, rang_z, num_mesh = [0.25, 1.25], [-0.16, 0.16], 400
elif y_loc == 0.9:
    rang_x, rang_z, num_mesh = [0.4, 1.3], [-0.14, 0.14], 400
elif y_loc == 1.2:
    rang_x, rang_z, num_mesh = [0.6, 1.4], [-0.12, 0.12], 400

# Generate all mesh points
x = torch.linspace(rang_x[0], rang_x[1], num_mesh).reshape(num_mesh, 1)
z = torch.linspace(rang_z[0], rang_z[1], num_mesh).reshape(num_mesh, 1)
expand_x = x.repeat_interleave(z.size(0), dim=0)  # (640000, 1)
expand_z = z.repeat(x.size(0), 1)  # (640000, 1)
expand_x_z = torch.cat((expand_x, expand_z), 1)  # (640000, 2)
print(expand_x_z.shape)

# Generate the complete input data
data_y_ = (torch.ones((expand_x_z.shape[0], 1)) * y_loc).to(device)
data_ma_ = (torch.ones((expand_x_z.shape[0], 1)) * Ma).to(device)
data_aoa_ = (torch.ones((expand_x_z.shape[0], 1)) * AOA).to(device)

# Combine the inputs of all mesh points
input_data_ = torch.cat((data_ma_, data_aoa_, expand_x_z[:, 0].reshape(-1, 1).to(device),
                         data_y_, expand_x_z[:, 1].reshape(-1, 1).to(device)), 1)

# Normalization
input_data_ = normalize_tensor(input_data_, input_max, input_min)

# Add geometric feature encoding
data_latent_6_ = latent_relu_6[num_wing].unsqueeze(0).expand(data_y_.size(0), -1).to(device)
input_data_ = torch.cat((input_data_, data_latent_6_), 1)

# Compute the model outputs for all mesh points
data_output_ = model(input_data_)

# Obtain flow field data
u = data_output_[:, 0].reshape(-1, 1) # 0 represents velocity U, 1 represents velocity V, 2 represents velocity W, and 3 represents Cp

# Generate the airfoil boundary path and remove interior points
boundary_wing = torch.cat((data_x, data_z), 1)
polygon_path = Path(boundary_wing.cpu())
inside = polygon_path.contains_points(expand_x_z)

# Filter out the points inside the airfoil and their corresponding data
filtered_points = expand_x_z[~inside]
filtered_u = u[~inside]
print(filtered_u.shape)

# Merge x, z, and flow field data and save
all_data = torch.cat((filtered_points.to(device), filtered_u), 1)
np.savetxt("data_contour.txt", all_data.cpu().detach().numpy(), fmt="%.6f", delimiter=" ")



