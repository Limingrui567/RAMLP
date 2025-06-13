import os
import sys
import time
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

# If a GPU is available, use it for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set the random seed to ensure code reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(4)

# Squeeze-and-Excitation (SE) block
class SEBlock(nn.Module):
    def __init__(self, hidden_dim, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // reduction),
            nn.SiLU(),
            nn.Linear(hidden_dim // reduction, hidden_dim),
            nn.Sigmoid()  # Normalize
        )

    def forward(self, x):
        batch_size, channels = x.shape
        x_se = self.global_avg_pool(x.unsqueeze(-1)).view(batch_size, channels) # Global pooling
        attention_weights = self.fc(x_se)  # Compute channel weights
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
            x = x + residual  # Introduce residual
        x = self.output_layer(x)
        return x

# Load training data
data_path = "D:/pycharm_project/MHP/Github"
path_input = os.path.join(data_path, "input.pt")
path_output = os.path.join(data_path, "output.pt")
input = torch.load(path_input).detach().to(device).to(torch.float32)
output = torch.load(path_output).detach().to(device).to(torch.float32)

# Split the dataset into training and validation sets
train_input = input
train_output = output

train_size = int(0.8 * input.shape[0])  # 80% training
val_size = input.shape[0] - train_size  # 20% validation

dataset= TensorDataset(train_input, train_output)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Set hyperparameters
num_epochs = 400
batch_size = 1000
learning_rate = 0.0005

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, optimizer, scheduler
tra_losses = []
val_losses = []
model = RAMLP()
model = model.to(device)
criterion = nn.MSELoss(reduction="mean")
train_time_per_epoch, test_time_per_epoch = [], []
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

# Save a checkpoint for resuming from the last stopping point next time
checkpoint_file = os.path.join(data_path, "checkpoint_RAMLP")

def load_train_state(filename, model, optimizer, scheduler):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    loss_record = checkpoint["loss_record"]
    return checkpoint["epoch"], model, optimizer, scheduler, loss_record

def save_train_state(filename, model, optimizer, scheduler, epoch, loss_record):
    chkpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "loss_record": loss_record
    }
    torch.save(chkpt, filename)

if os.path.exists(checkpoint_file):
    start_epoch, model_MLP, optimizer, scheduler, tra_losses = load_train_state(checkpoint_file, model, optimizer, scheduler)
else:
    start_epoch = 0

# Train the model
for epoch in range(start_epoch, num_epochs):
    model.train()
    loss1 = 0
    num_batch = 0
    epoch_start_time = time.time()
    for data_train_input, data_train_output in tqdm(train_loader, file=sys.stdout):
        data_train_input = data_train_input.to(device)
        data_train_output = data_train_output.to(device)
        outputs = model(data_train_input)
        loss = criterion(outputs, data_train_output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss1 += loss.item()
        num_batch += 1
    scheduler.step()
    train_time = time.time() - epoch_start_time
    train_time_per_epoch.append(train_time)
    loss1 = loss1 / num_batch
    tra_losses.append(loss1)
    save_train_state(checkpoint_file, model, optimizer, scheduler, epoch + 1, tra_losses)

    model.eval()
    loss2 = 0
    num_batch = 0

    epoch_start_time = time.time()
    with torch.no_grad():
        for data_input, data_output in tqdm(val_loader, file=sys.stdout):
            outputs = model(data_input)
            loss = criterion(outputs, data_output)
            loss2 += loss.item()
            num_batch += 1
    test_time = time.time() - epoch_start_time
    test_time_per_epoch.append(test_time)
    loss2 = loss2 / num_batch
    val_losses.append(loss2)
    tqdm.write(f"Epoch [{epoch+1}/{num_epochs}], train_Loss: {loss1:.10f}, train_time: {train_time:.2f}s, val_Loss: {loss2:.10f}, val_time: {test_time:.2f}s")
    torch.save(model, os.path.join(data_path, "model_RAMLP.pth"))
    torch.save(tra_losses, os.path.join(data_path, "tra_losses_RAMLP.pth"))
    torch.save(val_losses, os.path.join(data_path, "val_losses_RAMLP.pth"))

def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes}m {seconds:.2f}s"

tqdm.write(f'Total train time: {format_time(sum(train_time_per_epoch))}, Total validation time: {format_time(sum(test_time_per_epoch))}')




