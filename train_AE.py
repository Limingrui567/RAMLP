import os
import sys
import time
import numpy as np
from model_AE import *
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

checkpoint_file = "checkpoint_train_AE"

# Use GPU for training if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.cuda.empty_cache()

# Set a random seed to ensure the reproducibility of the code
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  

set_seed(4)

def load_train_state(filename, model, optimizer, scheduler):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    tra_loss_record = checkpoint["tra_loss_record"]
    val_loss_record = checkpoint["val_loss_record"]
    return checkpoint["epoch"], model, optimizer, scheduler, tra_loss_record, val_loss_record

def save_train_state(filename, model, optimizer, scheduler, epoch, tra_loss_record, val_loss_record):
    chkpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "tra_loss_record": tra_loss_record,
        "val_loss_record": val_loss_record
    }
    torch.save(chkpt, filename)

# Hyperparameter settings
num_epochs = 500
batch_size = 10
learning_rate = 0.0001

# Load training and validation datasets
tra_inputs = torch.load("inputs_tra_AE.pth").cpu() # A five-dimensional tensor of shape n×1×80×40×80
val_inputs = torch.load("inputs_val_AE.pth").cpu()

tra_dataset = TensorDataset(tra_inputs)
val_dataset = TensorDataset(val_inputs)

# Create data loaders
train_loader = DataLoader(tra_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
tra_losses = []
val_losses = []
model = Autoencoder()
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Record time
train_time_per_epoch, test_time_per_epoch = [], []

if os.path.exists(checkpoint_file):
    start_epoch, model, optimizer, scheduler, tra_losses, val_losses = load_train_state(checkpoint_file, model, optimizer, scheduler)
else:
    start_epoch = 0

# Training
for epoch in range(start_epoch, num_epochs):
    model.train()
    loss1 = 0
    num_batch = 0
    epoch_start_time = time.time()
    for data in tqdm(train_loader, file=sys.stdout):
        batch = data[0].to(device)
        batch += 0.05 * torch.randn_like(batch)
        outputs, latent = model(batch)
        loss = F.mse_loss(outputs, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss1 += loss.item()
        num_batch += 1
    loss1 = loss1 / num_batch
    scheduler.step(loss1)
    tra_losses.append(loss1)
    train_time = time.time() - epoch_start_time
    train_time_per_epoch.append(train_time)

    model.eval()
    loss2 = 0
    num_batch = 0

    epoch_start_time = time.time()
    with torch.no_grad():
        for data in tqdm(val_loader, file=sys.stdout):
            batch = data[0].to(device)
            outputs, latent = model(batch)
            loss = F.mse_loss(outputs, batch)
            loss2 += loss.item()
            num_batch += 1
    test_time = time.time() - epoch_start_time
    test_time_per_epoch.append(test_time)
    loss2 = loss2 / num_batch
    val_losses.append(loss2)
    save_train_state(checkpoint_file, model, optimizer, scheduler, epoch + 1, tra_losses, val_losses)
    print(f'Epoch [{epoch+1}/{num_epochs}], train_Loss: {loss1:.10f}, train_time: {train_time:.2f}s, val_Loss: {loss2:.10f}, val_time: {test_time:.2f}s')

    torch.save(model, "model_AE.pth")
    torch.save(tra_losses, "tra_losses_AE.pth")
    torch.save(val_losses, "val_losses_AE.pth")

def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes}m {seconds:.2f}s"

tqdm.write(f'Total train time: {format_time(sum(train_time_per_epoch))}, Total test time: {format_time(sum(test_time_per_epoch))}')

