import torch
from matplotlib import  pyplot as plt

# load model
MLP_losses = "/tra_losses_MLP.pth"
MHP_U_losses = "/tra_losses_MHP_U.pth"
MHP_V_losses = "/tra_losses_MHP_V.pth"
MHP_W_losses = "/tra_losses_MHP_W.pth"
MHP_Cp_losses = "/tra_losses_MHP_CP.pth"
RAMLP_losses = "/tra_losses_RAMLP.pth"

font_properties = {
    'family': 'serif',
    'weight': 'bold',
    'style': 'italic',
}

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

x_MLP = list(range(1, len(MLP_losses)+1))
x_U_MHP = list(range(1, len(MHP_U_losses)+1))
x_V_MHP = list(range(1, len(MHP_V_losses)+1))
x_W_MHP = list(range(1, len(MHP_W_losses)+1))
x_Cp_MHP = list(range(1, len(MHP_Cp_losses)+1))
x_RAMLP = list(range(1, len(RAMLP_losses)+1))
fig, ax = plt.subplots()
ax.plot(x_MLP, MLP_losses, label="MLP", color="c", linestyle="-", linewidth=2, antialiased=True)
ax.plot(x_U_MHP, MHP_U_losses, label="MHP_U", color="r", linestyle="-", linewidth=2, antialiased=True)
ax.plot(x_V_MHP, MHP_V_losses, label="MHP_V", color="g", linestyle="-", linewidth=2, antialiased=True)
ax.plot(x_W_MHP, MHP_W_losses, label="MHP_W", color="magenta", linestyle="-", linewidth=2, antialiased=True)
ax.plot(x_Cp_MHP, MHP_Cp_losses, label="MHP_Cp", color="m", linestyle="-", linewidth=2, antialiased=True)
ax.plot(x_RAMLP, RAMLP_losses, label="RAMLP", color="b", linestyle="-", linewidth=2, antialiased=True)

ax.set_yscale('log')

ax.set_xlim(0, 400)
ax.set_ylim()

ax.tick_params(axis='both', direction='in', which="both", labelsize=12)

ax.set_xlabel("epoch", fontsize=14, fontdict=font_properties)    
ax.set_ylabel("loss", fontsize=14, fontdict=font_properties)    
legend = ax.legend(loc='upper right', prop={'family': 'serif', 'weight': 'bold', 'style': 'italic'})  
for text in legend.get_texts():
    text.set_fontsize(14)

# plt.savefig("tra_loss.tiff",  format='tiff', dpi=300)
# plt.savefig("tra_loss.jpeg",  format='jpeg', dpi=300)

plt.show()
