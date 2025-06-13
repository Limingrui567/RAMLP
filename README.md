# RAMLP
![image](https://github.com/user-attachments/assets/fdf4ad1d-2d23-435b-93f7-2ecc0874e0bb)

Fig.1. Schematic diagram of Autoencoder model structure: (a) Main architecture. (b) Residual block.

Autoencoder consists of three main components: the encoder, MLP, and decoder，as shown in Fig.(a).
Both the encoder and decoder are CNN with residual blocks35 and the structure of the residual block is shown in Fig.(b).
Description of files related to the AE:
1. The train_AE.py script is used to train the predefined autoencoder (AE) model, which is defined in model_AE.py;
2. After training, the model is saved as model_AE.pth. The current model_AE.pth is a pretrained model;
3. We provide a minimal sample dataset for training and validation. Due to the large file size, the dataset has been uploaded to the Releases section；
4. The script SSIM.py is used to compute the corresponding SSIM values to evaluate the similarity between the input and output point cloud data of the AE model.
   
![image](https://github.com/user-attachments/assets/70378273-592a-4d31-aa5d-18d74be7d30d)

Fig.2. Schematic diagram of MLP model structure.

![image](https://github.com/user-attachments/assets/791e8993-fbd8-4b2f-a622-dbf76db296cc)

Fig.3. Schematic diagram of MHP model structure.

![image](https://github.com/user-attachments/assets/d6b4d755-c3b3-42de-86c2-298fa581d9e5)

Fig.4. Schematic diagram of RAMLP model structure: (a) Main architecture. (b) SE block.

A novel fully connected neural network architecture, the residual-attention multilayer perceptron (RAMLP), is proposed, and its main structure is shown in Fig. 4(a). The architecture includes the squeeze-and-excitation (SE) blocks, as shown in Fig. 4(b). 
Description of files related to the MLP, MHP and RAMLP:
1. The train_MLP.py, train_MHP_U.py, train_MHP_V.py, train_MHP_W.py, train_MHP_CP.py, and train_RAMLP.py scripts are used to train the predefined MLP, MHP and RAMLP model, And the corresponding model is directly defined within the above script
2. After training, the model is saved as model_xx.pth. The current model_xx.pth is a pretrained model;
3. We have uploaded a minimal sample dataset used for training in the Releases section；
4. We have also uploaded a pretrained loss file named tra_losses_xx.pth，and the loss curves can be directly generated using the loss_curves.py script;
5. To generate the contour plots, you first need to run the generate_contour_data.py script to produce the required data, and then use plot_contour.py to visualize and save the contour plots；
6. 

