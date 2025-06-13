# RAMLP
![image](https://github.com/user-attachments/assets/fdf4ad1d-2d23-435b-93f7-2ecc0874e0bb)

Autoencoder consists of three main components: the encoder, MLP, and decoder，as shown in Fig.(a).
Both the encoder and decoder are CNN with residual blocks35 and the structure of the residual block is shown in Fig.(b).
Description of files related to the AE:
1. The train_AE.py script is used to train the predefined autoencoder (AE) model, which is defined in model_AE.py;
2. After training, the model is saved as model_AE.pth. The current model_AE.pth is a pretrained model;
3. We provide a minimal sample dataset for training and validation. Due to the large file size, the dataset has been uploaded to the Releases section；
4. The script SSIM.py is used to compute the corresponding SSIM values to evaluate the similarity between the input and output point cloud data of the AE model.
