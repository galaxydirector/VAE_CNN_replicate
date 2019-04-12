# VAE_CNN_replicate

This project is to test performance of original [Variational Auto-Encoder paper](https://arxiv.org/pdf/1606.05908.pdf).

## There are three parts:

### Image Preprocessing
This part extract the faces from the selfie database and compress them into greyscale lower dimension images

### Model
Fully connected models written in Tensorflow
CNN models written in Tensorflow

### Tests
Two result images using only decoder. Z is chosen to be [-3,3] in two dimension.

30 epoch train
![alt text](https://github.com/galaxydirector/VAE_selfies/blob/master/img/recon3-30epoch.png)

130 epoch train
![alt text](https://github.com/galaxydirector/VAE_selfies/blob/master/img/recon4-130epoch.png)
