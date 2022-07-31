# VAE Autoencoder

The goal is a continuous transfer function from the one hot to the rendered ASCII image. This may be achieved by using a model that can translate a slightly noisy one hot representation into the rendered image. Call this onehot2raa. Freeze onehot2raa during training of the one hot vae. The reconstruction loss of the vae will be the output of onehot2raa compared to the baseline rendered image.
