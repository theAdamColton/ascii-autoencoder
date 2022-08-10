# VAE Autoencoder

The goal is a continuous transfer function from the one hot to the rendered ASCII image. This may be achieved by using a model that can translate a slightly noisy one hot representation into the rendered image. Call this onehot2raa. Freeze onehot2raa during training of the one hot vae. The reconstruction loss of the vae will be the output of onehot2raa compared to the baseline rendered image.


### cb2c7dd50f251e700ee6a8ad22e63917bc5a4f3b: model `./models/16precision_random_roll_with_im_loss_nz256_with_char_weightcheckpoint`

* This model was trained for about 1500 epochs
* Trained using CE loss with character weights, `(1/char_freq)**0.7`
* Also included some varying image loss using the continuous renderer
* It was able to start to generate some patterns that matched the shape and characters of the input image, but every reconstruction was very splotchy and vague.
* 'Images' containing lots of characters, being outliers, were actually predicted very well by the model
* Images containing mostly whitespace in comparison, were usually not well done


### commit: model `models/stage0_gumbel`

* This model was trained using these args: `bpython train.py --renderer-type continuous --ce-recon-loss-scale 0.0 --kl-coeff 0.05 --image-recon-loss-coeff 100.0 --n-workers 16 --validation-prop 0.05 --learning-rate 4e-5 -b 128 -n 500 --run-name stage0_gumbel --print-every 2 --gumbel-tau 0.5`
* After about 90 epochs, the model reconstructed spaces everywhere for most samples
* It was difficult looking at a plot of the loss over time to determine that the loss was decreasing.
