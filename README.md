# How to train a model

* Clone this repo: `git clone --recursive https://github.com/theAdamColton/ascii-autoencoder/tree/image-based-vae`
* Obtain the txt files required for the ascii art dataset. Instructions for this can be found in `./ascii-dataset/data_aggregation/`, basically there are some scripts you can run to scrape the text files from the internet.
* Start the trainer. If you want to use the discrete ascii art renderer, you can use this command to start training with some defaults:
	* `bpython train.py --should-discrete-render --run-name models/discrete_render --validation-prop 0.05 --print-every 2 --learning-rate 7e-4 -b 64 --ce-recon-loss-scale 0.05`


# Training notes

`bpython train.py --run-name models/16precision_image_losstest --n-workers 16 --datapath /dev/shm/ascii-data/ --validation-prop 0.05 --print-every 1 --learning-rate 1e-4 -b 32 --ce-recon-loss-scale 0.0 --image-recon-loss-coeff 1.0 --char-weights-scaling 0.001 --kl-coef 0.05 -n 10000  --neural-renderer-path ../pytorch-neural-font-renderer/models/9_kernel_randcheckpoint/lightning_logs/version_2/checkpoints/epoch=9-step=13150.ckpt`

* Only image loss using neural renderer
* Noticeably, this does not produce only space characters in the output; exclusively using image reconstruction loss seems to stop the model from doing this. The 'shape' of the characters in the decoded output does seem to be semantically consistent with the input. It is important when using image reconstruction loss that the transformation to the rendered image is differentiable, which means using the pytorch-neural-font-renderer instead of the discrete neural renderer. 
