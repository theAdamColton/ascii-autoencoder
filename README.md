# How to train a model

* Clone this repo: `git clone --recursive https://github.com/theAdamColton/ascii-autoencoder/tree/image-based-vae`
* Obtain the txt files required for the ascii art dataset. Instructions for this can be found in `./ascii-dataset/data_aggregation/`, basically there are some scripts you can run to scrape the text files from the internet.
* Start the trainer. If you want to use the discrete ascii art renderer, you can use this command to start training with some defaults:
	* `bpython train.py --should-discrete-render --run-name models/discrete_render --validation-prop 0.05 --print-every 2 --learning-rate 7e-4 -b 64 --ce-recon-loss-scale 0.05`
