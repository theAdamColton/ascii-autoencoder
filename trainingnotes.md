# VAE Autoencoder

# Why does the training reconstruction loss not show precision past 3 decimal places?

I've gone in through the debugger and the loss is higher precision, from stepping through some interations after resuming a model.

What I've tried so far:

* Character weights
	* Give the space character less emphasis in the loss function because spaces are very common
		* This works at stopping the model from predictings spaces everywhere
		* The predicted characters are now all '_', '.', and '|' characters
	* Give the characters emphasis inversely proportional to the frequency in the dataset

* Decoder sigmoid layer in conjunction with CE loss

* CE Loss
	* Weird artiface where the loss doesn't go past 3 decimal places

* NLLLoss
	* Used in conjunction with gumbel softmax, the loss goes past the precision of 3 decimal places

* Gumbel softmax along dim 1 of the output of the decoder
	* Predicted space characters everywhere
	* Loss did not go past 2 decimal places, and instead looked like it was rounded
* Plain softmax along dim 1 of the output of the decoder
	* Predicted space characters everywhere
	* Loss did not go past 2 decimal places, and instead looked like it was rounded
* No softmax
	* Fastest decrease in loss so far
	* Trained for a while, loss eventually converges, predicting space characters everywhere
* Sigmoid

* Weight initialization
	* u of 0 and std of .1 up to .2
	* STD of 0.3 produces invalid values
