import torch 
import bpdb
from dataset import AsciiArtDataset
import tqdm


def calculate_character_frequencies(dataset: AsciiArtDataset, device=torch.device('cuda')):
    """
    Goes through every character in every artwork in the training dataset and counts it

    Returns a 95 length tensor of the character counts
    """

    counts = torch.zeros(95, dtype=torch.int32)

    print("Loading character frequencies")
    for x, label in tqdm.tqdm(dataset):
        x = torch.IntTensor(x)
        character_indeces = x.argmax(0)
        x_char_counts = torch.bincount(character_indeces.flatten(), minlength=95)
        counts += x_char_counts


    return counts
        
def get_char_weights_from_space_deemphasis(space_loss_deemphasis: float) -> torch.Tensor:
        """
        Useful for generating class weights for CE loss, or NLL loss

        space_loss_deemphasis of greater than one makes the space characters
        have less wieght in the loss calculation. A space_loss_deemphasis of
        less than one makes them have more weight. 
        """
        char_weights = torch.zeros(95)
        # emphasis on space characters
        char_weights[0] = 1 / 95 / space_loss_deemphasis
        char_weights[1:] = (1 - char_weights[0]) / 94
