# Steps to obtain data:
* scrape data, `scrape_ascii_art.py`, downloads txt files
* filter data, `filter_data.py`, deletes large and small files, deletes files with tabs
* (Optional) pad data, `pad_data.py`, pads each line in each file with spaces up to the maximum line length.

# Pre Train model
* Batch size 128, z dim 128, resolution 64x64, 4240 epochs, one hot encoding and adversarial loss [mirror1](https://adamcolton.info/publicfiles/adversarial_autoenc.tar.gz).


# PCA Character Embeddings
Instead of using the square ascii art pieces as image data input, I generated character embeddings based on the character's appearance. Characters that look similar, are less distant in the embedding space than non similar looking characters. I performed PCA on images of the 95 ascii characters. This 2D representation shows the distribution of characters.

![Embedded characters on 2d plane](/figures/2d character embeddings.png)

*Steps to reproduce this plot*
* First generate images of the 95 characters using the script `generate_characters.py`, this saves the images to the `character_embeddings/out` directory.
* Run the script `pca.py`

# One Hot Character Embeddings
This was the technique that I found allowed the model to produce the best results, albeit at a memory and training cost. Having 95 channels at each pixel was expensive, but using cross entropy loss, the model was able to select the proper character at each pixel with good accuracy.

# Autoencoder trained with one hot vector character encodings
I trained a simple convolutional autoencoder on the one hot character encodings with a resolution of 64x64x95, and a latent dimension of 256. After about 3,500 epochs, the model was able to successfully encode the all of the training data. Here are some reconstructed strings. 

```

                               __________
                    __________/VVVVVVVVVV\
                   /VVVVVVVVVVVVVVVVVVVVVV`
                 /VVVVVVVVVVVVVVVVVVVVVVV/
               /VVVVVVVVVVVVVVVVVVVVVVVV/
               VVVV^^^^^^^^^^^^         |
             |                    vvvvvv\
             |     vvvvvvvVVVVVVVVVVVVVV/
             |/VVVVVVVVVVVVVVVVVVVVVVVVV|
             |VVVVVVV^^^^^^^^^^         |
              |V/                        \
              |             vvvvvvvvvvvvv|
               \  /VVVVVVVVVVVVVVVVVVVVVV\
                \/VVVVVVVVVVVVVVVVVVVVVVVV\____
                  VVVVVVVV^^^^^^^^^^___________)
             |\__// ______//--------   \\  \
             | * \ /%%%%///   __     __  \\ \
             \_xxx %%%%  /   /  \   /  \    |
             / \*%%%%       ((o) ) () ) )   |
            /  /|%%%%        \__/   \__/     \__  ______-------
            \ / |%%%%             @@            \/
              _/%%%%                             |_____
     ________/|%%%%                              |    -----___
-----         |%%%%     \___                  __/
           ___/\%%%%    /  --________________//
     __---'     \%%%%                     ___/
    /             %%%%%                   _/
                     \%%%%              _/
                       \%%%%           /
                          \%%         |
                           `%%        |
```

```
                                    __..._
                                ..-'      '.
                             .-'            :
                         _..'             .'__..--
                  ...--""                 '-.
              ..-"                       __.'
            .'                  ___...--'
           ::       ____....---'
          ::      .'
         :                   _____
         :           _..--"""     """--..__
        :          ."                      ""---.
        :       ..:                         :    '.
        :         '--...___----""""--..___.'      :
         :                 ""---...---""          :
          :.                                     :
            '-.                                 :
               `--...
                     ""---....._____.....---""
                 '.    '.
                   '-..  '.
                       '.  :
                          .'
                          :
                     .    :
                   .' .--'

```

Even though the recreations were impressive, I thought that this model wasn't very impressive because all it had to do was overfit to the training data. I then trained a adversarial autoencoder, which introduced a discriminator loss to the encoder decoder block, with the purpose of getting the autoencoder to make realistic art depictions across the entire latent space. This model had a z dimension of 128, but was still capable of recreating the training data. The script `latent_space_explorer.py` demonstrated that the adversarial autoencoder had better looking art when interpolating linearly between different training data points. The interpolated representation of the latent space still didn't look like good ascii artwork, but it was an improvement over the first model I trained.

The adversarial model was more conservative about it's placement of rare characters, and tended not to overfit as aggressively as the native model. It was quick to learn horizontal and vertical lines, but struggled with recreating unusual characters.

# Autoencoder trained with PCA generated character embeddings

I attempted to train a cnn autoencoder with 8 dimension character embeddings generated by PCA. The PCA embeddings for the datapoints were scaled between 0 and 1. I used L2 loss to try to minimize the distance between the decoded arrays and the original 64x64x8 array. The array was turned back into a string by finding the nearest 8 length character vector for each pixel. This technique did not produce desireable results. The model was not able to achieve the precision that is required to pick the correct character, and would mostly place space characters everywhere.

# Gan Attempts

I also experimented with a convolutional GAN to try and generate realistic art pieces. The results were poor. 24,000 epochs of 8x64x64 network, trained with character embeddings. Most generated images were tiled `~` signs.

```
                                       ~                        
                                                                
                                                                
                                                                
                                ~                               
                                                                
                                                                
                                       ~                        
        ........        ~               ........                
         .......                        .......~                
        ........                        ........                
        .~......                        .......~                
        ........                        ...~...~                
        ...~..~.              ~         .......~                
        .~..~...              .         .~..  ..                
        .....~..              ..        .            .. ~.......
        .          .    ~       ~                               
                        .                                     ~~
                        .        .                            ~ 
            .           ~   ~                       .      .   ~
           .            ~.       .           .                ~ 
         ...  .         ~ ~.:~~                           ~   ~~
             ~.         ~~ ~ ~~      .                       .~ 
         ..   ..     .  ...~:.l        ~               >  .   ..
                        ~       ~      ~   ~                    
                        ~*      ~.~~~~                  ~       
          .  . ~~     ~    ~. %Q ~ ~~ ~~                        
           .>       ~   ~   G  ~~%~ ~~~   b ~     ~     ~       
           .   ~ {   ~     . .  -=~ ~~   >   %                  
           .          \ ~     .~ ~_   .                         
      ~       ~         . ~ ~ .   ~% ~~                         
~                                 ~  ~*Q               ~       .
........ ~~~~~~~    :  \ ~ .    ~      ~ ~ ~      **%   ~       
  **\   (%~~~~   {& "            %                 `   .        
        :~ ~               ~                                    
        :~                      ~         ~                     
        :~        ` `      %b   *~b b    ~ ~~ ~                 
                 .*   *   ~ ~~% ~   ~'b   \ ~ ~~                
        :   ~   -  ~ L:  %    +  k       %~<~<\                 
              <\~      .  .   :~       *  \~ ~\k.  ~   ~       .
.......~       ~~      ..      ~       ..           .           
.               ..    .          . ~ ~    ~     .               
...~..~        .. .               . ..         ..               
... ..      .  ...          ~    .. ..         .. ...           
.. ..  ~       ..~  .       .    ..  .. .      ..   .           
...~.~~~        .     ~       ~ ...  .~      . . .   .          
.~....~ .      ..             ~   . .. ..     ~..    ..         
 . . .~~      ... .  ..~          ~~ ~~~ .. .  .........~       
        .   ....               .                           .. . 
               .        .....           . .    .                
         ...   .        .                   .  .         .      
               .        .               . . .. .                
         .   . .         .  .           ..     .         .      
               .         .  .             .   ..                
               .        ..  . ~         .. .. ..                
                        ........               .         ~~~~~~~
                        ~  ..~.~~~~~~~~~         ~~  ~~.        
~~~~~~~~ .        ~ ~~~  .      ~~~~~~~~         ~~~ ~~~        
~~~~~~~~          ~~~~~         ~~~~~~~~          .   ~         
~~~~~~~~          ~~ ~~~    .   ~~~ ~~~~           ~  ~         
~~~ ~~~~         ~~ ~~~~.       ~~~~~~~~                        
~~~~~~~~         ~~~~ ~~      . ~~~~~ ~~                        
~~~~~~~~         ~~ ~~~~      . ~~~~~ ~            ~~           
~~~~~~~~         ~~~~~~~       .~~~~~~~~        ~~ ~ ~         .
```