# Steps to obtain data:
* scrape data, `scrape_ascii_art.py`, downloads txt files
* filter data, `filter_data.py`, deletes large and small files, deletes files with tabs
* pad data, `pad_data.py`, pads each line in each file with spaces up to the maximum line length.


# Plan
Instead of using the square ascii art pieces as image data input, I want to generate character embeddings based on the character's appearance. Characters that look similar, should be less distant in the embedding space than non similar looking characters. 

# Character Embedding Ideas
* PCA on character font images
* Embeddings are length 2 vectors for each character,

24,000 Iterations of 8x64x64 network:
	LR: 5E-5
	
	Experienced mode collapse - most generated images were tiled `~` signs:


### Example of mode collapse: similar artworks were produced, with different variations on this tiling theme.
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

```
....... ~~~~~~~                         ~~~~~~~~......          
        ~...~~~. .  . .  .      ..      ~~~~~~~~                
        ~~~~~~~. ..  .           . .    ~~~~~~~.   .. .         
.       ~~~~~~~. .....          .       ~~~~~~~.  ... .         
        ~~~~~~~.. ....          ..      ~~~~~~~~   . .          
.       ~~.~~.~. .... .                 ~~~~~~~..               
.       ~~~~~~~.  ... ..        .       ~~~~~~~~.  . .          
        ~~~~~~~~.  . ..         ....... ~    ...        ~       
                         ~ ~~            ~~~~~~~                
         ~   ~ ~                         .~.~  ~                
         ~~ ~~~                          ..~                    
         ~ ~ ~                            ~ .~~                 
         ~~ ~ ~~                         ~~~   ~               ~
         ~ ~  ~~                         ~~ ~ ~                 
          ~ ~ ~~                         ~    ~~                
               ~                ~       .~~.~ ~         ~     ~~
                           .    ~                               
                        ..                                      
                        ..                                      
                                              .                 
                          ~      .             .                
  ~                       ~                                 ~   
               .                     :~                         
                       ,       ~      .... . ..        .        
        ..  ... . . .  ..~ :     c .   ..+...... .      ...   . 
        ........ % . . ..{...... { c  . .((.....        ........
         .......~. .  L :&~~.... {6%6b~ %G.:bb~..       .... ...
        .  .....        %{..%..  {..    ..~+....        .... .  
        ....   . .      .%~   ~~'%      .%......        ... ....
        ........      . c:. ..%~      . ........        ..  ... 
        ...... .        ..... .         ........ ~      ........
        ~......        ..    ~~.        .........     .        .
........... .~~~ [.[    .......  _     ... ,..  ..   ...        
        .    ~...{l[......;. .. :{..   ........ ... ....        
        ..   .   .... . ..  .%  %  .    ....... ........        
        . . .   .. .... .       :       ......   .......        
        ..       .. .  . .      ;%      ....... ..... ..  .   . 
        .   . ~ ..$be:..  .   ~   .   c ....... ... ...       ~ 
    ~-  .. ...:...*;t. ...       %      ..... . ..... .. .      
        ........ .... ..      ~ _      _        . . ....        
                ~...~  ... .   ~       ~                        
        .      . .     .  .     ~                               
        .      .   ~      ..    ~       .      .          .     
        .                  . .   .             .                
        .      .          .  .          .      .                
        .               .. . .                                  
        .                 .  ..         .                       
                       .      ..        ..       .  ... ~       
                .         ..   .  .....               ..       .
                 .. ...   .     . . ....             .          
 .  .           .. . .           .......              .        .
   . ...        .... ..              . .        .               
 .. .           ..   .            .... .                      ..
  ....          . .  .    .        .  ..        .               
  ....          .                 . .  .        .. ..          .
 . ...          ..        .   .. .......                        
                        ~       .      ~                        
.                                                  .            
                                                                
                                                                
                                                                
                                                                
                                                                
~                                               ~  . .          
```


# Full vector encoding of character information
* As an alternative to PCA decomposed space of the character images, the entire list of characters can be input as a vector, with a 1 in the position of the character.