import numpy as np 
import tensorflow as tf 
class encoder_input ( tf.keras.layers.Layer) : 

    def __init__ ( self  ,  unique_words , input_length  , dimensions   ):
        super().__init__() 
        self.unique_words =unique_words  
        self.number_of_words = input_length 
        self.dimensions = dimensions
        self.embedding = tf.keras.layers.Embedding( input_dim =  self.unique_words , output_dim = self.dimensions  , input_length = self.number_of_words    )
        self.positional_encoding = self.position(input_length , dimensions   )  
     
    def position(  self ,    input_length , dimensions   ):
        
            sentence = [] 
            for i in range ( 0 ,   input_length):
               position = i 
               word = [] 
               start = 0 
               count = 0 
               while ( count < dimensions ):
                    power = ( ( 2 * start )/ dimensions)
                    num = np.sin(  position / 10000**power )
                    word.append ( num )
                    count+=1 
                    if ( count >= dimensions ):
                         break 
                    num = np.cos(  position / 10000**power )
                    word.append ( num )
                    count += 1   
                    start += 1 
               sentence.append ( word )
            
            return  tf.convert_to_tensor( sentence, dtype=tf.float32)
    def call ( self  , input ):
        # input word  should be in  integer [ tokenize ]
        x = self.embedding(input )
        y = self.positional_encoding[tf.newaxis, :, :]
        return x + y 

