#  1 encoder 
class EncoderLayer(tf.keras.layers.Layer):
   
    def __init__(self, d_model: int, num_heads: int, dff: int, dropout_rate: float=0.1):
        #  num_heads - number of self attention 
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
            )

        self.ffn = FeedForward(d_model, dff)

    def call(self, x: tf.Tensor) -> tf.Tensor:
       
        x = self.self_attention(x)
        x = self.ffn(x)
        return x
    


#   many  enocders
class Encoder(tf.keras.layers.Layer):
    
    def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int, vocab_size: int, input_length  , dropout_rate: float=0.1 ):
       
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = encoder_input(unique_words= vocab_size , input_length = input_length ,     dimensions = self.d_model  )

        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                        num_heads=num_heads,
                        dff=dff,
                        dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        #  x - input [ tokenized ]
        x = self.pos_embedding(x)  
        
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  
    
