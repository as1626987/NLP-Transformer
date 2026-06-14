def Transformer(
    input_vocab_size: int, 
    target_vocab_size: int, 
    encoder_input_size: int = None,
    decoder_input_size: int = None,
    num_layers: int=1, 
    d_model: int= 5, 
    num_heads: int=1,
    dff: int=2048,
    dropout_rate: float=0.1,
    ) -> tf.keras.Model:
   
    inputs = [
        tf.keras.layers.Input(shape=(encoder_input_size,), dtype=tf.int64), 
        tf.keras.layers.Input(shape=(decoder_input_size,), dtype=tf.int64)
        ]
    
    encoder_input, decoder_input = inputs


    encoder = Encoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, vocab_size=input_vocab_size, dropout_rate=dropout_rate , input_length = encoder_input_size )(encoder_input)
    decoder = Decoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, vocab_size=target_vocab_size, dropout_rate=dropout_rate,input_length =  decoder_input_size)(decoder_input, encoder)

    output = tf.keras.layers.Dense(target_vocab_size)(decoder)


    return tf.keras.Model(inputs=inputs, outputs=output)

model = Transformer( input_vocab_size=len(integer_input.word_counts) , target_vocab_size=len(integer_output.word_counts)-1,encoder_input_size = 20  , decoder_input_size =  320    )

model.compile(
    optimizer="adam",
    loss=MaskedLoss(),
    metrics=[MaskedAccuracy()]
)

model.fit( input     ,  epochs  = 10  )