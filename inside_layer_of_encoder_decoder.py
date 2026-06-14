from tensorflow.keras.utils import pad_sequences
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Embedding  , Input
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from mltu.tensorflow.transformer.utils import MaskedAccuracy, MaskedLoss
from tensorflow.keras.utils import Sequence
import pickle 
import re 
class BaseAttention(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs: dict):
        
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

#  cross attention layer 
class CrossAttention(BaseAttention):
   
    def call(self, x: tf.Tensor, context: tf.Tensor) -> tf.Tensor:
        #  context - encoder outputs
        #  x - decoder inputs 
        attn_output, attn_scores = self.mha(query=x, key=context, value=context, return_attention_scores=True)

       
        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x
#  simple multihead attention layer 
class GlobalSelfAttention(BaseAttention):
   
    def call(self, x: tf.Tensor) -> tf.Tensor:
        #  x - encoder inputs 
        attn_output = self.mha(query=x, value=x, key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x
#  masked self attention layer 
class CausalSelfAttention(BaseAttention):
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        #  x -  decoder inputs 
        attn_output = self.mha(query=x, value=x, key=x, use_causal_mask = True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x
    
#   ANN layer 
class FeedForward(tf.keras.layers.Layer):
   
    def __init__(self, d_model: int, dff: int, dropout_rate: float=0.1):
        #  dff - 2048 
        #  d_model - word  vector size 
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x) 
        return x
    


