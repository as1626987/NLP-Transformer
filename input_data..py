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
from sklearn.model_selection  import train_test_split 

with open("", "rb") as f:
    integer_input = pickle.load(f)

with open("", "rb") as f:
    integer_input = pickle.load(f)

with open(
    r"/kaggle/input/datasets/ankushsharma1108/machine-translation/IITB.en-hi.en",
    "r",
    encoding="utf-8"
) as f:
    english_sentences = f.readlines()

with open(
    r"/kaggle/input/datasets/ankushsharma1108/machine-translation/IITB.en-hi.hi",
    "r",
    encoding="utf-8"
) as f:
    hindi_sentences = f.readlines()

input  = [line.replace('\n', '') for line in english_sentences]
output  = [line.replace('\n', '') for line in hindi_sentences]


inp = [] 
out= [] 


for i in range (  0  , len(input )):
    line = len(input[i].split()) 
    
    if ( line <=20 ):
        
        inp.append ( input[i])
        out.append (  output[i])

input = inp 
output = out

class data_input ( Sequence ):
    def __init__ ( self  , input , output ,  batch_size  ,  total_sentences , input_length  , output_length ,  integer_input  , integer_output ) :
        self.batch_size = batch_size
        self.length = input_length
        self.integer_input =  integer_input 
        self.integer_output =  integer_output
        self.sentences= total_sentences 
        self.length_output = output_length 
        self.input  = input 
        self.output = output 
        
    def __len__(self):
        #  it will return  number of batches of a dataset 
        return  int ( self.sentences   / self.batch_size)
    def __getitem__( self ,   index ):
         start = index  * self.batch_size
         end = ( index + 1 ) * self.batch_size 
         input_batch = self.input[ start : end ] 
         output_bat =self.output[start : end ] 
         input_batch = [
         re.sub(r'\s*\([^)]*\)$', '', sentence)
          for sentence in  input_batch
           ]
         output_batch = [
         re.sub(r'\s*\([^)]*\)$', '', sentence)
          for sentence in  output_bat
           ]
         input_batch = self.integer_input.texts_to_sequences(   input_batch   )
         input_batch = pad_sequences(  input_batch   ,padding='post' , maxlen  = self.length  )
         out= [] 
         for i in range ( 0 , len ( output_bat )):
              line = output_bat[i] 
              line = "[START] " +  line 
              out.append( line )
             
         output_batch = self.integer_output.texts_to_sequences(   out  )
         output_batch = pad_sequences(  output_batch   ,padding='post' , maxlen  = self.length_output   )


         target_out = [] 
         for i in range ( 0 , len ( output_bat )):
              line = output_bat[i] 
              line =  line  + " [END]"
              target_out.append( line )
             
         target_batch = self.integer_output.texts_to_sequences(   target_out  )
         target_batch = pad_sequences(   target_batch   ,padding='post' , maxlen  = self.length_output   )
       
         return (input_batch , output_batch) , target_batch

x_train , x_test , y_train , y_test = train_test_split( input , output ,  test_size = 0.05 , random_state = 3 )
input_length  =  20 
output_length =  320 
sentences = 1305510
input   = data_input(    x_train   ,  y_train ,   10      , len ( x_train )    , input_length ,  output_length,    integer_input , integer_output   ) 