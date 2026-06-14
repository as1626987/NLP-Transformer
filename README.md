NLP Transformer From Scratch

This project implements a complete Transformer architecture from scratch using Python for Natural Language Processing (NLP) tasks. The implementation includes all major components of the original Transformer model, such as embeddings, positional encoding, encoder layers, decoder layers, multi-head attention, and inference.


<img width="492" height="697" alt="image" src="https://github.com/user-attachments/assets/a8f803ab-58c9-4015-b486-03376476dc84" />


Features
Transformer architecture implemented from scratch
Encoder and Decoder modules
Positional Encoding
Tokenization and vocabulary handling
Inference pipeline for generating predictions
Modular and easy-to-understand code structure
Educational implementation for learning Transformer internals
Project Structure
├── embedding_position.py      # Token embedding and positional encoding
├── encoder_layer.py           # Transformer encoder layer
├── decoder_layer.py           # Transformer decoder layer
├── inside_layer_of_encoder_decoder.py  # Internal attention and feed-forward blocks
├── transformer_model.py       # Complete Transformer model
├── input_data.py              # Data preprocessing and loading
├── input_tokenizer.pkl        # Input tokenizer
├── output_tokenizer.pkl       # Output tokenizer
├── inference.py               # Model inference and prediction
Transformer Architecture

The model follows the architecture proposed in the paper:

"Attention Is All You Need" (Vaswani et al., 2017)

Key components include:

Multi-Head Self Attention
Positional Encoding
Feed Forward Neural Networks
Residual Connections
Layer Normalization
Encoder-Decoder Attention
Technologies Used
Python
NumPy
TensorFlow / Keras (if used)
Pickle for tokenizer serialization
Purpose

This project was developed to gain a deep understanding of how Transformer models work internally rather than relying on pre-built libraries. It demonstrates the implementation of attention mechanisms and sequence-to-sequence modeling from the ground up.

Future Improvements
Training on larger datasets
Beam Search decoding
Attention visualization
Support for pre-trained embeddings
Extension to BERT and GPT-style architectures


IMPORTANT LINKS  :
Dataset : https://www.cfilt.iitb.ac.in/iitb_parallel/dataset.html




