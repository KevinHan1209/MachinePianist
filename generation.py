import torch
from transformer import MusicTransformer
from build_vocabulary import VOCABULARY_SIZE
from pre_processing import TOKEN_TO_INDEX, INDEX_TO_TOKEN
from midi_tokenizer import tokens_to_midi

import config

# Configuration parameters (same as training)
seq_len = config.SEQ_LEN  # Sequence length of the model
vocab_size = VOCABULARY_SIZE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_size = config.EMBED_SIZE
num_heads = config.NUM_HEADS
num_layers  = config.NUM_LAYERS
hidden_dim = config.HIDDEN_DIM

max_length = 1024 * 4 # Length of new generation

# Load your trained model
model = MusicTransformer(vocab_size=vocab_size, embed_size=embed_size, num_heads=num_heads, 
                         num_layers=num_layers, hidden_dim=hidden_dim, seq_len=seq_len-1)
model.load_state_dict(torch.load('models/checkpoints/music_transformer_run1.pth'))
model.to(device)
model.eval()  # Set to evaluation mode

# Function to generate tokens using sliding window attention
def generate_with_sliding_window(model, start_sequence, max_length=max_length, window_size=512, overlap_size=256):
    """
    Generate tokens with a sliding window mechanism.

    model: The trained model
    start_sequence: A tensor of initial tokens (of length <= window_size)
    max_length: Maximum length of generated sequence
    window_size: The size of the sliding window
    overlap_size: The number of tokens that overlap between consecutive windows
    """
    generated_sequence = start_sequence  # Start with the initial sequence
    current_window = start_sequence  # Current window of input to the model
    
    while generated_sequence.size(1) < max_length:
        # Get the output from the model
        with torch.no_grad():
            output = model(current_window.to(device))  # Output from the model

        # Get the predicted next token (this assumes the model outputs logits for each token in the vocabulary)
        next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(1)  # Get the most likely next token

        # Append the predicted token to the generated sequence
        generated_sequence = torch.cat((generated_sequence, next_token), dim=1)

        # Slide the window
        if generated_sequence.size(1) > window_size:
            # Keep the overlap between the current window and the new tokens
            current_window = generated_sequence[:, -window_size + overlap_size:]  # Last `window_size - overlap_size` tokens

    return generated_sequence

# Function to read tokens from the file and convert them to indices
def read_tokens_from_file(file_path):
    tokens = []
    with open(file_path, 'r') as f:
        for line in f:
            token = line.strip()  # Remove any leading/trailing whitespace
            if token in TOKEN_TO_INDEX:
                tokens.append(TOKEN_TO_INDEX[token])
            else:
                print(f"Warning: Token '{token}' not found in TOKEN_TO_INDEX.")
    return torch.tensor(tokens).unsqueeze(0).to(device)  # Add batch dimension

file_path = 'Chopin_Tokens/Chopin, Frédéric, Nocturnes, Op.48, -7mntyrW3HU.mid_tokens.txt'

# Read the start sequence from the file
start_sequence = read_tokens_from_file(file_path)

# Generate a new sequence
generated_sequence = generate_with_sliding_window(model, start_sequence, max_length=max_length)

# Convert the generated sequence from token indices back to tokens
generated_tokens = [INDEX_TO_TOKEN[token.item()] for token in generated_sequence.squeeze(0)]

tokens_to_midi(generated_tokens, 'Test_Sequence.mid')