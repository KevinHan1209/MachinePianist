import torch
from transformer import MusicTransformer
from build_vocabulary import VOCABULARY_SIZE
from pre_processing import TOKEN_TO_INDEX, INDEX_TO_TOKEN
from midi_tokenizer import tokens_to_midi

import config

# Configuration parameters (same as training)
seq_len = config.SEQ_LEN  # Sequence length of the model
vocab_size = config.VOCAB_SIZE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_size = config.EMBED_SIZE
num_heads = config.NUM_HEADS
num_layers  = config.NUM_LAYERS
hidden_dim = config.HIDDEN_DIM

# Generation parameters
max_length = 1024 * 4 # Length of new generation
TEMP = 0.6 # "randomness" in generating new tokens. Probably should set smaller values for longer maximum lengths? 

# Load your trained model
model = MusicTransformer(vocab_size=vocab_size, embed_size=embed_size, num_heads=num_heads, 
                         num_layers=num_layers, hidden_dim=hidden_dim, seq_len=seq_len-1)
checkpoint = torch.load('checkpoints/music_transformer.pth', map_location=device)
model_state_dict = checkpoint["model_state_dict"]

# Load model weights into the model
model.load_state_dict(model_state_dict)
model.to(device)
model.eval()  # Set to evaluation mode

def generate(model, start_sequence, max_len, device, temperature=1.0):
    model.eval()  # Set model to evaluation mode
    generated = start_sequence.clone().to(device)

    for _ in range(max_len):
        with torch.no_grad():
            output = model(generated)  # Forward pass with the current sequence
            next_token = output[:, -1, :]  # Take the last token's output

            # Apply temperature scaling or sampling
            next_token = sample_with_temperature(next_token, temperature)

            # Append the generated token to the sequence
            generated = torch.cat((generated, next_token.unsqueeze(1)), dim=1)

    return generated

def sample_with_temperature(logits, temperature=TEMP):
    logits = logits / temperature  # Scale logits by temperature
    probabilities = torch.nn.functional.softmax(logits, dim=-1)  # Convert to probabilities
    return torch.multinomial(probabilities, 1)  # Sample from the distribution


def read_tokens_from_file(file_path, n=None):
    tokens = []
    with open(file_path, 'r') as f:
        for line in f:
            token = line.strip()  # Remove any leading/trailing whitespace
            if token in TOKEN_TO_INDEX:
                tokens.append(TOKEN_TO_INDEX[token])
            else:
                print(f"Warning: Token '{token}' not found in TOKEN_TO_INDEX.")
    
    # If n is not specified, use the entire file length
    if n is None or n > len(tokens):
        n = len(tokens)

    tokens = tokens[-n:]  # Keep only the last n tokens

    return torch.tensor(tokens).unsqueeze(0).to(device)  # Add batch dimension

file_path = 'Chopin_Tokens/Chopin, Frédéric, Nocturnes, Op.48, -7mntyrW3HU.mid_tokens.txt'

# Read the start sequence from the file
start_sequence = read_tokens_from_file(file_path, n = seq_len-1)

# Generate a new sequence
generated_sequence = generate(model, start_sequence, max_len=max_length, device = device)

# Convert the generated sequence from token indices back to tokens
generated_tokens = [INDEX_TO_TOKEN[token.item()] for token in generated_sequence.squeeze(0)]
print(len(generated_tokens))
tokens_to_midi(generated_tokens, 'Test_Sequence.mid')