import torch
from transformer import MusicTransformer
from build_vocabulary import VOCABULARY_SIZE
from pre_processing import TOKEN_TO_INDEX, INDEX_TO_TOKEN, read_tokens_from_file
from midi_tokenizer import tokens_to_midi
import torch.nn.functional as F
from tqdm import tqdm

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
                         num_layers=num_layers, hidden_dim=hidden_dim, seq_len=seq_len)
checkpoint = torch.load('checkpoints/music_transformer.pth', map_location=device)
file_path = 'Chopin_Tokens/Chopin, Frédéric, Nocturnes, Op.48, -7mntyrW3HU.mid_tokens.txt'
# Read the start sequence from the file
start_sequence = read_tokens_from_file(file_path, n = seq_len)


# Load model weights into the model
model.load_state_dict(checkpoint)
model.to(device)
model.eval()  # Set to evaluation mode
print(f"len: {len(start_sequence)}")

def generate_music(model, start_sequence, max_length, temp=0.6):
    model.eval()  # Set model to evaluation mode
    generated_sequence = torch.tensor(start_sequence, dtype=torch.long, device=device).unsqueeze(0)  # Shape (1, seq_len-1)

    for _ in tqdm(range(max_length), desc="Generating Music", unit="token"):
        with torch.no_grad():
            logits = model(generated_sequence[:, -seq_len:])  # Ensure input is always seq_len
            logits = logits[:, -1, :]  # Get the last token logits
        
        # Apply temperature scaling and softmax
        scaled_logits = logits / temp
        probs = F.softmax(scaled_logits, dim=-1)

        # Sample the next token
        next_token = torch.multinomial(probs, num_samples=1).item()
        
        # Append the new token to the sequence
        generated_sequence = torch.cat([generated_sequence, torch.tensor([[next_token]], device=device)], dim=1)

    return generated_sequence.squeeze(0).tolist()

# Generate new music
generated_tokens = [INDEX_TO_TOKEN[token_index] for token_index in generate_music(model, start_sequence, max_length, TEMP)]

# Convert generated tokens back to MIDI
output_midi = tokens_to_midi(generated_tokens, "generated_music.mid")
output_midi.save("generated_music.mid")

print("Music generation complete! Saved as 'generated_music.mid'.")