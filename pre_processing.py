import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import config
import os

# Function to load the midi vocabulary from the file and create a token-to-index mapping
def load_vocab(vocab_file):
    token_to_index = {}
    with open(vocab_file, "r") as file:
        for idx, line in enumerate(file.readlines()):
            token = line.strip()
            token_to_index[token] = idx  # Assign an index to each token
    return token_to_index

def create_index_to_token(token_to_index):
    index_to_token = {index: token for token, index in token_to_index.items()}
    return index_to_token

# Function to encode the tokens into indices using the token-to-index mapping
def encode_tokens(tokens, token_to_index):
    return [token_to_index.get(token.strip(), token_to_index['PAD']) for token in tokens]  # 'PAD' if token is unknown

def process_pieces(directory, vocab_file, sequence_length):
    token_to_index = load_vocab(vocab_file)  # Load the vocabulary from midi_vocab.txt
    
    all_sequences = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):  # Ensure we're reading only text files
            file_path = os.path.join(directory, filename)
            
            with open(file_path, "r") as file:
                tokens = file.readlines()  # Read all tokens from the file
                
                token_count = len(tokens)
                
                # Split the tokens into sequences of 'sequence_length' tokens
                for i in range(0, token_count, sequence_length):
                    seq = tokens[i:i+sequence_length]
                    
                    # If the sequence is shorter than 'sequence_length', pad it
                    if len(seq) < sequence_length:
                        seq = seq + ["PAD"] * (sequence_length - len(seq))  # Pad with 'PAD' token
                    
                    # Encode the tokens into indices
                    seq_indices = encode_tokens(seq, token_to_index)
                    all_sequences.append(seq_indices)
    
    return all_sequences

# Define parameters
directory = "Chopin_Tokens"  # Directory containing tokenized pieces
vocab_file = "midi_vocab.txt"  # Path to the midi vocabulary file
sequence_length = config.SEQ_LEN  # Length of each sequence (can be adjusted)

# Process the pieces and encode them
TOKEN_TO_INDEX = load_vocab(vocab_file)
INDEX_TO_TOKEN = create_index_to_token(TOKEN_TO_INDEX)
ENCODED_SEQUENCES = process_pieces(directory, vocab_file, sequence_length)

print("Total sequences:",len(ENCODED_SEQUENCES))

class MIDIDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences  # Encoded MIDI token sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long) 
        return sequence[:-1], sequence[1:]  


train_size = int(0.8 * len(ENCODED_SEQUENCES)) 
val_size = len(ENCODED_SEQUENCES) - train_size

train_data, val_data = random_split(ENCODED_SEQUENCES, [train_size, val_size])

train_loader = DataLoader(MIDIDataset(train_data), batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(MIDIDataset(val_data), batch_size=config.BATCH_SIZE, shuffle=False)

print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")