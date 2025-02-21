import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split

import os

# Function to load the midi vocabulary from the file and create a token-to-index mapping
def load_vocab(vocab_file):
    token_to_index = {}
    with open(vocab_file, "r") as file:
        for idx, line in enumerate(file.readlines()):
            token = line.strip()
            token_to_index[token] = idx  # Assign an index to each token
    return token_to_index

# Function to encode the tokens into indices using the token-to-index mapping
def encode_tokens(tokens, token_to_index):
    return [token_to_index.get(token.strip(), token_to_index['PAD']) for token in tokens]  # 'PAD' if token is unknown

# Example of how to use this script with your tokenized Chopin pieces
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
sequence_length = 1024  # Length of each sequence (can be adjusted)

# Process the pieces and encode them
TOKEN_TO_INDEX = load_vocab(vocab_file)
ENCODED_SEQUENCES = process_pieces(directory, vocab_file, sequence_length)

print("Total sequences:",len(ENCODED_SEQUENCES))

class MusicDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Create the src (input) and tgt (target) sequences
        src = torch.tensor(seq)  # Use the entire padded sequence
        tgt = torch.tensor(seq[1:])  # All except the first token for the target

        return src, tgt

# Create DataLoader
def create_dataloader(encoded_sequences, batch_size, shuffle=True):
    dataset = MusicDataset(encoded_sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def split_dataset(encoded_sequences, val_split=0.2):
    val_size = int(len(encoded_sequences) * val_split)
    train_size = len(encoded_sequences) - val_size
    train_dataset, val_dataset = random_split(encoded_sequences, [train_size, val_size])
    return train_dataset, val_dataset
