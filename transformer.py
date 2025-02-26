import torch
import torch.nn as nn
import math

class MusicTransformer(nn.Module):
    '''
    Absolute positional encoding instead of learned
    Implements dropout
    '''
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, hidden_dim, seq_len, dropout=0.1):
        super(MusicTransformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)  # Token embedding
        self.positional_encoding = self.create_positional_encoding(seq_len, embed_size)  # Fixed encoding
        self.dropout = nn.Dropout(dropout)  # Dropout layer

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,  # Apply dropout in the transformer layer
            batch_first=True,  # Ensures (batch, seq_len, embed_size) input format
        )
        
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)  # Output layer to predict next token

    def create_positional_encoding(self, seq_len, embed_size):
        """Creates sinusoidal positional encoding."""
        pe = torch.zeros(seq_len, embed_size)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))

        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices

        return pe.unsqueeze(0)  # Shape: (1, seq_len, embed_size)

    def generate_causal_mask(self, size, device):
        """Creates a causal mask for auto-regressive decoding."""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)  # Upper triangular matrix
        return mask.masked_fill(mask == 1, float('-inf'))  # Convert 1s to -inf for masking

    def forward(self, x):
        batch_size, seq_len = x.shape
        x = self.embedding(x) + self.positional_encoding[:, :seq_len, :].to(x.device)  # Add fixed positional encoding
        x = self.dropout(x)  # Apply dropout after embeddings and positional encoding
        mask = self.generate_causal_mask(seq_len, x.device)  # Generate causal mask
        
        x = self.decoder(x, memory=None, tgt_mask=mask)  # Apply transformer decoder with mask
        x = self.fc_out(x)  # Predict next token
        return x

