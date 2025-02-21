import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split


# Define the MusicTransformer model
class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, hidden_dim, seq_len):
        super(MusicTransformer, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)  # Token embedding
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, embed_size))  # Learnable positional encoding
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim,
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)  # Output layer to predict next token

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding  # Add positional encoding to embeddings
        x = x.permute(1, 0, 2)  # Transpose to (seq_len, batch_size, embed_size)
        
        transformer_out = self.transformer(x, x)  # Apply transformer
        output = self.fc_out(transformer_out)  # Get the predicted token indices
        
        return output

