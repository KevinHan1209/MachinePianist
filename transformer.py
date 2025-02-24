import torch
import torch.nn as nn

class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, hidden_dim, seq_len):
        super(MusicTransformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)  # Token embedding
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, embed_size))  # Learnable positional encoding
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,  # Ensures (batch, seq_len, embed_size) input format
        )
        
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)  # Output layer to predict next token

    def generate_causal_mask(self, size, device):
        """ Creates a causal mask for auto-regressive decoding """
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)  # Upper triangular matrix
        return mask.masked_fill(mask == 1, float('-inf'))  # Convert 1s to -inf for masking

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]  # Add positional encoding
        mask = self.generate_causal_mask(x.size(1), x.device)  # Generate causal mask
        
        x = self.decoder(x, x, tgt_mask=mask)  # Apply transformer decoder with mask
        return self.fc_out(x)  # Predict next token
