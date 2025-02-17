import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

class InputEmbeddings(nn.Module):
    '''
    x: Tensor containing token indices (context)
    d_model: dimensionality of embeddings (size of each embedding vector)
    vocab_size: self-explanatory
    '''
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    '''
    pe: A zero tensor of shape (seq, d_model), which will hold the positional encodings.
    position: A tensor containing the position indices from 0 to seq-1, reshaped to (seq, 1).
    div_term: A tensor used for scaling the positions, calculated as 10000 raised to the power of (2i / d_model) where i is the index of the embedding dimension.
    '''
    def __init__(self, d_model: int, seq: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq = seq
        self.dropout = nn.Dropout(dropout)
        
        # Create a matrix of shape (seq, d_model)
        pe = torch.zeros(seq, d_model)
        
        # Create a vector of shape (seq)
        position = torch.arange(0, seq, dtype=torch.float).unsqueeze(1) # (seq, 1)
        
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq, d_model)
        
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq, d_model)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    '''
    self.eps: Stores the epsilon value to prevent numerical instability during division.
    self.alpha: A learnable scale parameter initialized to ones, of shape (features,). It scales the normalized output.
    self.bias: A learnable shift parameter initialized to zeros, of shape (features,). It shifts the normalized output.
    '''
    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha (multiplicative) is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias (additive) is a learnable parameter

    def forward(self, x):
        # x: (batch, seq, hidden_size)
        # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    

class FeedForwardBlock(nn.Module):
    '''
    self.linear_1: A linear layer (fully connected layer) that maps from d_model to d_ff. This is the first linear transformation applied to the input.
    self.linear_2: A linear layer that maps from d_ff back to d_model. This is the second linear transformation applied to the intermediate representation.
    '''
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq, d_model) --> (batch, seq, d_ff) --> (batch, seq, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttentionBlock(nn.Module):
    '''
    Attributes:
    self.d_model: Stores the dimension of the embedding vectors.
    self.h: Stores the number of attention heads.
    self.d_k: The dimension of the vectors processed by each head, calculated as d_model // h. This ensures the embedding dimension is evenly split across the heads.
    self.w_q, self.w_k, self.w_v, self.w_o: Linear layers that project the input vectors to queries, keys, values, and outputs, respectively.

    Attention Calculation (attention static method) Parameters:
    query, key, value: The input tensors for the attention mechanism.
    mask: An optional mask tensor to prevent attention to certain positions.

    Steps:
    Compute Attention Scores: The attention scores are computed using the dot product of the query and key tensors, scaled by the square root of d_k to maintain stable gradients. This is done as (query @ key.transpose(-2, -1)) / math.sqrt(d_k).
    Apply Mask: If a mask is provided, positions where the mask is zero are set to a very low value (effectively -inf) to ensure they don’t affect the attention calculation.
    Compute Attention Output: The attention output is obtained by multiplying the normalized attention scores with the value tensor.
    Return: The method returns the attention output and the attention scores for possible visualization.

    Reshape and Split Heads: The projected tensors are reshaped and transposed to split the embedding dimension into multiple heads. The shape changes from (batch, seq, d_model) to (batch, seq, h, d_k) and then transposed to (batch, h, seq, d_k).

    Calculate Attention: The reshaped queries, keys, and values are passed to the attention method to compute the attention output and scores.

    Combine Heads: The attention outputs from all heads are concatenated and reshaped back to the original embedding dimension. The shape changes from (batch, h, seq, d_k) to (batch, seq, h, d_k) and finally to (batch, seq, d_model).

    Final Linear Transformation: The combined output is projected back to the original embedding dimension using the w_o linear layer.
    '''
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq, d_k) --> (batch, h, seq, seq)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq, seq) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq, seq) --> (batch, h, seq, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq, d_model) --> (batch, seq, d_model)
        key = self.w_k(k) # (batch, seq, d_model) --> (batch, seq, d_model)
        value = self.w_v(v) # (batch, seq, d_model) --> (batch, seq, d_model)

        # (batch, seq, d_model) --> (batch, seq, h, d_k) --> (batch, h, seq, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq, d_k) --> (batch, seq, h, d_k) --> (batch, seq, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq, d_model) --> (batch, seq, d_model)  
        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    '''
    Forward Pass (forward method) Parameters:
    sublayer: A callable sub-layer (e.g., multi-head attention or feedforward network) to be applied to the normalized input.

    Steps:
    Apply Sublayer: The normalized input is then passed to the sub-layer (e.g., multi-head attention or feedforward network) specified by sublayer. This call to sublayer(self.norm(x)) applies the sub-layer’s transformation to the normalized input.
    Add Residual Connection: Finally, the original input tensor x is added to the output of the dropout layer. This addition operation implements the residual connection, which allows the network to learn identity mappings and mitigates the vanishing gradient problem in deep networks. The resulting tensor is the output of the residual connection block.
    Return: The method returns the result of the residual connection, which is the sum of the original input and the transformed (normalized, sub-layered, and dropped out) input.
    '''
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    '''
    Attributes:
    self.self_attention_block: Stores the multi-head self-attention block.
    self.feed_forward_block: Stores the feed-forward network block.
    self.residual_connections: A list of two ResidualConnection instances, one for the self-attention sub-layer and one for the feed-forward sub-layer.

    Feed Forward Steps:
    Self-Attention with Residual Connection: The input tensor x is first normalized and passed through the self-attention block (self.self_attention_block), with a residual connection around it.
    This is done using the first ResidualConnection in the list: self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)).
    The lambda function is used to pass the normalized input to the self-attention block.

    Feed-Forward with Residual Connection: The output from the self-attention sub-layer is then normalized and passed through the feed-forward block (self.feed_forward_block), with another residual connection around it. This is done using the second ResidualConnection in the list: self.residual_connections[1](x, self.feed_forward_block).

    Return: The method returns the output tensor after passing through both the self-attention and feed-forward sub-layers with residual connections and normalization.
    '''
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    '''
    Feed Forward Steps:
    Pass Through Encoder Layers: The input tensor x is sequentially passed through each layer in the self.layers list. Each layer processes the input tensor and updates it. This is done using a loop:
    for layer in self.layers: x = layer(x, mask).

    Final Layer Normalization: The output tensor from the last encoder layer is normalized using self.norm(x).

    Return: The method returns the normalized output tensor after processing through the stack of encoder layers.
    '''
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    '''
    Parameters:
    self_attention_block: An instance of the MultiHeadAttentionBlock class for self-attention within the target sequence.
    cross_attention_block: An instance of the MultiHeadAttentionBlock class for attention over the encoder’s output (cross-attention).
    feed_forward_block: An instance of a feed-forward network (e.g., FeedForwardBlock).

    Attributes:
    self.self_attention_block: Stores the self-attention block.
    self.cross_attention_block: Stores the cross-attention block.
    self.feed_forward_block: Stores the feed-forward network block.
    self.residual_connections: A list of three ResidualConnection instances, one for each sub-layer (self-attention, cross-attention, and feed-forward).

    Feed Forward Parameters:
    src_mask: A mask tensor for the source sequence to prevent attention to certain positions.
    tgt_mask: A mask tensor for the target sequence to prevent attention to future positions (during training) and certain positions.

    Steps:
    Cross-Attention with Residual Connection: The output from the self-attention sub-layer is then normalized and passed through the cross-attention block (self.cross_attention_block), with another residual connection around it. This is done using the second ResidualConnection in the list:
    self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)).
    The cross-attention mechanism allows each position in the target sequence to attend to all positions in the source sequence (encoder output), providing the necessary context for generating the target sequence.

    Return: The method returns the output tensor after processing through the self-attention, cross-attention, and feed-forward sub-layers with residual connections and normalization.


    '''
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    '''
    Forward Pass:
    The forward method processes the input tensor x through each DecoderBlock in the self.layers list.
    Each DecoderBlock layer performs self-attention, cross-attention, and feed-forward operations, with residual connections and layer normalization applied to each sub-layer.
    The src_mask ensures that the attention mechanism only attends to valid positions in the source sequence.
    The tgt_mask prevents the decoder from attending to future positions in the target sequence during training, maintaining the autoregressive nature of the model.
    After processing through all the layers, the final output tensor is normalized using the LayerNormalization instance (self.norm).
    '''
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    '''
    Predicts next token in sequence.

    Attributes:
    self.proj: An instance of nn.Linear, a fully connected linear layer that performs the projection from the model’s output dimension (d_model) to the vocabulary size (vocab_size).

    Forward Pass:
    The forward method takes the input tensor x from the decoder, which has a shape of (batch, seq, d_model).
    The input tensor is passed through the linear layer self.proj.
    The linear layer computes the dot product between the input tensor and its weight matrix, and adds the bias (if applicable), transforming the input tensor from the model’s feature space to the vocabulary space.
    The output tensor now has a shape of (batch, seq, vocab_size), where each position in the sequence is associated with a vector of logits for each word in the vocabulary.
    '''
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq, d_model) --> (batch, seq, vocab_size)
        return self.proj(x)
    
