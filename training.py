import torch
import torch.nn as nn
import torch.optim as optim
import os
import wandb
from tqdm import tqdm

from transformer import MusicTransformer
from build_vocabulary import VOCABULARY_SIZE
from pre_processing import TOKEN_TO_INDEX, train_loader, val_loader
import config  # Import configuration

# Initialize wandb
wandb.init(
    project=config.WANDB_PROJECT,
    config={
        "vocab_size": config.VOCAB_SIZE,
        "embed_size": config.EMBED_SIZE,
        "num_heads": config.NUM_HEADS,
        "num_layers": config.NUM_LAYERS,
        "hidden_dim": config.HIDDEN_DIM,
        "seq_len": config.SEQ_LEN,
        "batch_size": config.BATCH_SIZE,
        "epochs": config.EPOCHS,
        "learning_rate": config.LEARNING_RATE,
        "device": str(config.DEVICE),
    }
)

# Use configuration values
vocab_size = config.VOCAB_SIZE  
embed_size = config.EMBED_SIZE
num_heads = config.NUM_HEADS
num_layers = config.NUM_LAYERS
hidden_dim = config.HIDDEN_DIM
seq_len = config.SEQ_LEN
batch_size = config.BATCH_SIZE
epochs = config.EPOCHS
learning_rate = config.LEARNING_RATE
device = config.DEVICE

# Initialize the model
model = MusicTransformer(
    vocab_size=vocab_size,
    embed_size=embed_size,
    num_heads=num_heads,
    num_layers=num_layers,
    hidden_dim=hidden_dim,
    seq_len=seq_len,
    dropout=0.1
).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=TOKEN_TO_INDEX['PAD'])  # Ignore padding token in loss calculation
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()  # Clear gradients
        outputs = model(inputs)  # Forward pass
        
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        
        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:  # Print loss every 10 batches
            print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}] Completed. Avg Training Loss: {avg_train_loss:.4f}")

    # Validation loop
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss after Epoch {epoch+1}: {avg_val_loss:.4f}")

# Save the final model
torch.save(model.state_dict(), "music_transformer.pth")
print("Training complete. Model saved.")