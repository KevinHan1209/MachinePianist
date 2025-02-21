import torch
import torch.nn as nn
import torch.optim as optim
import os
import wandb
from tqdm import tqdm

from transformer import MusicTransformer
from build_vocabulary import VOCABULARY_SIZE
from pre_processing import MusicDataset, split_dataset, create_dataloader, TOKEN_TO_INDEX, ENCODED_SEQUENCES
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

# Split dataset into train and validation sets
train_dataset, val_dataset = split_dataset(ENCODED_SEQUENCES)

# Create DataLoaders
train_dataloader = create_dataloader(train_dataset, batch_size, shuffle=True)
val_dataloader = create_dataloader(val_dataset, batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
model = MusicTransformer(vocab_size=vocab_size, embed_size=embed_size, num_heads=num_heads, 
                         num_layers=num_layers, hidden_dim=hidden_dim, seq_len=seq_len-1)
model.to(device)

loss_fn = nn.CrossEntropyLoss(ignore_index=TOKEN_TO_INDEX['PAD'])  # Ignore padding tokens
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Load existing model weights if available
if os.path.exists(config.MODEL_PATH) and config.FROM_SAVED:
    print(f"Loading existing model weights from {config.MODEL_PATH}...")
    checkpoint = torch.load(config.MODEL_PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resuming training from epoch {start_epoch}.\n")
else:
    start_epoch = 0

# Watch the model in wandb (logs gradients and model parameters)
wandb.watch(model, log="all")

# Training and validation loop
def train(model, train_loader, val_loader, optimizer, loss_fn, device, epochs, start_epoch):
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        for src, tgt in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()
            output = model(src)

            loss = loss_fn(output.view(-1, vocab_size), tgt.view(-1))
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_train_loss}")

        # Log training loss to wandb
        wandb.log({"Training Loss": avg_train_loss, "Epoch": epoch + 1})

        # Save model weights after each epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, config.MODEL_PATH)
        print(f"Model saved to {config.MODEL_PATH}.")

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}/{epochs}"):
                src, tgt = src.to(device), tgt.to(device)
                output = model(src)

                loss = loss_fn(output.view(-1, vocab_size), tgt.view(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Validation Loss: {avg_val_loss}")

        # Log validation loss to wandb
        wandb.log({"Validation Loss": avg_val_loss, "Epoch": epoch + 1})

# Start training
train(model, train_dataloader, val_dataloader, optimizer, loss_fn, device, epochs, start_epoch)

# Finish wandb run
wandb.finish()
