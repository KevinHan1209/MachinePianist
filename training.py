import torch
import torch.nn as nn
import torch.optim as optim
from transformer import MusicTransformer
from build_vocabulary import VOCABULARY_SIZE
from pre_processing import MusicDataset, split_dataset, create_dataloader, TOKEN_TO_INDEX, ENCODED_SEQUENCES
from tqdm import tqdm

# Configuration parameters
vocab_size = VOCABULARY_SIZE  
embed_size = 256
num_heads = 8
num_layers = 6
hidden_dim = 512
seq_len = 1024
batch_size = 16
epochs = 10
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming ENCODED_SEQUENCES is already processed as per your previous code
# Split dataset into train and validation sets
train_dataset, val_dataset = split_dataset(ENCODED_SEQUENCES)

# Create DataLoaders for both training and validation sets
train_dataloader = create_dataloader(train_dataset, batch_size, shuffle=True)
val_dataloader = create_dataloader(val_dataset, batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
model = MusicTransformer(vocab_size=vocab_size, embed_size=embed_size, num_heads=num_heads, 
                         num_layers=num_layers, hidden_dim=hidden_dim, seq_len=seq_len)
model.to(device)  # Move the model to the device (GPU if available)

loss_fn = nn.CrossEntropyLoss(ignore_index=TOKEN_TO_INDEX['PAD'])  # For token prediction
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training and validation loop
def train(model, train_loader, val_loader, optimizer, loss_fn, device, epochs):
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        total_loss = 0
        for src, tgt in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            src, tgt = src.to(device), tgt.to(device)  # Move data to the device
            
            optimizer.zero_grad()  # Zero the gradients
            
            # Forward pass
            output = model(src)
            
            # Compute the loss
            loss = loss_fn(output.view(-1, vocab_size), tgt.view(-1))  # Flatten and compute loss
            total_loss += loss.item()
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {total_loss / len(train_loader)}")
        
        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss = 0
        with torch.no_grad():  # No gradients needed for validation
            for src, tgt in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}/{epochs}"):
                src, tgt = src.to(device), tgt.to(device)  # Move data to device
                
                # Forward pass
                output = model(src)
                
                # Compute the loss
                loss = loss_fn(output.view(-1, vocab_size), tgt.view(-1))
                val_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Validation Loss: {val_loss / len(val_loader)}")

# Start the training process
train(model, train_dataloader, val_dataloader, optimizer, loss_fn, device, epochs)