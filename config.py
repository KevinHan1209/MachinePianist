# config.py
from build_vocabulary import VOCABULARY_SIZE
import torch
import os

# Model parameters
VOCAB_SIZE = VOCABULARY_SIZE  # Adjust as needed
EMBED_SIZE = 256
NUM_HEADS = 8
NUM_LAYERS = 6
HIDDEN_DIM = 512
SEQ_LEN = 513

# Training parameters
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-4

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths for saving/loading model weights
MODEL_DIR = "checkpoints"
MODEL_PATH = os.path.join(MODEL_DIR, "music_transformer.pth")

# Ensure the checkpoint directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Weights & Biases (wandb) settings
WANDB_PROJECT = "Machine_Pianist"
WANDB_ENTITY = "kevinhan"  # Replace with your wandb username or team
