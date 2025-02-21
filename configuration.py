from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,                # Learning rate
        "seq_length": 1024,          # Sequence length
        "d_model": 512,              # Dimension of the model (hidden size)
        "n_heads": 8,                # Number of attention heads
        "num_layers": 6,             # Number of transformer layers
        "dropout": 0.1,              # Dropout rate
        "dataset_path": "Chopin_Tokens",  # Path to the tokenized Chopin pieces
        "tokenizer_file": "midi_vocab.txt",  # Path to MIDI token vocabulary
        "model_folder": "weights",   # Folder to save the model weights
        "model_basename": "musicgen_",  # Base name for saved model files
        "preload": "latest",         # Option to preload the latest weights
        "experiment_name": "runs/musicgen", # Folder for experiment tracking
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['dataset_path']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['dataset_path']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
