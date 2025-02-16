import os
from midi_tokenizer import midi_to_tokens

# Directory paths
midi_dir = "/Users/kevinhan/Desktop/Independent Study/Music Analysis/midis"
tokens_dir = "/Users/kevinhan/Desktop/Independent Study/Music Analysis/Chopin_Tokens"

# Create the tokens directory if it doesn't exist
if not os.path.exists(tokens_dir):
    os.makedirs(tokens_dir)

# Function to convert all Chopin MIDI files to tokens and save them to separate files
def convert_chopin_midis_to_tokens(midi_dir, tokens_dir):
    # Iterate over all files in the directory
    for filename in os.listdir(midi_dir):
        # Check if the file is a MIDI file and starts with 'Chopin'
        if filename.endswith(".mid") and filename.startswith("Chopin"):
            midi_path = os.path.join(midi_dir, filename)
            print(f"Processing: {filename}")
            
            # Convert the MIDI file to tokens
            tokens = midi_to_tokens(midi_path)  # Assuming midi_to_tokens is already defined
            
            # Generate output file path
            output_file = os.path.join(tokens_dir, f"{filename}_tokens.txt")
            
            # Save tokens to the file
            with open(output_file, 'w') as f:
                for token in tokens:
                    f.write(token + "\n")
            
            print(f"Tokens for {filename} saved to {output_file}")

# Call the function
convert_chopin_midis_to_tokens(midi_dir, tokens_dir)
