from mido import MidiFile, MidiTrack, Message, MetaMessage
import os

# Function to convert MIDI to tokens
def midi_to_tokens(midi_path):
    midi = MidiFile(midi_path)
    tokens = []
    
    #ticks_per_beat = midi.ticks_per_beat  # Get ticks_per_beat from the MIDI file
    #tokens.append(f'Ticks_Per_Beat_{ticks_per_beat}')  # Store this as a token

    for track in midi.tracks:
        time_counter = 0  # Track accumulated time in ticks
        
        for msg in track:
            time_counter += msg.time  # Accumulate time
            
            if msg.type == 'note_on':
                if time_counter > 0:
                    tokens.append(f'Time_Shift_{time_counter}')
                    time_counter = 0
                tokens.append(f'Note_On_{msg.note}_Vel_{msg.velocity}')
            elif msg.type == 'note_off':
                if time_counter > 0:
                    tokens.append(f'Time_Shift_{time_counter}')
                    time_counter = 0
                tokens.append(f'Note_Off_{msg.note}')
            elif msg.type == 'control_change' and msg.control == 64:
                if time_counter > 0:
                    tokens.append(f'Time_Shift_{time_counter}')
                    time_counter = 0
                if msg.value > 0:
                    tokens.append(f'Sustain_On')
                else:
                    tokens.append(f'Sustain_Off')
    
    return tokens


# Function to convert tokens back to MIDI
def tokens_to_midi(tokens, output_path):
    midi = MidiFile()

    # Create two tracks: one for meta data and one for the notes
    meta_track = MidiTrack()
    note_track = MidiTrack()
    
    # Append both tracks to the MIDI file
    midi.tracks.append(meta_track)
    midi.tracks.append(note_track)
    
    time_counter = 0
    tempo = 500000  # Default tempo (500000 microseconds per beat)
    time_signature_set = False  # Ensure time signature is only set once
    ticks_per_beat = 480  # Default value, in case no token is found
    
    for token in tokens:
        if token.startswith('Ticks_Per_Beat_'):
            ticks_per_beat = int(token.split('_')[3])  # Extract ticks_per_beat from the token
            midi.ticks_per_beat = ticks_per_beat  # Set the ticks_per_beat value

        # Handle tempo changes
        elif token.startswith('Tempo_'):
            tempo = int(token.split('_')[1])
            meta_track.append(MetaMessage('set_tempo', tempo=tempo, time=time_counter))
            time_counter = 0  # Reset time after tempo change

        # Handle time shifts
        elif token.startswith('Time_Shift_'):
            time_counter += int(token.split('_')[2])

        # Handle Note_On events
        elif token.startswith('Note_On_'):
            parts = token.split('_')
            note = int(parts[2])
            velocity = int(parts[4])
            note_track.append(Message('note_on', note=note, velocity=velocity, time=time_counter))
            time_counter = 0  # Reset time after note on

        # Handle Note_Off events
        elif token.startswith('Note_Off_'):
            note = int(token.split('_')[2])
            note_track.append(Message('note_off', note=note, velocity=0, time=time_counter))
            time_counter = 0  # Reset time after note off

        # Handle sustain on
        elif token == 'Sustain_On':
            note_track.append(Message('control_change', control=64, value=127, time=time_counter))
            time_counter = 0

        # Handle sustain off
        elif token == 'Sustain_Off':
            note_track.append(Message('control_change', control=64, value=0, time=time_counter))
            time_counter = 0
        
        # Set time signature once (if not already set)
        if not time_signature_set:
            meta_track.append(MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
            time_signature_set = True  # Only set once

    # Add end of track message to the meta track
    meta_track.append(MetaMessage('end_of_track', time=1))

    # Save the reconstructed MIDI file
    midi.save(output_path)
    print(f"✅ Saved MIDI to {output_path}")


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

# Directory paths
midi_dir = "Music Analysis/midis"
tokens_dir = "Music Analysis/Chopin_Tokens"

# Create the tokens directory if it doesn't exist
if not os.path.exists(tokens_dir):
    os.makedirs(tokens_dir)

if __name__ == '__main__':
    convert_chopin_midis_to_tokens(midi_dir, tokens_dir)


