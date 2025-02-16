import os
import re
from collections import defaultdict

tokens_dir = "/Users/kevinhan/Desktop/Independent Study/Music Analysis/Chopin_Tokens"

note_pitches = set()
velocities = set()
time_shifts = set()
special_tokens = {"Sustain_On", "Sustain_Off"}

# Regular expressions to extract numeric values
note_pattern = re.compile(r"Note_On_(\d+)_Vel_(\d+)")
time_shift_pattern = re.compile(r"Time_Shift_(\d+)")

for filename in os.listdir(tokens_dir):
    if filename.endswith("_tokens.txt"):
        with open(os.path.join(tokens_dir, filename), 'r') as f:
            for line in f:
                line = line.strip()
                if match := note_pattern.match(line):
                    pitch, velocity = map(int, match.groups())
                    note_pitches.add(pitch)
                    velocities.add(velocity)
                elif match := time_shift_pattern.match(line):
                    time_shifts.add(int(match.group(1)))
                elif line in special_tokens:
                    pass  # Sustain events already accounted for

# Build Vocabulary
vocab = {
    f"Note_On_{p}_Vel_{v}": i for i, (p, v) in enumerate(
        (p, v) for p in sorted(note_pitches) for v in sorted(velocities)
    )
}
vocab.update({f"Time_Shift_{t}": len(vocab) + i for i, t in enumerate(sorted(time_shifts))})
vocab.update({token: len(vocab) + i for i, token in enumerate(special_tokens)})

# Add PAD token (for padding sequences)
vocab["PAD"] = len(vocab)

# Save vocabulary
with open("midi_vocab.txt", "w") as f:
    for token, index in vocab.items():
        f.write(f"{token},{index}\n")

print("Vocabulary size:", len(vocab))
