#!/bin/bash

# Generate sentences.
python3 holistic_bias/generate_sentences.py

# Generate outputs for the sentences.
python3 generate_outputs.py

# Calculate the input-output distances.
python3 calculate_distances.py

# Create plots.
python3 create_plots.py

