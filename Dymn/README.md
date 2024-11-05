# Quick Start for Inference with EfficientAT
# This script runs inference on an audio file using a specified pre-trained model.

# Basic usage example:
python inference_2.py --cuda --model_name=dymn20_as --audio_path="/path/to/audiofile.wav" --output_dir="/path/to/output_directory/"

# Options:
# --cuda            : Enables GPU for faster inference (optional).
# --model_name      : Specifies the pre-trained model to use (e.g., "dymn20_as").
# --audio_path      : Path to the audio file for analysis.
# --output_dir      : Directory to save output results.

# Full example command:
python inference_2.py --cuda --model_name=dymn20_as --audio_path="/home/lsh/share/EfficientAT/datasets/esc50/audio_32k/1-137-A-32.wav" --output_dir="/home/lsh/share/EfficientAT/result_image/"
