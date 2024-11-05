import argparse
import torch
import librosa
import numpy as np
from torch import autocast
from contextlib import nullcontext
import matplotlib.pyplot as plt
import os
from datetime import datetime

from models.mn.model import get_model as get_mobilenet
from models.dymn.model import get_model as get_dymn
from models.ensemble import get_ensemble_model
from models.preprocess import AugmentMelSTFT
from helpers.utils import NAME_TO_WIDTH, labels

def audio_tagging(args):
    """
    Running Inference on an audio clip and saving the mel spectrogram as an image.
    """
    model_name = args.model_name
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    audio_path = args.audio_path
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    n_mels = args.n_mels

    # Load pre-trained model
    if len(args.ensemble) > 0:
        model = get_ensemble_model(args.ensemble)
    else:
        if model_name.startswith("dymn"):
            model = get_dymn(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name,
                                  strides=args.strides)
        else:
            model = get_mobilenet(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name,
                                  strides=args.strides, head_type=args.head_type)
    model.to(device)
    model.eval()

    # Model to preprocess waveform into mel spectrograms
    mel = AugmentMelSTFT(n_mels=n_mels, sr=sample_rate, win_length=window_size, hopsize=hop_size)
    mel.to(device)
    mel.eval()

    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    waveform = torch.from_numpy(waveform[None, :]).to(device)

    # Generate mel spectrogram
    with torch.no_grad(), autocast(device_type=device.type) if args.cuda else nullcontext():
        spec = mel(waveform)  # Mel spectrogram 생성
        preds, features = model(spec.unsqueeze(0))
    preds = torch.sigmoid(preds.float()).squeeze().cpu().numpy()

    # Save the spectrogram as an image with timestamp and filename
    spec = spec.squeeze().cpu().numpy()  # (n_mels, time) 형태로 변환
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_filename = os.path.basename(audio_path).split('.')[0]
    save_path = os.path.join(args.output_dir, f"{timestamp}_{audio_filename}.png")

    plt.figure(figsize=(10, 4))
    plt.imshow(spec, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Mel Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path)  # 이미지 저장
    plt.close()
    print(f"Mel spectrogram saved at {save_path}")

    sorted_indexes = np.argsort(preds)[::-1]

    # Print audio tagging top probabilities
    print("************* Acoustic Event Detected: *****************")
    for k in range(10):
        print('{}: {:.3f}'.format(labels[sorted_indexes[k]], preds[sorted_indexes[k]]))
    print("********************************************************")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    # model name decides, which pre-trained model is loaded
    parser.add_argument('--model_name', type=str, default='dymn20_as')
    parser.add_argument('--strides', nargs=4, default=[2, 2, 2, 2], type=int)
    parser.add_argument('--head_type', type=str, default="mlp")
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--audio_path', type=str, required=True)

    # preprocessing
    parser.add_argument('--sample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=800)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--n_mels', type=int, default=128)

    # output directory for saving images
    parser.add_argument('--output_dir', type=str, default='/home/lsh/share/EfficientAT/result_image/')

    # overwrite 'model_name' by 'ensemble_model' to evaluate an ensemble
    parser.add_argument('--ensemble', nargs='+', default=[])

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    audio_tagging(args)
