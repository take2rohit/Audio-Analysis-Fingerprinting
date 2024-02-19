import copy
import os
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import entropy
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import pandas as pd
from audio_analysis import AudioAnalyzer


def apply_alteration_and_crop(waveform, sample_rate, alteration):
    """
    Applies specified audio alteration to the waveform and crops it to a fixed duration.

    Parameters:
        waveform (Tensor): The original audio waveform.
        sample_rate (int): The sample rate of the audio.
        alteration (str): The type of alteration to apply ('original', 'minor', 'moderate', 'strong').

    Returns:
        Tensor: The altered audio waveform.
    """
    if alteration == 'original':
        return waveform

    if alteration == 'minor':
        effects = [['bass', '+1']]
        cropped_waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, effects=effects)

    elif alteration == 'moderate':
        noise = torch.randn_like(waveform) * 0.001 * waveform.abs().max()
        waveform += noise
        effects = [['bandpass', '20', '1000']]
        cropped_waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, effects=effects)

    elif alteration == 'strong':
        effects = [['lowpass', '50']]
        cropped_waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, effects=effects)

    return cropped_waveform

def load_and_resample_audio(file_path, target_sample_rate):
    """
    Loads an audio file and resamples it to a specified sample rate.

    Parameters:
        file_path (str): Path to the audio file.
        target_sample_rate (int): The desired sample rate.

    Returns:
        Tensor: The loaded and resampled audio waveform.
        int: The sample rate of the loaded audio.
    """
    waveform, sr = torchaudio.load(file_path)
    if sr != target_sample_rate:
        waveform = T.Resample(orig_freq=sr, new_freq=target_sample_rate)(waveform)
    return waveform, target_sample_rate

def save_altered_audio(alterations, waveform, sample_rate, output_folder='data/altered_audio'):
    """
    Applies alterations to the audio and saves the results.

    Parameters:
        alterations (dict): A dictionary mapping alteration names to their type.
        waveform (Tensor): The original audio waveform.
        sample_rate (int): The sample rate of the audio.
        output_folder (str): The folder where the altered audio will be saved.
    """
    for alter_name, alter_type in alterations.items():
        print(f'Doing {alter_name}')
        altered_waveform = apply_alteration_and_crop(copy.deepcopy(waveform), sample_rate, alter_type)
        torchaudio.save(f'{output_folder}/{alter_name}.wav', altered_waveform, sample_rate)

def generate_spectrogram(file_path, n_fft=400, hop_length=None, win_length=None):
    """
    Generates a spectrogram from an audio file.

    Parameters:
        file_path (str): Path to the audio file.
        n_fft (int): Number of FFT components.
        hop_length (int): Number of audio frames between STFT columns.
        win_length (int): Each frame of audio is windowed by window of length win_length.

    Returns:
        Tensor: The generated spectrogram.
        int: The sample rate of the audio.
    """
    waveform, sample_rate = torchaudio.load(file_path)
    spectrogram_transform = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=win_length, power=2)
    spectrogram = spectrogram_transform(waveform)
    return spectrogram, sample_rate

def plot_spectrograms(spectrograms, sample_rates, titles):
    """
    Plots multiple spectrograms.

    Parameters:
        spectrograms (list): A list of spectrogram tensors.
        sample_rates (list): A list of sample rates corresponding to each spectrogram.
        titles (list): A list of titles for each spectrogram plot.
    """
    n = len(spectrograms)
    plt.figure(figsize=(10 * n, 4))
    for i, (spectrogram, sample_rate, title) in enumerate(zip(spectrograms, sample_rates, titles), start=1):
        plt.subplot(1, n, i)
        plt.imshow(spectrogram.log2()[0,:,:].numpy(), cmap='viridis', aspect='auto', origin='lower')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.xlabel('Time Frame')
        plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()


def plot_audio_metrics_wide_layout(df,title='Effect of Noise levels and Metric comparison'):
    """
    Plots each audio metric from a DataFrame in separate subplots with more columns than rows.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing audio metrics. 
                         The first column should be the categorical names (e.g., "Audio Name"),
                         followed by columns of metrics.
    """
    num_metrics = len(df.columns) - 1  # Exclude the "Audio Name" column
    num_columns = min(num_metrics, 3)  # Set the number of columns, adjust this to increase/decrease columns
    num_rows = np.ceil(num_metrics / num_columns).astype(int)
    
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(5 * num_columns, 4 * num_rows))
    fig.subplots_adjust(hspace=1, wspace=0.3)  # Adjust space between plots

    # Ensure axs is an array even if single subplot
    if num_metrics == 1:
        axs = np.array([axs])
    axs = axs.flatten()
    
    for i, column in enumerate(df.columns[1:]):  # Skip the "Audio Name" column
        ax = axs[i]
        ax.plot(df["Audio Name"], df[column], marker='o', label=column)
        ax.set_title(column)
        ax.set_xlabel(df.columns[0].split('.')[0])

        ax.set_ylabel(column)
        ax.tick_params(axis='x', labelrotation=45)
        ax.legend()
    
    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.suptitle(title, fontsize=16)
    plt.show()