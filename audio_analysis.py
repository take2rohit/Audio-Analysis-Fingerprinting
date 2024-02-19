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


class AudioAnalyzer:
    """
    A class for analyzing audio files, generating spectrograms, and comparing them using various metrics.
    """

    def __init__(self, file_paths):
        """
        Initializes the AudioAnalyzer with a list of audio file paths.

        Parameters:
            file_paths (list): A list of strings, where each string is a path to an audio file.
        """
        self.file_paths = file_paths
        self.spectrograms = []
        self.sample_rates = []
        self.titles = []
        self.df_metrics = pd.DataFrame(columns=['Audio Name', 'MSE', 'PSNR', 'SSIM', 'Cosine Similarity', 'KL Divergence', 'Histogram Correlation'])


    def generate_spectrograms(self, n_fft=400, hop_length=None, win_length=None):
        """
        Generates and stores spectrograms for all audio files initialized with the class.

        Parameters:
            n_fft (int): Number of FFT components.
            hop_length (int): Number of audio frames between STFT columns.
            win_length (int): Each frame of audio is windowed by window of length win_length.
        """
        for file_path in self.file_paths:
            waveform, sample_rate = torchaudio.load(file_path)
            spectrogram_transform = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=win_length, power=2)
            spectrogram = spectrogram_transform(waveform)
            self.spectrograms.append(spectrogram)
            self.sample_rates.append(sample_rate)
            self.titles.append(file_path.split('/')[-1])

    @staticmethod
    def compare_mse(original_spectrogram, altered_spectrogram):
        """
        Compares two spectrograms using Mean Squared Error (MSE).

        Parameters:
            original_spectrogram (Tensor): The original spectrogram.
            altered_spectrogram (Tensor): The altered spectrogram to compare against.

        Returns:
            float: The MSE value.
        """
        mse = torch.mean((original_spectrogram - altered_spectrogram) ** 2).item()
        return mse

    @staticmethod
    def compare_psnr(original_spectrogram, altered_spectrogram):
        """
        Compares two spectrograms using Peak Signal-to-Noise Ratio (PSNR).

        Parameters:
            original_spectrogram (Tensor): The original spectrogram.
            altered_spectrogram (Tensor): The altered spectrogram to compare against.

        Returns:
            float: The PSNR value.
        """
        mse = torch.mean((original_spectrogram - altered_spectrogram) ** 2)
        max_pixel = torch.max(original_spectrogram)
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse)).item()
        return psnr

    @staticmethod
    def compare_ssim(spectrogram1, spectrogram2):
        """
        Compares two spectrograms using Structural Similarity Index (SSIM).

        Parameters:
            spectrogram1 (Tensor): The first spectrogram.
            spectrogram2 (Tensor): The second spectrogram.

        Returns:
            float: The average SSIM value.
        """
        spectrogram1_np = spectrogram1.numpy().transpose(1, 2, 0)
        spectrogram2_np = spectrogram2.numpy().transpose(1, 2, 0)
        
        ssim_scores = []
        for i in range(spectrogram1_np.shape[-1]):  # Iterate through channels
            score = ssim(spectrogram1_np[..., i], spectrogram2_np[..., i], data_range=spectrogram2_np[..., i].max() - spectrogram2_np[..., i].min())
            ssim_scores.append(score)
        
        avg_ssim = sum(ssim_scores) / len(ssim_scores)
        return avg_ssim

    @staticmethod
    def compare_cosine_similarity(original_spectrogram, altered_spectrogram):
        """
        Compares two spectrograms using cosine similarity.

        Parameters:
            original_spectrogram (Tensor): The original spectrogram.
            altered_spectrogram (Tensor): The altered spectrogram to compare against.

        Returns:
            float: The cosine similarity value.
        """
        original_flat = original_spectrogram.flatten().numpy()
        altered_flat = altered_spectrogram.flatten().numpy()
        similarity = 1 - cosine(original_flat, altered_flat)
        return similarity

    @staticmethod
    def compare_kl_divergence(original_spectrogram, altered_spectrogram):
        """
        Compares two spectrograms using Kullback-Leibler Divergence.

        Parameters:
            original_spectrogram (Tensor): The original spectrogram.
            altered_spectrogram (Tensor): The altered spectrogram to compare against.

        Returns:
            float: The KL divergence value.
        """
        original_flat = original_spectrogram.flatten().numpy() + 1e-10  # Avoid log(0)
        altered_flat = altered_spectrogram.flatten().numpy() + 1e-10
        kl_div = entropy(original_flat, altered_flat)
        return kl_div

    @staticmethod
    def compare_histogram_correlation(original_spectrogram, altered_spectrogram):
        """
        Compares two spectrograms using histogram correlation.

        Parameters:
            original_spectrogram (Tensor): The original spectrogram.
            altered_spectrogram (Tensor): The altered spectrogram to compare against.

        Returns:
            float: The histogram correlation value.
        """
        original_hist = np.histogram(original_spectrogram.flatten().numpy(), bins=256)[0]
        altered_hist = np.histogram(altered_spectrogram.flatten().numpy(), bins=256)[0]
        correlation = np.corrcoef(original_hist, altered_hist)[0, 1]
        return correlation

    def rank_alterations(self):
        """
        Ranks the alterations of the spectrograms based on PSNR.

        Returns:
            list: A sorted list of tuples with the alteration name and its PSNR score.
        """
        scores = {}
        original_spectrogram = self.spectrograms[0]  # Assuming the first spectrogram is the original
        for i, spectrogram in enumerate(self.spectrograms):
            psnr = self.compare_psnr(original_spectrogram, spectrogram)
            scores[self.titles[i]] = psnr

        ranked_alterations = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked_alterations
    
    def compute_metrics(self, original_spectrogram, altered_spectrogram):
        """
        Computes various metrics to compare the original and altered spectrograms.

        Parameters:
            original_spectrogram (Tensor): The original spectrogram.
            altered_spectrogram (Tensor): The altered spectrogram to compare against.

        Returns:
            dict: A dictionary of computed metrics.
        """
        metrics = {
            'MSE': self.compare_mse(original_spectrogram, altered_spectrogram),
            'PSNR': self.compare_psnr(original_spectrogram, altered_spectrogram),
            'SSIM': self.compare_ssim(original_spectrogram, altered_spectrogram),
            'Cosine Similarity': self.compare_cosine_similarity(original_spectrogram, altered_spectrogram),
            'KL Divergence': self.compare_kl_divergence(original_spectrogram, altered_spectrogram),
            'Histogram Correlation': self.compare_histogram_correlation(original_spectrogram, altered_spectrogram)
        }
        return metrics

    def populate_metrics_dataframe(self):
        """
        Populates the DataFrame with metrics comparing each altered spectrogram to the original.
        Assumes the first spectrogram in self.spectrograms is the original for comparison.
        """
        original_spectrogram = self.spectrograms[0]  # Assuming the first spectrogram is the original
        for title, spectrogram in zip(self.titles, self.spectrograms):
            metrics = self.compute_metrics(original_spectrogram, spectrogram)
            row = [title] + list(metrics.values())
            self.df_metrics.loc[len(self.df_metrics)] = row


if __name__ == "__main__":
    
    # Example usage:
    file_paths = ['data/altered_audio/original_cropped.wav', 
                  'data/altered_audio/minor_alter.wav', 
                  'data/altered_audio/moderate_alter.wav', 
                  'data/altered_audio/strong_alter.wav']

    analyzer = AudioAnalyzer(file_paths)
    analyzer.generate_spectrograms()
    analyzer.populate_metrics_dataframe()
    analyzer.df_metrics
