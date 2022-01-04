""" This script contains class definitions for classes used to extract features from audio """
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from scipy.fft import fft


__author__ = "Matthaios Stylianidis"

class BaseFeatExtractor(ABC):
    """ Base class for extracting features from audio

     Attributes:
        audio_array (np.ndarray): A numpy array with the audio waveform values to extract features from

    """

    def __init__(self, audio_array: np.ndarray):
        self.audio_array = audio_array

    @abstractmethod
    def extract_features(self):
        """ Abstract method for generating features from the given audio."""
        pass


class BaseFreqFeatExtractor(BaseFeatExtractor):
    """ Abstract class for extracting frequency based features using Short-Time-Fourier-Transform.

    Attributes:
        audio_array (np.ndarray): A numpy array with the audio waveform values to extract features from.
        sr (int): Sampling rate of audio_array.
        window_size (float): Window size of STFT in seconds.
        window_size_samples (int): Window size of STFT in samples.
        hop_size (float): Hope size of STFT in seconds.
        hope_size_samples (int): Hop size of STFT in samples.
    """
    def __init__(self, audio_array: np.ndarray, sr: int, window_size: float, hop_size: float):
        super().__init__(audio_array)
        self.sr = sr
        self.window_size = window_size
        self.window_size_samples = int(self.sr * window_size)
        self.hop_size = hop_size
        self.hop_size_samples = int(self.sr * hop_size)

    @abstractmethod
    def extract_features(self):
        """ Abstract method for generating features from the given audio."""
        pass

    def extract_stft_features(self, stft_function):
        """ Generator method for calling stft on a signal and applying a function on the result to calculate features.

        Args:
            stft_function (function): A function to apply to each window extracted with STFT.

        Returns:
            The features calculated using stft_function on each frame.
        """
        half_window_size = int(self.window_size_samples / 2)
        window_function = np.hamming(self.window_size_samples)
        index = + half_window_size + 1
        for i in range(index, len(self.audio_array) - half_window_size, self.hop_size_samples):
            frame = self.audio_array[index:index + self.hop_size_samples]
            frame *= window_function
            frequencies = fft(frame)[0:self.window_size_samples//2]
            features = stft_function(frequencies)
            yield features


class DrumFreqFeatExtractor(BaseFreqFeatExtractor):
    """ A class for extracting features corresponding to different drum sounds.

    Attributes:
        audio_array (np.ndarray): A numpy array with the audio waveform values to extract features from.
        sr (int): Sampling rate of audio_array.
        window_size (float): Window size of STFT in seconds.
        window_size_samples (int): Window size of STFT in samples.
        hop_size (float): Hope size of STFT in seconds.
        hope_size_samples (int): Hop size of STFT in samples.
        LOW_FREQ_RANGE (Tuple[int]): A tuple with the range of frequency values for low frequency drum parts such as
            a kick or a floor drum.
        MED_FREQ_RANGE (Tuple[int]): A tuple with the range of frequency values for medium frequency drum parts such as
            a snare drum or tom.
        HIGH_FREQ_RANGE (Tuple[int]): A tuple with the range of frequency values for low frequency drum parts such as
            certain toms or cymbals.
    """
    LOW_FREQ_RANGE = (50, 120)
    MED_FREQ_RANGE = (120, 300)
    HIGH_FREQ_RANGE = (300, 17000)

    def __init__(self, audio_array: np.ndarray, sr: int, window_size: float, hop_size: float):
        super().__init__(audio_array, sr, window_size, hop_size)

    def extract_features(self):
        """ Method generating features from the given audio.

        Returns:
            A np.ndarray with the frequency sum features calculated from the STFT frequencies.
        """
        X = []
        for features in self.extract_stft_features(self.extract_frequency_band_sums):
            X.append(features)
        X = np.stack(X)
        return X

    def extract_frequency_band_sums(self, frequencies: np.ndarray):
        """ Groups frequencies extracted with FFT into bands and calculates the sum over those bands.

        Args:
            frequencies (np.ndarray): A numpy array with the extracted features for a given audio frame.

        Returns:
            A numpy array with three elements consisting of the sum over the low, medium, and high frequency bands.
        """
        low_freq = frequencies
        med_freq = frequencies
        high_freq = frequencies
        return np.array(low_freq.sum(), med_freq.sum(), high_freq.sum())