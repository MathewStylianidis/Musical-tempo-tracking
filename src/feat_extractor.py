""" This script contains class definitions for classes used to extract features from audio """
from abc import ABC, abstractmethod
from typing import Tuple

import math
import numpy as np
from scipy.fft import fft


__author__ = "Matthaios Stylianidis"


def proper_round(x):
    """ Rounds x to the closest integer"""
    return math.ceil(x) if x % 1 >= 0.5 else round(x)


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
        hop_size (float): Hop size of STFT in seconds.
        hop_size_samples (int): Hop size of STFT in samples.
        fft_size_samples (int): FFT result size in samples.
        fft_resolution (float): Frequency resolution of FFT.
    """
    def __init__(self, audio_array: np.ndarray, sr: int, window_size: float, hop_size: float):
        super().__init__(audio_array)
        self.sr = sr
        self.window_size = window_size
        self.window_size_samples = int(self.sr * window_size)
        self.hop_size = hop_size
        self.hop_size_samples = int(self.sr * hop_size)
        self.fft_size_samples = self.window_size_samples // 2
        self.fft_resolution = self.sr / self.fft_size_samples

    @abstractmethod
    def extract_features(self):
        """ Abstract method for generating features from the given audio."""
        pass

    @abstractmethod
    def stft_function(self, frequencies: np.ndarray) -> np.ndarray:
        """ Abstract method that is applied to each set of frequencies extracted with STFT. """
        pass

    def extract_stft_features(self):
        """ Generator method for calling stft on a signal and applying a function on the result to calculate features.

        Returns:
            The features calculated using stft_function on each frame.
        """
        half_window_size = int(self.window_size_samples / 2)
        window_function = np.hanning(self.window_size_samples)
        start_index = half_window_size + 1
        for i in range(start_index, len(self.audio_array) - half_window_size, self.hop_size_samples):
            # Extract frame centered on index
            frame = self.audio_array[i - half_window_size:i + half_window_size + 1]
            frame *= window_function
            frequencies = np.abs(fft(frame)[0:self.window_size_samples//2] * 2 / self.window_size_samples)
            features = self.stft_function(frequencies)
            yield features


class DrumFreqFeatExtractor(BaseFreqFeatExtractor):
    """ A class for extracting features corresponding to different drum sounds.

    Attributes:
        audio_array (np.ndarray): A numpy array with the audio waveform values to extract features from.
        sr (int): Sampling rate of audio_array.
        window_size (float): Window size of STFT in seconds.
        window_size_samples (int): Window size of STFT in samples.
        hop_size (float): Hop size of STFT in seconds.
        hop_size_samples (int): Hop size of STFT in samples.
        low_freq_range_samples (int): Range of low frequencies in FFT results.
        med_freq_range_samples (int): Range of medium frequencies in FFT results.
        high_freq_range_samples (int): Range of high frequencies in FFT results.
        LOW_FREQ_RANGE (Tuple[int]): A tuple with the range of frequency values for low frequency drum parts such as
            a kick or a floor drum.
        MED_FREQ_RANGE (Tuple[int]): A tuple with the range of frequency values for medium frequency drum parts such as
            a snare drum or tom.
        HIGH_FREQ_RANGE (Tuple[int]): A tuple with the range of frequency values for low frequency drum parts such as
            certain toms or cymbals.
    """
    LOW_FREQ_RANGE = (0, 120)
    MED_FREQ_RANGE = (120, 300)
    HIGH_FREQ_RANGE = (300, 22050)

    def __init__(self, audio_array: np.ndarray, sr: int, window_size: float, hop_size: float):
        super().__init__(audio_array, sr, window_size, hop_size)
        self.low_freq_range_samples = (0,
                                       proper_round(self.LOW_FREQ_RANGE[1] / self.fft_resolution))
        self.med_freq_range_samples = (proper_round(self.MED_FREQ_RANGE[0] / self.fft_resolution),
                                       proper_round(self.MED_FREQ_RANGE[1] / self.fft_resolution))
        self.high_freq_range_samples = (proper_round(self.HIGH_FREQ_RANGE[0] / self.fft_resolution),
                                        self.fft_size_samples)

    def extract_features(self):
        """ Method generating features from the given audio.

        Returns:
            A np.ndarray with the frequency sum features calculated from the STFT frequencies.
        """
        X = []
        for features in self.extract_stft_features():
            X.append(features)
        X = np.stack(X)
        return X

    def stft_function(self, frequencies: np.ndarray) -> np.ndarray:
        """ Groups frequencies extracted with FFT into bands and calculates the sum over those bands.

        Args:
            frequencies (np.ndarray): A numpy array with the extracted features for a given audio frame.

        Returns:
            A numpy array with three elements consisting of the sum over the low, medium, and high frequency bands.
        """
        low_freq = frequencies[self.low_freq_range_samples[0]:self.low_freq_range_samples[1]]
        med_freq = frequencies[self.med_freq_range_samples[0]:self.med_freq_range_samples[1]]
        high_freq = frequencies[self.high_freq_range_samples[0]:self.high_freq_range_samples[1]]
        return np.array([low_freq.sum(), med_freq.sum(), high_freq.sum()])
