""" Script defining classes for loading different datasets (e.g. building generators, etc.)
"""
from abc import ABC, abstractmethod
import os

import librosa
import numpy as np
import pandas as pd
from typing import List

import pretty_midi

__author__ = "Matthaios Stylianidis"


class DataLoader(ABC):
    """Abstract class for data loader - a class for loading various datasets

    Attributes:
        dataset_root_path (str): The path to the root directory of the dataset.
    """

    def __init__(self, dataset_root_path: str):
        self.dataset_root_path = dataset_root_path

    @abstractmethod
    def get_data(self, split: str):
        pass


class GmdDataLoader(DataLoader):
    """Class for loading the GMD (Groove MIDI dataaset)

    Attributes:
        LABEL_REL_PATH (str): The relative path to the label csv file within the dataset root directory.
        VALID_FILE_TYPES (List[str]): Possible file types to be loaded (i.e. wav or MIDI types).
        VALID_SPLIT_TYPES (List[str]): Possible split types to be loaded (i.e. training or test split).
        dataset_root_path (str): The path to the root directory of the dataset.
        sr (int): Sampling rate of recorded files. Initialized after audio files start being loaded (e.g.
            by running get_data).
    """

    LABEL_REL_PATH = os.path.join("info.csv")  # Path to labels within directory
    WAV_TYPE = "wav"
    MIDI_TYPE = "midi"
    VALID_FILE_TYPES = [WAV_TYPE, MIDI_TYPE]
    TRAIN_SPLIT = "train"
    TEST_SPLIT = "test"
    VALID_SPLIT_TYPES = [TRAIN_SPLIT, TEST_SPLIT]

    def __init__(self, dataset_root_path: str):
        super().__init__(dataset_root_path)
        self.sr = None

    def get_data(self, split: str, min_duration: float = 30.0) -> (np.ndarray, pd.Series, float):
        """ Generator function returning a tuple with wav file numpy content, a pandas series with the meta-data and
        a float corresponding to the onset time of the first note in the MIDI file.

        Args:
            split: The data split to return (e.g. train for training data or test for test data)
            min_duration (float): The minimum duration for the recordings returned. All recordings with a length lower
                than that are discarded.

        Returns:
            A tuple(np.ndarray, pd.Series, float).

        Raises:
            Exception if the split is not any of the valid split types.
        """

        if split not in self.VALID_SPLIT_TYPES:
            raise Exception("Split type not in valid split types: {}".split(str(self.VALID_SPLIT_TYPES)))

        meta_data_df = pd.read_csv(os.path.join(self.dataset_root_path, self.LABEL_REL_PATH))
        meta_data_df = meta_data_df[meta_data_df["split"] == split]

        for i in range(meta_data_df.shape[0]):
            meta_data_row = meta_data_df.iloc[i, :]
            if meta_data_row["duration"] < min_duration:
                continue
            wav_file_path = os.path.join(self.dataset_root_path, meta_data_row["audio_filename"])
            wav_array, self.sr = librosa.load(wav_file_path)
            midi_file_path = os.path.join(self.dataset_root_path, meta_data_row["midi_filename"])
            start_time = pretty_midi.PrettyMIDI(midi_file_path, initial_tempo=meta_data_row["bpm"]).get_onsets()[0]
            yield wav_array, meta_data_row, start_time

    def get_dataset_size(self, split: str = None, min_duration: float = 0.0):
        """ Gets the size of the dataset given the specified filtering arguments.

        Args:
            split: The data split to return (e.g. train for training data or test for test data).
            min_duration (float): The minimum duration for the recordings returned. All recordings with a length lower
            than that are discarded.

        Returns:
            An integer with the dataset size.

        Raises:
            Exception if the split is not any of the valid split types.
       """
        if split is not None and split not in self.VALID_SPLIT_TYPES:
            raise Exception("Split type not in valid split types: {}".split(str(self.VALID_SPLIT_TYPES)))

        meta_data_df = pd.read_csv(os.path.join(self.dataset_root_path, self.LABEL_REL_PATH))
        if split is not None:
            meta_data_df = meta_data_df[meta_data_df["split"] == split]
        meta_data_df = meta_data_df[meta_data_df["duration"] >= min_duration]

        return len(meta_data_df)

