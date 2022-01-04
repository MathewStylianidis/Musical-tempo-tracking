# Tempo tracking using probabilistic estimation/ML/signal processing methods

This repository contains work from a project done for the EL2320 course (Applied Estimation) in KTH. 

## Environment details

* Python 3.9

## XGBoost beat detector + Particle filter tempo tracker

In this approach an XGBoost classifier is trained to detect beats from the audio using spectral features and a particle filter uses the XGBoost output as observations and tracks the beat location and the tempo period in a song.

#### Datasets

* [Groove MIDI dataset](https://magenta.tensorflow.org/datasets/groove#midi-data)

#### XGBoost beat detection model

We perform STFT with a window size of 0.1s and a hop size of 0.0375s, which means that we detect beats within audio blocks slightly larger than than 16th note sized blocks, with a resolution corresponding to 32th notes, for a maximum tempo value of 200 BPM. A hann window is used with STFT. We then split the frequency bands to the ranges (50,120), (120,300) and (300, 22050) and calculate the sum over those frequency bands as features. XGboost then takes as input those sum values for 32 time steps, which corresponds to 96 values in total. The prediction of XGBoost infers whether there is a beat or not at the center of the last STFT frame.

## Particle filter tempo tracking on MIDI onset times

We extract the note onsets from the available MIDI files and perform tempo tracking using the method proposed in [this paper](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.217.32&rep=rep1&type=pdf), using a particle filter switching state model.

### Datasets

* [Groove MIDI dataset](https://magenta.tensorflow.org/datasets/groove#midi-data)
