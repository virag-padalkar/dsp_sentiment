#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 20:34:50 2020

@author: virag
"""

import librosa
import matplotlib.pyplot as plt
import librosa.display as lbdisplay

def user_input():
    path = input ("Enter audio path: ")
    return path

def extract_features(audio_filepath):
    x, sr = librosa.load(audio_filepath)
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    mfccs = librosa.feature.mfcc(x,sr=sr)
    return x, sr, X, Xdb, mfccs

def display_features(audio_filepath):
    x, sr, X, Xdb, mfccs = extract_features(audio_filepath)
    plt.figure()
    lbdisplay.waveplot(x,sr=sr)
    plt.show()
    plt.figure()
    spectrogram = lbdisplay.specshow(Xdb,sr=sr,x_axis="time",y_axis="hz")
    plt.colorbar()
    plt.figure()
    mfcc = lbdisplay.specshow(mfccs,sr=sr,x_axis="time")
    plt.show()

display_features("osr_us_sample.wav")