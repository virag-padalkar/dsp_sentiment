#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 21:18:20 2020

@author: virag
"""

import librosa
import librosa.display as lbdisplay
import matplotlib.pyplot as plt
import sklearn

x, sr = librosa.load("bassoon_mozart.ogg")
X = librosa.stft(x)
scentroid = librosa.feature.spectral_centroid(y=x,sr=sr)
frames = range(len(scentroid))  # since every column in numpy array is equal to the number of frames
t = librosa.frames_to_time(frames)
