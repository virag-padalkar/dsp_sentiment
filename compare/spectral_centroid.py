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

# load audio and extract features
x1, sr1 = librosa.load("mkgandhi.oga")
X1 = librosa.stft(x1)
Xdb1 = librosa.amplitude_to_db(abs(X1))
x2, sr2 = librosa.load("ahitler.ogg")
X2 = librosa.stft(x2)
Xdb2 = librosa.amplitude_to_db(abs(X2))
sc1 = librosa.feature.spectral_centroid(y=x1,sr=sr1)
sc2 = librosa.feature.spectral_centroid(y=x2,sr=sr2)

# plot audio
plt.subplots(2, 2)
ax1 = plt.subplot(2,1,1)
ax1.title.set_text("MK Gandhi")
lbdisplay.waveplot(x1,sr=sr1)
ax1.set_ylim([-1, 1])
ax2 = plt.subplot(2,1,2)
ax2.title.set_text("A Hitler")
lbdisplay.waveplot(x2,sr=sr2)
ax2.set_ylim([-1,1])
plt.tight_layout()
fig1, ax1 = plt.subplots()
librosa.display.specshow(Xdb1, sr=sr1, x_axis='time', y_axis='hz')
ax1.set(title="MK Gandhi")
plt.colorbar()
plt.tight_layout()
fig2, ax2 = plt.subplots()
ax2.set(title="A Hitler")
librosa.display.specshow(Xdb2, sr=sr2, x_axis='time', y_axis='hz')
plt.colorbar()
plt.tight_layout()