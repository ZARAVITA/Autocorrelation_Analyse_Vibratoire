# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 14:23:04 2025

@author: ZARAVITA Haydar
"""

import pandas as pd
import numpy as np
from numpy.fft import fft, fftfreq
import plotly.graph_objects as go
import streamlit as st

# Titre de l'application
st.title("Analyse Vibratoire - Autocorrélation et FFT")

# Upload du fichier CSV
uploaded_file = st.file_uploader("Importez votre fichier CSV", type=["csv"])

if uploaded_file is not None:
    # Lecture du fichier CSV
    vibration_data = pd.read_csv(uploaded_file, sep=";", skiprows=1)
    
    # Affichage des premières lignes du dataset
    st.subheader("Aperçu des données")
    st.write(vibration_data.head())

    # Extraction des données
    x1 = vibration_data.iloc[:, 0].values / 1000  # Convertir en secondes
    y1 = vibration_data[" amplitude[g]"].values

    # Calcul de la fréquence d'échantillonnage
    fs = 1 / np.mean(np.diff(x1))  # Fréquence d'échantillonnage
    st.write(f"Fréquence d'échantillonnage : {fs:.2f} Hz")

    # Calcul de la fonction d'autocorrélation
    autocorr = np.correlate(y1, y1, mode='full')
    autocorr = autocorr[autocorr.size // 2:] / autocorr.max()
    
    # Génération des valeurs de temps pour l'autocorrélation
    t = np.arange(len(autocorr)) / fs

    # Création de la FFT du coefficient d'autocorrélation
    N = len(autocorr)
    X = fft(autocorr)
    freq = fftfreq(N, 1 / fs)[:N // 2]
    X_abs = np.abs(X[:N // 2]) * 2.0 / N

    # Détection du pic maximal
    locX = np.argmax(X_abs)
    peakX = X_abs[locX]
    freq_posX = freq[locX]

    # Graphique d'autocorrélation
    st.subheader("Graphique d'autocorrélation")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=t, y=autocorr, mode='lines', name='Autocorrelation Factor'))
    fig1.update_layout(
        title="Autocorrelation Factor",
        xaxis_title="Time [s]",
        yaxis_title="Autocorrelation Factor",
        xaxis_range=[0, 0.2],
        hovermode="x unified"
    )
    st.plotly_chart(fig1)

    # Graphique de la FFT
    st.subheader("FFT du coefficient d'autocorrélation")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=freq, y=X_abs, mode='lines', name='Amplitude absolue'))
    fig2.add_trace(go.Scatter(x=[freq_posX], y=[peakX], mode='markers', name='Pic maximum'))
    fig2.update_layout(
        title=f'FFT of Autocorrelation Factor<br>Peak value: {peakX:.6f}, Location: {freq_posX:.6f} Hz',
        xaxis_title="Fréquence (Hz)",
        yaxis_title="Autocorrelation Factor",
        xaxis_range=[0, 0.4 * fs],
        hovermode="x unified"
    )
    st.plotly_chart(fig2)

else:
    st.write("Veuillez importer un fichier CSV pour commencer l'analyse.")
