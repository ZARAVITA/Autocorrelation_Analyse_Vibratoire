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
    vibration_data = pd.read_csv(uploaded_file, sep=";")
    
    # Affichage des premières lignes du dataset
    st.subheader("Aperçu des données")
    st.write(vibration_data.head())

    # Création des données de waveform
    x1 = vibration_data["time[ms]"] / 1000  # Convertir en secondes
    y1 = vibration_data[" amplitude[g]"]

    # Calcul de la fréquence d'échantillonnage
    dt = np.diff(x1)  # Calculer les intervalles de temps entre les échantillons
    fs = 1 / np.mean(dt)  # Fréquence d'échantillonnage en Hz
    st.write(f"Fréquence d'échantillonnage : {fs} Hz")

    # Calcul de la fonction d'autocorrélation
    df = vibration_data[" amplitude[g]"]
    N = len(df)
    y = []
    for j in range(0, (int(N / 2) + 1)):
        R = 0
        for i in range(0, (int(N / 2))):
            R = R + df[i] * df[i + j]
        y.append(R * 2 / N)

    # Calcul du coefficient d'autocorrélation
    df = pd.DataFrame(y)
    df.columns = ['R']
    R0 = df.iloc[0, 0]
    df['A'] = df['R'] / R0
    dt = vibration_data["time[ms]"] / 1000
    t = []
    for i in range(len(df)):
        t.append(dt[i])
    
    df['t'] = t

    x1 = df["t"]
    y1 = df["A"]

    # Création de la FFT du coefficient d'autocorrélation
    N = y1.size  # Nombre d'échantillons
    Te = 1 / fs  # Temps d'échantillonnage
    X = fft(y1)  # Transformée de Fourier rapide
    freq = fftfreq(y1.size, 1 / fs)  # Fréquence de la transformée de Fourier rapide

    # Prendre seulement les valeurs absolues positives et normalisation
    X_abs = np.abs(X[:N // 2]) * 2.0 / N
    freq_pos = freq[:N // 2]

    peakX = np.max(X_abs)  # Trouver le pic maximum
    locX = np.argmax(X_abs)  # Trouver sa position
    freq_posX = freq_pos[locX]  # Obtenir la valeur de fréquence correspondante

    # Graphique d'autocorrélation avec Plotly
    st.subheader("Graphique d'autocorrélation")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=x1, y=y1, mode='lines', name='Autocorrelation Factor'))
    fig1.update_layout(
        title="Autocorrelation Factor",
        xaxis_title="Time [s]",
        yaxis_title="Autocorrelation Factor",
        xaxis_range=[0, 200],  # Zoom
        hovermode="x unified"  # Afficher les coordonnées au survol
    )
    st.plotly_chart(fig1)

    # Graphique de la FFT avec Plotly
    st.subheader("FFT du coefficient d'autocorrélation")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=freq_pos, y=X_abs, mode='lines', name='Amplitude absolue'))
    fig2.add_trace(go.Scatter(x=[freq_posX], y=[peakX], mode='markers', name='Pic maximum'))
    fig2.update_layout(
        title=f'FFT of Autocorrelation Factor<br>Peak value: {peakX:.6f}, Location: {freq_posX:.6f} Hz',
        xaxis_title="Fréquence (Hz)",
        yaxis_title="Autocorrelation Factor",
        xaxis_range=[0, 0.390625 * fs],  # Zoom sur la fréquence
        hovermode="x unified"  # Afficher les coordonnées au survol
    )
    st.plotly_chart(fig2)

else:
    st.write("Veuillez importer un fichier CSV pour commencer l'analyse.")