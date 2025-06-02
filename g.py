import pandas as pd
import numpy as np
import os
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# ─── KONFIGURATION ─────────────────────────────────────────────────────────────
# Glättungsfenster (zwischen 10 und 20)
window_size = 15
# Parameter für die Dip-Erkennung
prominence  = 0.05   # relative Tropfentiefe (5 % des Maximalwerts)
width_min   = 5      # minimale Breite eines Dips (in Zeilen)
width_max   = 20     # maximale Breite eines Dips (in Zeilen)
# ────────────────────────────────────────────────────────────────────────────────

# 0) Verzeichnis des Skripts ermitteln
script_dir = os.path.dirname(os.path.abspath(__file__))

# 1) Alle CSV-Dateien im Skript-Verzeichnis finden
csv_files = [
    fname for fname in os.listdir(script_dir)
    if fname.lower().endswith(".csv")
]

# 2) Jede CSV-Datei verarbeiten
for csv_file in csv_files:
    file_path = os.path.join(script_dir, csv_file)
    base_name = os.path.splitext(csv_file)[0]
    output_dir = os.path.join(script_dir, f"{base_name}_segments")
    os.makedirs(output_dir, exist_ok=True)

    # 2a) CSV einlesen
    df = pd.read_csv(file_path)

    # 2b) sum_dev berechnen: Summe der absoluten Abweichungen
    df["sum_dev"] = (
          df["x-wert"].abs()
        + df["y-wert"].abs()
        + df["z-wert"].abs()
    )

    # 2c) Glätten mit Moving Average (zentriertes Fenster)
    df["smooth"] = (
        df["sum_dev"]
          .rolling(window=window_size, center=True, min_periods=1)
          .mean()
    )

    # 2d) Invertierte Kurve, damit wir Täler als Peaks finden
    inv = df["smooth"].max() - df["smooth"]

    # 2e) Dips mittels scipy.signal.find_peaks identifizieren
    peaks, _ = find_peaks(
        inv,
        prominence = prominence * df["smooth"].max(),
        width      = (width_min, width_max)
    )

    # 2f) Optional: Marker-Spalte setzen (zur Kontrolle)
    df["marker"] = ""
    df.loc[peaks, "marker"] = "x"

    # 3) Für jedes Intervall zwischen zwei aufeinanderfolgenden Dips einen Plot erzeugen
    for i in range(len(peaks) - 1):
        start_idx = peaks[i]
        end_idx   = peaks[i + 1]
        segment   = df.loc[start_idx:end_idx]

        plt.figure(figsize=(5, 3))
        plt.plot(segment["x-wert"], label="x-wert")
        plt.plot(segment["y-wert"], label="y-wert")
        plt.plot(segment["z-wert"], label="z-wert")
        plt.axis("off")  # Achsen und Rahmen ausblenden

        out_path = os.path.join(output_dir, f"{base_name}_segment_{i+1}.png")
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    print(f"✔ {csv_file}: {max(len(peaks)-1, 0)} Segment-Plots gespeichert in '{output_dir}'")
