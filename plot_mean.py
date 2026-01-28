import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

SEARCH_PATTERN = "gte/gradual/gradual_*.csv"
# Parametry wykresu
DRIFT_START_BATCH = 21
ALPHA = 0.05

# 1. WCZYTYWANIE DANYCH
all_files = glob.glob(SEARCH_PATTERN)

if not all_files:
    exit()

data_frames = []

# Lista kolumn
REQUIRED_COLS = ['batch_num', 'lsdd_p', 'bbsd_pcorr', 'mmd_p', 'tested']

for filename in all_files:
    try:
        try:
            df = pd.read_csv(filename, usecols=lambda c: c in REQUIRED_COLS)
        except ValueError:
            df = pd.read_csv(filename)

        if 'batch_num' not in df.columns:
            continue

        df = df[pd.to_numeric(df['batch_num'], errors='coerce').notnull()]
        
        cols_to_numeric = [c for c in df.columns if c in REQUIRED_COLS]
        df[cols_to_numeric] = df[cols_to_numeric].apply(pd.to_numeric, errors='coerce')

        if 'tested' in df.columns:
            df = df[df['tested'] == 1]
            
                
        data_frames.append(df)

    except Exception as e:
        pass

if not data_frames:
    exit()

combined = pd.concat(data_frames)
# 2. AGREGACJA (ŚREDNIA I ODCHYLENIE)
grouped = combined.groupby('batch_num')

mean_df = grouped.mean()
std_df = grouped.std().fillna(0)

limit_mask = mean_df.index <= 22
mean_df = mean_df[limit_mask]
std_df = std_df[limit_mask]

x_vals = mean_df.index 

plt.figure(figsize=(12, 7))
plt.title("Detekcja Dryfu (Dryf stopniowy, model embeddingowy BERT) - Średnia liniowa", fontsize=14)
plt.xlabel("Numer Okna")
plt.ylabel("P-value (skala liniowa)")

def plot_linear_mean(ax, x_idx, means, stds, col, label, color, marker):
    if col not in means.columns: return
    y_mean = means[col]
    y_std = stds[col]
    ax.plot(x_idx, y_mean, label=label, color=color, linewidth=2, marker=marker, markersize=5)
    lower_bound = np.maximum(y_mean - y_std, 0)
    upper_bound = np.minimum(y_mean + y_std, 1.05)
    ax.fill_between(x_idx, lower_bound, upper_bound, color=color, alpha=0.15)

plot_linear_mean(plt, x_vals, mean_df, std_df, 'lsdd_p', 'LSDD', 'blue', 's')
plot_linear_mean(plt, x_vals, mean_df, std_df, 'bbsd_pcorr', 'MKS', 'green', '^')
plot_linear_mean(plt, x_vals, mean_df, std_df, 'mmd_p', 'MMD', 'red', 'o')

plt.axvline(x=DRIFT_START_BATCH, color='black', linestyle='--', label='Start Dryfu')
plt.axhline(y=ALPHA, color='gray', linestyle=':', label=f'Próg {ALPHA}')
plt.ylim(-0.05, 1.1)
plt.legend(loc='lower left')
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()