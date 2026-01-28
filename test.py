import pandas as pd
import numpy as np
import glob
from scipy import stats

SEARCH_PATTERN = "bert_grad/sudden*.csv"
DRIFT_START = 11

all_files = glob.glob(SEARCH_PATTERN)
data_frames = []


for filename in all_files:
    try:
        df = pd.read_csv(filename)
        drift_zone = df[df['batch_num'] >= DRIFT_START].copy()
        
        if not drift_zone.empty:
            cols = ['lsdd_p', 'bbsd_pcorr', 'mmd_p']
            if set(cols).issubset(drift_zone.columns):
                data_frames.append(drift_zone[cols])
    except Exception:
        pass

if not data_frames:
    exit()

combined_drift = pd.concat(data_frames).reset_index(drop=True)

combined_drift = combined_drift.replace(0.0, 1e-20)


# 2. TEST WILCOXONA (PAIRED)
print("TESTY PARAMI")
print("Hipoteza: Algorytm A daje MNIEJSZE p-value (jest lepszy) niż Algorytm B")
print("-" * 60)

# Definiujemy pary do porównania
comparisons = [
    ("LSDD", 'lsdd_p', "BBSD", 'bbsd_pcorr'),
    ("LSDD", 'lsdd_p', "MMD", 'mmd_p'),
    ("BBSD", 'bbsd_pcorr', "MMD", 'mmd_p')
]

for name_a, col_a, name_b, col_b in comparisons:
    data_a = combined_drift[col_a]
    data_b = combined_drift[col_b]

    # alternative='less' oznacza: sprawdzamy czy mediana różnic (A - B) jest ujemna.
    # Czyli czy A ma zazwyczaj mniejsze p-value niż B.
    try:
        stat, p_val = stats.wilcoxon(data_a, data_b, alternative='less')
        
        verdict = "Istotnie lepszy" if p_val < 0.05 else "Brak roznicy"
        
        print(f"{name_a} vs {name_b}:")
        print(f"   Średnia A: {data_a.mean():.4e} | Średnia B: {data_b.mean():.4e}")
        print(f"   Wilcoxon p-value: {p_val:.6e} -> {verdict}")
        
        if p_val < 0.05:
            print(f"   Lepszy: {name_a}")
        else:
            print(f"   Brak roznic")
            
    except ValueError as e:
        print(f"{name_a} vs {name_b}:")
        print("   Algorytmy zwróciły identyczne wyniki (różnica zerowa).")

    print("-" * 30)