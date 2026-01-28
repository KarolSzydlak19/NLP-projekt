import pandas as pd
import numpy as np
import glob
from scipy import stats

PATTERN_A = "bert_50/gradual_bert_50/gradual*.csv"   
PATTERN_B = "bert_grad/sudden*.csv" 

NAME_A = "Okno 50 (LSDD)"
NAME_B = "Okno 100 (LSDD)"

DRIFT_START = 11      
TARGET_COL = 'lsdd_p'
ALPHA = 0.05

def collect_drift_values(file_pattern, drift_start_idx, col_name):
    files = glob.glob(file_pattern)
    collected_values = []
    
    
    for filename in files:
        try:
            df = pd.read_csv(filename)
            drift_zone = df[df['batch_num'] >= drift_start_idx]
            
            if not drift_zone.empty and col_name in drift_zone.columns:
                collected_values.extend(drift_zone[col_name].values)
        except Exception as e:
            pass            
    return np.array(collected_values)

# 2. GROMADZENIE DANYCH

values_a = collect_drift_values(PATTERN_A, DRIFT_START, TARGET_COL)
values_b = collect_drift_values(PATTERN_B, DRIFT_START, TARGET_COL)

values_a = np.where(values_a == 0.0, 1e-20, values_a)
values_b = np.where(values_b == 0.0, 1e-20, values_b)

if len(values_a) == 0 or len(values_b) == 0:
    exit()

# 3. TEST MANNA-WHITNEYA
print(f"Hipoteza: {NAME_A} daje MNIEJSZE p-value (szybsza detekcja) niż {NAME_B}")
print("-" * 60)

# Ustawienie zmiennych
name_1, d1 = NAME_A, values_a
name_2, d2 = NAME_B, values_b

try:
    # Test Manna-Whitneya (dla prób niezależnych o różnej liczności)
    stat, p_val = stats.mannwhitneyu(d1, d2, alternative='less')
    
    verdict = "Istotnie lepszy" if p_val < ALPHA else "Brak roznic"
    
    print(f"{name_1} vs {name_2}:")
    print(f"   Średnia A: {d1.mean():.4e} | Średnia B: {d2.mean():.4e}")
    print(f"   Mediana A: {np.median(d1):.4e} | Mediana B: {np.median(d2):.4e}")
    print(f"   M-W p-value: {p_val:.6e} -> {verdict}")
    
    if p_val < ALPHA:
        print(f"   LEPSZY: {name_1}")
    else:
        print(f"   REMIS")

except Exception as e:
    pass