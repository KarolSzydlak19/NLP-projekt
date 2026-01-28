import pandas as pd
import numpy as np
import glob
import os
import re
from scipy import stats

PATH_BERT = "bert_grad/"
PATH_GTE  = "gte/gradual/"
DRIFT_START = 11
TARGET_COL  = 'lsdd_p' 

# 1. PAROWANIE PLIKÓW (SORTOWANIE NUMERYCZNE)
def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def get_files(path):
    return sorted(glob.glob(os.path.join(path, "*.csv")), key=numerical_sort)

files_bert = get_files(PATH_BERT)
files_gte  = get_files(PATH_GTE)

min_len = min(len(files_bert), len(files_gte))
files_bert = files_bert[:min_len]
files_gte = files_gte[:min_len]

# 2. EKSTRAKCJA P-VALUE W MOMENCIE DRYFU
pairs = []

for f_b, f_g in zip(files_bert, files_gte):
    try:
        df_b = pd.read_csv(f_b)
        df_g = pd.read_csv(f_g)
        
        val_b = df_b[df_b['batch_num'] >= DRIFT_START][TARGET_COL].min()
        val_g = df_g[df_g['batch_num'] >= DRIFT_START][TARGET_COL].min()
        
        if pd.isna(val_b) or pd.isna(val_g):
            continue
            
        pairs.append((val_b, val_g))
        
        diff = val_b - val_g
        if diff == 0:
            status = "Identyczne"
        elif val_g < val_b:
            status = "GTE lepsze"
        else:
            status = "BERT lepszy"
            
        seed_name = os.path.basename(f_b).split('.')[0]
        print(f"{seed_name:<20} | {val_b:.2e}        | {val_g:.2e}        | {status}")

    except Exception as e:
        pass
# 3. TEST STATYSTYCZNY
bert_scores = np.array([p[0] for p in pairs])
gte_scores  = np.array([p[1] for p in pairs])

diffs = bert_scores - gte_scores
n_ties = np.sum(diffs == 0)


try:
    stat, p_val = stats.wilcoxon(bert_scores, gte_scores, alternative='greater')
    
    print("\n WYNIK TESTU WILCOXONA")
    print(f" p-value: {p_val:.6e}")
    
    if p_val < 0.05:
        print("GTE jest istotnie lepsze od BERTa")
    else:
        print("Brak istotnej różnicy statystycznej")

except ValueError as e:
    print(f"\nErr: {e}")
    if n_ties == len(pairs):
        print("Wszystkie wyniki są identyczne!")