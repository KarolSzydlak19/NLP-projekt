import pandas as pd
import glob
import os

SEARCH_PATTERN = "bert_sudden/sudden*.csv"

OUTPUT_FILENAME = "tabela_p_value_bert_50_gradualcsv" 

# Nazwy kolumn z p-value
METRIC_COLS = ['lsdd_p', 'bbsd_pcorr', 'mmd_p']

# 1. WCZYTYWANIE DANYCH
print(f"Szukam plików: {SEARCH_PATTERN}")
all_files = glob.glob(SEARCH_PATTERN)

data_frames = []

for filename in all_files:
    try:
        df = pd.read_csv(filename)
        
        if 'tested' in df.columns:
            df = df[df['tested'] == 1]
            
        df['batch_num'] = pd.to_numeric(df['batch_num'], errors='coerce')
        
        cols_to_keep = ['batch_num'] + METRIC_COLS
        
        if set(METRIC_COLS).issubset(df.columns):
            data_frames.append(df[cols_to_keep])
            
    except Exception as e:
        print(f"Pass {filename}: {e}")

if not data_frames:
    exit()

combined = pd.concat(data_frames)

# 2. OBLICZANIE TABELI (AGREGACJA - ŚREDNIA)

final_df = combined.groupby('batch_num')[METRIC_COLS].mean()

final_df = final_df.sort_index()

# 3. ZAPIS DO PLIKU
final_df.to_csv(OUTPUT_FILENAME, sep=',')
