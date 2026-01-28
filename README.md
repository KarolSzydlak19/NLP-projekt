# Detekcja Dryfu w Danych

## Opis
Projekt implementuj detekcję dryfu w danych używając trzech algorytmów:
- **LSDD** (Least-Squares Density Difference)
- **BBSD+MKS** (PCA + Multiple Kolmogorov-Smirnov Tests + Bonferroni Correction)
- **UAE+MMD** (Untrained AutoEncoder + Maximum Mean Discrepancy)

Główny skrypt detekcji to **`data_drift_detection.py`**.

## Instalacja

```bash
pip install -r requirements.txt
```

## Przygotowanie danych

### 1. Sortowanie danych Yelp (opcjonalne)
```bash
python sort.py
```

### 2. Tworzenie zbalansowanego datasetu
```bash
python prepare_balanced_dataset.py \
  --ip "./nlp-datasets/yelp_dataset/yelp_academic_dataset_review.json" \
  --bp "./nlp-datasets/yelp_dataset/yelp_academic_dataset_business.json" \
  --op "./data/sudden_drift.jsonl" \
  --dt "sudden"
```

**Parametry:**
- `--ip`: ścieżka do review.json
- `--bp`: ścieżka do business.json  
- `--op`: plik wyjściowy
- `--dt`: typ dryfu (`sudden`, `gradual`, `sudden_by_year`)
- `--ca`: kategoria A (domyślnie "Restaurants")
- `--cb`: kategoria B (domyślnie "Beauty & Spas")
- `--ds`: rozmiar datasetu (domyślnie 20,000)

### 3. Generowanie embeddingów
```bash
python embed.py \
  --ip "./data/sudden_drift.jsonl" \
  --op "./data/sudden_drift_embedded.jsonl" \
  --m "sentence-transformers/all-MiniLM-L6-v2"
```

## Detekcja dryfu

```bash
python data_drift_detection.py \
  --file "./data/sudden/bert/_0.json_embedded.jsonl" \
  --batch_size 100 \
  --ref_size 1000 \
  --alpha 0.05 \
  --meta "acat" \
  --patience 2 \
  --test_every 1 \
  --out_csv "./results/drift_results.csv"
```

**Parametry:**
- `--file`: plik z embeddingami (JSONL)
- `--batch_size`: rozmiar okna testowego (domyślnie 100)
- `--ref_size`: rozmiar danych referencyjnych (domyślnie 1000)
- `--alpha`: poziom istotności globalny (domyślnie 0.05)
- `--meta`: metoda meta-testu (`acat` lub `fisher`)
- `--patience`: ile kolejnych detekcji dryfu przed zatrzymaniem (domyślnie 2)
- `--test_every`: testuj co k-ty batch (domyślnie 1)
- `--out_csv`: plik wyjściowy CSV z wynikami

## Analiza wyników

### 1. Tabela średnich p-value
```bash
python table_mean.py
```

### 2. Wizualizacja wyników
```bash
python plot_mean.py
```

### 3. Testy statystyczne
```bash
# Porównanie algorytmów
python test.py

# Porównanie różnych rozmiarów okien
python test_okno.py

# Porównanie modeli embeddingowych
python test_model.py
```

## Struktura katalogów

```
./data/               # dane wejściowe
./results/           # wyniki CSV
```

## Przykład uruchomienia

```bash
# 1. Przygotuj dane
python prepare_balanced_dataset.py --ip review.json --bp business.json --op data.jsonl --dt sudden

# 2. Wygeneruj embeddingi
python embed.py --ip data.jsonl --op data_embedded.jsonl

# 3. Uruchom detekcję dryfu
python data_drift_detection.py --file ./data/sudden/bert/_0.json_embedded.jsonl --out_csv results.csv

# 4. Analizuj wyniki
python plot_mean.py
```
