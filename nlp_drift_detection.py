import json
import math
import numpy as np
import traceback
from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# --- IMPORTY ---
from frouros.detectors.concept_drift import ADWIN, ADWINConfig
from frouros.detectors.data_drift.batch.distance_based import MMD, EnergyDistance
from scipy.stats import ks_2samp

# KONFIGURACJA
FILE_PATH = (
    "./yelp_academic_dataset_review_embed/yelp_academic_dataset_review_embed.json"
)
WARMUP_SIZE = 300
TEST_WINDOW_SIZE = 300
CHECK_INTERVAL = 50
ADWIN_DELTA = 0.002


def sigmoid(x):
    if x < -500:
        return 0.0
    if x > 500:
        return 1.0
    return 1 / (1 + math.exp(-x))


def process_stream_features_only(file_path):
    print(f"⏳ Otwieranie strumienia: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                yield (
                    np.array(data["embedding"]),
                    data.get("date", "N/A"),
                    data.get("text", ""),
                )
            except json.JSONDecodeError:
                continue


# ==========================================
# 1. MODEL BAZOWY
# ==========================================
scaler = StandardScaler()
anomaly_model = OneClassSVM(nu=0.1, kernel="rbf", gamma="scale")

# ==========================================
# 2. DETEKTORY DRYFU
# ==========================================
adwin_config = ADWINConfig(delta=ADWIN_DELTA)
adwin = ADWIN(config=adwin_config)

energy_dist = EnergyDistance()
mmd = MMD()

# Zmienne stanu
step = 0
warmup_data = []
warmup_scores = []
current_window = deque(maxlen=TEST_WINDOW_SIZE)
current_scores_window = deque(maxlen=TEST_WINDOW_SIZE)
model_trained = False

print("🚀 Rozpoczynam Detekcję Dryfu (Fix Types)")

try:
    for embedding, date_str, text in process_stream_features_only(FILE_PATH):
        step += 1

        # -------------------------------------------------
        # FAZA 1: WARMUP
        # -------------------------------------------------
        if not model_trained:
            warmup_data.append(embedding)
            if len(warmup_data) >= WARMUP_SIZE:
                print(f"\n🔧 Warmup zakończony. Trenuję SVM...")
                X_ref = np.array(warmup_data)

                scaler.fit(X_ref)
                X_ref_scaled = scaler.transform(X_ref)
                anomaly_model.fit(X_ref_scaled)

                raw_scores_ref = anomaly_model.score_samples(X_ref_scaled)
                warmup_scores = [sigmoid(-s) for s in raw_scores_ref]

                model_trained = True
                print("✅ Model gotowy.\n")
            continue

        # -------------------------------------------------
        # FAZA 2: DETEKCJA ONLINE
        # -------------------------------------------------

        # Scoring
        X_sample = embedding.reshape(1, -1)
        X_scaled = scaler.transform(X_sample)
        raw_score = anomaly_model.score_samples(X_scaled)[0]
        prob_anomaly = sigmoid(-raw_score)

        # Aktualizacja buforów
        current_window.append(embedding)
        current_scores_window.append(prob_anomaly)

        # 1. ADWIN
        adwin.update(value=prob_anomaly)
        if adwin.drift:
            print(f"🚨 [ADWIN] Drift w kroku {step}! (Prob: {prob_anomaly:.4f})")
            adwin.reset()

        # -------------------------------------------------
        # TESTY STATYSTYCZNE (Poprawiona logika)
        # -------------------------------------------------
        if len(current_window) == TEST_WINDOW_SIZE and step % CHECK_INTERVAL == 0:

            X_ref = np.array(warmup_data)
            X_curr = np.array(current_window)
            S_ref = np.array(warmup_scores)
            S_curr = np.array(current_scores_window)

            print(f"🔍 [KROK {step}] Testy statystyczne...")
            drift_reasons = []

            # --- POPRAWKA 1: KS TEST ---
            # ks_2samp zwraca obiekt/tuple. Musimy wyciągnąć pvalue.
            ks_res = ks_2samp(S_ref, S_curr)

            # Obsługa różnych wersji scipy (obiekt vs tuple)
            p_value = ks_res.pvalue if hasattr(ks_res, "pvalue") else ks_res[1]

            if p_value < 0.01:
                drift_reasons.append(f"KS (p={p_value:.2e})")

            # --- POPRAWKA 2: MMD / ENERGY ---
            # Najpierw uczymy na danych referencyjnych (stateful detectors)
            mmd.fit(X=X_ref)
            energy_dist.fit(X=X_ref)

            # Pobieramy wynik porównania
            # Frouros zwraca tuple (result, logs). Interesuje nas result.
            mmd_raw, _ = mmd.compare(X=X_curr)
            energy_raw, _ = energy_dist.compare(X=X_curr)

            # Obsługa typu ndarray/obiekt
            # Jeśli wynik ma atrybut 'distance', użyj go. Jeśli nie, to SAM jest dystansem.
            mmd_val = mmd_raw.distance if hasattr(mmd_raw, "distance") else mmd_raw
            energy_val = (
                energy_raw.distance if hasattr(energy_raw, "distance") else energy_raw
            )

            # Konwersja z tablicy 0-wymiarowej numpy do float (jeśli trzeba)
            if isinstance(mmd_val, np.ndarray):
                mmd_val = mmd_val.item()
            if isinstance(energy_val, np.ndarray):
                energy_val = energy_val.item()

            # Progi detekcji
            if mmd_val > 0.02:
                drift_reasons.append(f"MMD (dist={mmd_val:.4f})")

            if energy_val > 0.05:
                drift_reasons.append(f"Energy (dist={energy_val:.4f})")

            if drift_reasons:
                print(f"⚠️  [BATCH DRIFT] Wykryto dryf: {', '.join(drift_reasons)}")
                print(f"    -> Tekst: {text[:60]}...")

        if step % 1000 == 0:
            print(f"Step {step} | {date_str} | Prob: {prob_anomaly:.2f}")

except KeyboardInterrupt:
    print("\nPrzerwano.")
except Exception as e:
    traceback.print_exc()
    print(f"\n❌ BŁĄD: {e}")

print("\nKoniec analizy.")
