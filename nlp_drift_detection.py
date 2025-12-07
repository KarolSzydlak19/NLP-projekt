import sys
import re
from pathlib import Path

import json
import math
import numpy as np
import traceback
from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from typing import Any

from frouros.detectors.concept_drift import ADWIN, ADWINConfig
from frouros.detectors.data_drift.batch.distance_based import MMD, EnergyDistance
from scipy.stats import ks_2samp

# KONFIGURACJA
FILE_PATH = (
    "./yelp_academic_dataset_review_embed/yelp_academic_dataset_review_embed.json"
)
ANOMALY_MODEL_NAME = "svm"

WARMUP_TRAIN = 200  # Próbki do trenowania modelu
WARMUP_REF = 200  # Próbki do zbudowania dystrybucji odniesienia (nie użyte w treningu!)
WARMUP_TOTAL = WARMUP_TRAIN + WARMUP_REF

TEST_WINDOW_SIZE = 300
CHECK_INTERVAL = 50
ADWIN_DELTA = 0.002


def is_valid_path(path_string: str) -> bool:
    invalid_chars = r'[<>:"|?*]'
    if re.search(invalid_chars, path_string):
        return False
    try:
        path = Path(path_string)
        return path.exists()
    except Exception:
        return False


def init():
    global FILE_PATH
    print(len(sys.argv))
    if len(sys.argv) > 1:
        is_valid = is_valid_path(sys.argv[1])
        if not is_valid:
            print(f"Invalid file path: {sys.argv[1]}")
            sys.exit(1)
            return
        FILE_PATH = sys.argv[1]
    else:
        print(
            "Usage: python nlp_drift_detection.py <path_to_json_file> <anomaly_model>"
        )
        # sys.exit(1)
        # return

    global ANOMALY_MODEL_NAME
    if len(sys.argv) > 2:
        model_name = sys.argv[2].lower()
        if model_name in ["svm", "oneclasssvm"]:
            ANOMALY_MODEL_NAME = "svm"
        elif model_name in ["forest", "iforest", "isolationforest"]:
            ANOMALY_MODEL_NAME = "isolationforest"
    else:
        ANOMALY_MODEL_NAME = "svm"


def sigmoid(x: float) -> float:
    if x < -500:
        return 0.0
    if x > 500:
        return 1.0
    return 1 / (1 + math.exp(-x))


def process_stream(file_path: str):
    print(f"⏳ Otwieranie strumienia: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                yield (
                    np.array(data["embedding"], dtype=np.float64),  # Wymuszamy float64
                    data.get("date", "N/A"),
                    data.get("text", ""),
                )
            except json.JSONDecodeError:
                continue


def main():
    scaler = StandardScaler()

    adwin_config = ADWINConfig(delta=ADWIN_DELTA)
    adwin = ADWIN(config=adwin_config)

    # energy_dist = EnergyDistance()
    # mmd = MMD()

    # Bufory
    step = 0
    warmup_buffer = []  # Tu zbieramy wszystko na start

    # Dane referencyjne (po warmupie)
    X_ref_scaled = None
    scores_ref = None

    # Dane bieżące
    current_window_emb = deque(maxlen=TEST_WINDOW_SIZE)
    current_window_scores = deque(maxlen=TEST_WINDOW_SIZE)

    model_trained = False

    # MODEL BAZOWY
    model = None
    if ANOMALY_MODEL_NAME == "svm":
        model = OneClassSVM(nu=0.1, kernel="rbf", gamma="scale")
    elif ANOMALY_MODEL_NAME == "isolationforest":
        model = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42,
            n_jobs=-1,  # Używa wszystkich rdzeni CPU - będzie szybciej!
        )
    else:
        print(f"Error, unknown model: {ANOMALY_MODEL_NAME}")
        sys.exit(1)
    print(f"🚀 Start detekcji. Model: {ANOMALY_MODEL_NAME.upper()}")
    print(
        f"   Warmup Total: {WARMUP_TOTAL} (Train: {WARMUP_TRAIN} | Ref: {WARMUP_REF})"
    )

    try:
        for embedding, date_str, text in process_stream(str(FILE_PATH)):
            step += 1

            # ==========================================
            # FAZA 1: WARMUP (Zbieranie i Podział)
            # ==========================================
            if not model_trained:
                warmup_buffer.append(embedding)

                if len(warmup_buffer) >= WARMUP_TOTAL:
                    print(f"\n🔧 Warmup zakończony. Dzielenie danych...")

                    # Konwersja na numpy
                    X_all = np.array(warmup_buffer)

                    # PODZIAŁ: Train Set vs Reference Set
                    # Unikamy błędu "Train vs Test Bias" w KS Test
                    X_train = X_all[:WARMUP_TRAIN]
                    X_ref = X_all[WARMUP_TRAIN:]  # Te dane nie widziały modelu w .fit()

                    print(f"   Trening Scalera i Modelu na {len(X_train)} próbkach...")
                    scaler.fit(X_train)
                    X_train_scaled = scaler.transform(X_train)
                    model.fit(X_train_scaled)

                    print(
                        f"   Generowanie Reference Score na {len(X_ref)} próbkach (Hold-out)..."
                    )
                    # Przygotowanie referencji dla detektorów
                    # Używamy ascontiguousarray, aby naprawić błędy pamięci w Frouros
                    X_ref_scaled = np.ascontiguousarray(scaler.transform(X_ref))

                    if ANOMALY_MODEL_NAME == "svm":
                        raw_ref = model.score_samples(X_ref_scaled)
                        scores_ref = [sigmoid(-s) for s in raw_ref]
                    else:
                        raw_ref = model.decision_function(X_ref_scaled)
                        scores_ref = [
                            sigmoid(-s * 10) for s in raw_ref
                        ]  # Skalowanie dla iForest

                    scores_ref = np.array(scores_ref)
                    model_trained = True
                    # Czyścimy bufor, żeby zwolnić pamięć
                    warmup_buffer = []
                    print("✅ System gotowy. Przełączanie w tryb online.\n")
                continue

            # ==========================================
            # FAZA 2: DETEKCJA ONLINE
            # ==========================================

            # 1. Przetwarzanie próbki
            X_sample = embedding.reshape(1, -1)
            X_scaled = scaler.transform(X_sample)

            # Scoring
            if ANOMALY_MODEL_NAME == "svm":
                raw = model.score_samples(X_scaled)[0]
                prob = sigmoid(-raw)
            else:
                raw = model.decision_function(X_scaled)[0]
                prob = sigmoid(-raw * 10)

            # Aktualizacja okien
            current_window_emb.append(embedding)
            current_window_scores.append(prob)

            # 2. ADWIN (Szybki dryf koncepcji)
            adwin.update(value=prob)
            if adwin.drift:
                print(f"🚨 [ADWIN] Drift w kroku {step}! (Prob: {prob:.4f})")
                adwin.reset()

            # 3. TESTY STATYSTYCZNE (Batch)
            if (
                len(current_window_emb) == TEST_WINDOW_SIZE
                and step % CHECK_INTERVAL == 0
            ):

                # Przygotowanie danych bieżących
                X_curr = np.array(current_window_emb)
                X_curr_scaled = np.ascontiguousarray(scaler.transform(X_curr))
                scores_curr = np.array(current_window_scores)

                drift_reasons = []

                # A. KS Test (Porównanie rozkładu wyników modelu)
                # Teraz porównujemy Ref (Out-of-sample) vs Curr (Out-of-sample)
                # To powinno dać realne p-value, a nie 1e-20.
                ks_res = ks_2samp(scores_ref, scores_curr)
                p_val = ks_res.pvalue if hasattr(ks_res, "pvalue") else ks_res[1]  # type: ignore

                # Bardzo niski próg dla KS
                if p_val < 0.001:  # type: ignore
                    drift_reasons.append(f"KS (p={p_val:.2e})")

                # B. MMD & Energy (Porównanie surowych embeddingów)
                # Tworzymy NOWE instancje detektorów, aby uniknąć błędów stanu/wymiarów
                try:
                    # MMD
                    detector_mmd = MMD()
                    detector_mmd.fit(X=X_ref_scaled)  # type: ignore
                    res_mmd = detector_mmd.compare(X=X_curr_scaled)[0]
                    dist_mmd = (
                        res_mmd.distance if hasattr(res_mmd, "distance") else res_mmd  # type: ignore
                    )
                    if dist_mmd > 0.025:  # Lekko podniesiony próg
                        drift_reasons.append(f"MMD (dist={dist_mmd:.4f})")

                    # Energy
                    detector_energy = EnergyDistance()
                    detector_energy.fit(X=X_ref_scaled)  # type: ignore
                    res_energy = detector_energy.compare(X=X_curr_scaled)[0]
                    dist_energy = (
                        res_energy.distance  # type: ignore
                        if hasattr(res_energy, "distance")
                        else res_energy
                    )
                    if dist_energy > 0.05:
                        drift_reasons.append(f"Energy (dist={dist_energy:.4f})")

                except Exception as e:
                    # Wyłapujemy błędy Frouros, ale nie przerywamy skryptu
                    print(f"⚠️  Błąd Frouros (wymiary/pamięć): {e}")

                if drift_reasons:
                    print(
                        f"⚠️  [BATCH DRIFT {step}] Wykryto: {', '.join(drift_reasons)}"
                    )
                    print(f"    -> Tekst: {text[:60]}...")

            if step % 1000 == 0:
                print(f"Step {step} | {date_str} | Prob: {prob:.2f}")

    except KeyboardInterrupt:
        print("\nPrzerwano.")
    except Exception:
        traceback.print_exc()


# Usage: python nlp_drift_detection.py <parth_to_json_file> <anomaly_model>
if __name__ == "__main__":
    init()
    main()
