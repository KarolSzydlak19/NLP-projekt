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

# ==========================================
# KONFIGURACJA
# ==========================================
FILE_PATH = "./yelp_sudden_embed/yelp_sudden_embed.json"
ANOMALY_MODEL_NAME = "svm"

WARMUP_TRAIN = 200  # Próbki do trenowania modelu
WARMUP_REF = 200  # Próbki referencyjne (Hold-out dla testów statystycznych)
WARMUP_TOTAL = WARMUP_TRAIN + WARMUP_REF

TEST_WINDOW_SIZE = 300
CHECK_INTERVAL = 50
ADWIN_DELTA = 0.002
EMBEDDING_DIM = 384  # Oczekiwany wymiar embeddingów


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

    # Przechowywanie danych referencyjnych
    X_ref_raw = None  # Surowe embeddingi (dla MMD/Energy)
    scores_ref = None  # Wyniki modelu (dla KS)

    # Okna przesuwne (Queue)
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
        return

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
                    print(
                        f"\n🔧 Warmup zakończony. Przetwarzanie {WARMUP_TOTAL} próbek..."
                    )

                    X_all = np.array(warmup_buffer)  # Shape: (400, 384)

                    # Podział na Train i Reference
                    X_train = X_all[:WARMUP_TRAIN]
                    X_ref_raw = X_all[WARMUP_TRAIN:]  # Surowe embeddingi dla MMD/Energy

                    # 1. Trenujemy Scaler i Model na X_train
                    print(f"   Trening modelu na {len(X_train)} próbkach...")
                    scaler.fit(X_train)
                    X_train_scaled = scaler.transform(X_train)
                    model.fit(X_train_scaled)

                    # 2. Generujemy wyniki referencyjne na X_ref (Hold-out)
                    print(
                        f"   Generowanie Reference Score na {len(X_ref_raw)} próbkach..."
                    )
                    X_ref_scaled = scaler.transform(X_ref_raw)

                    if ANOMALY_MODEL_NAME == "svm":
                        raw_scores = model.score_samples(X_ref_scaled)
                        scores_ref = [sigmoid(-s) for s in raw_scores]
                    else:
                        raw_scores = model.decision_function(X_ref_scaled)
                        scores_ref = [sigmoid(-s * 10) for s in raw_scores]

                    scores_ref = np.array(scores_ref)
                    model_trained = True
                    warmup_buffer = []  # Czyścimy RAM
                    print("✅ System gotowy. Przełączanie w tryb online.\n")
                continue

            # ==========================================
            # FAZA 2: DETEKCJA ONLINE
            # ==========================================

            # A. Scoring (Model Anomalii)
            # Reshape jest kluczowy dla pojedynczej próbki (1, 384)
            X_sample = embedding.reshape(1, -1)
            X_sample_scaled = scaler.transform(X_sample)

            if ANOMALY_MODEL_NAME == "svm":
                raw = model.score_samples(X_sample_scaled)[0]
                prob = sigmoid(-raw)
            else:
                raw = model.decision_function(X_sample_scaled)[0]
                prob = sigmoid(-raw * 10)

            # B. Aktualizacja Buforów
            current_window_emb.append(embedding)
            current_window_scores.append(prob)

            # 2. ADWIN (Szybki dryf koncepcji)
            adwin.update(value=prob)
            if adwin.drift:
                print(f"🚨 [ADWIN] Drift w kroku {step}! (Prob: {prob:.4f})")
                adwin.reset()

            # D. Detekcja Batchowa (KS, MMD, Energy)
            if (
                len(current_window_emb) == TEST_WINDOW_SIZE
                and step % CHECK_INTERVAL == 0
            ):

                # Przygotowanie danych bieżących
                # Używamy .copy(), aby upewnić się, że pamięć jest ciągła (C-contiguous)
                X_curr_raw = np.array(list(current_window_emb)).copy()
                scores_curr = np.array(list(current_window_scores))

                # Wymuszenie kształtu 2D (Bezpiecznik dla Frouros)
                if X_curr_raw.ndim == 1:
                    X_curr_raw = X_curr_raw.reshape(-1, EMBEDDING_DIM)
                if X_ref_raw.ndim == 1:  # pyright: ignore[reportOptionalMemberAccess]
                    X_ref_raw = X_ref_raw.reshape(  # pyright: ignore[reportOptionalMemberAccess]
                        -1, EMBEDDING_DIM
                    )

                drift_reasons = []

                # --- 1. KS TEST ---
                # Porównujemy rozkład wyników modelu (Reference vs Current)
                try:
                    ks_res = ks_2samp(scores_ref, scores_curr)
                    p_val = (
                        ks_res.pvalue  # pyright: ignore[reportAttributeAccessIssue]
                        if hasattr(ks_res, "pvalue")
                        else ks_res[1]
                    )
                    # Próg p < 0.001 (0.1%)
                    if p_val < 0.001:  # pyright: ignore[reportOperatorIssue]
                        drift_reasons.append(f"KS (p={p_val:.2e})")
                except Exception as e:
                    print(f"⚠️ Błąd KS: {e}")

                # --- 2. MMD & ENERGY (Na surowych danych) ---
                try:
                    # Debug: Sprawdzenie kształtów przed wywołaniem
                    # print(f"DEBUG: Ref Shape: {X_ref_raw.shape}, Curr Shape: {X_curr_raw.shape}")

                    # Tworzymy nowe instancje (Stateless usage)
                    detector_mmd = MMD()
                    detector_mmd.fit(X=X_ref_raw)  # pyright: ignore[reportArgumentType]
                    res_mmd = detector_mmd.compare(X=X_curr_raw)[0]

                    # Obsługa różnych typów zwracanych
                    val_mmd = (
                        res_mmd.distance  # pyright: ignore[reportAttributeAccessIssue]
                        if hasattr(res_mmd, "distance")
                        else res_mmd
                    )
                    if isinstance(val_mmd, np.ndarray):
                        val_mmd = val_mmd.item()

                    if val_mmd > 0.025:
                        drift_reasons.append(f"MMD (dist={val_mmd:.4f})")

                    # Energy Distance
                    detector_energy = EnergyDistance()
                    detector_energy.fit(
                        X=X_ref_raw  # pyright: ignore[reportArgumentType]
                    )
                    res_energy = detector_energy.compare(X=X_curr_raw)[0]

                    val_energy = (
                        res_energy.distance  # pyright: ignore[reportAttributeAccessIssue]
                        if hasattr(res_energy, "distance")
                        else res_energy
                    )
                    if isinstance(val_energy, np.ndarray):
                        val_energy = val_energy.item()

                    if val_energy > 0.05:
                        drift_reasons.append(f"Energy (dist={val_energy:.4f})")

                except Exception as e:
                    print(f"⚠️ Błąd Frouros (Shape: {X_curr_raw.shape}): {e}")

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
