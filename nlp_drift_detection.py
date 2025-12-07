import sys
import re
import json
import math
import numpy as np
import traceback
from collections import deque
from pathlib import Path
from typing import Any

from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

from sklearn.decomposition import PCA

# --- IMPORTY FROUROS (STREAMING) ---
# 1. Concept Drift (ADWIN - klasyk)
from frouros.detectors.concept_drift import ADWIN, ADWINConfig

# 2. Data Drift Streaming - MMD (Multivariate)
from frouros.detectors.data_drift.streaming.distance_based import MMD as MMDStreaming

# 3. Data Drift Streaming - Incremental KS (Univariate)
from frouros.detectors.data_drift.streaming.statistical_test import IncrementalKSTest

# ==========================================
# KONFIGURACJA
# ==========================================
FILE_PATH = "./yelp_sudden_embed/yelp_sudden_embed.json"
ANOMALY_MODEL_NAME = "svm"

MMD_CHECK_INTERVAL = 10  # Uruchamiaj MMD tylko co 10 recenzji (10x przyspieszenie)
PCA_COMPONENTS = 32  # Redukcja embeddingu z 384 do 32 wymiarów dla MMD
STREAM_WINDOW_SIZE = 200  # Mniejsze okno = szybsze MMD

WARMUP_TRAIN = 200  # Próbki do trenowania modelu
WARMUP_REF = 200  # Próbki referencyjne (Baza dla KS i MMD)
WARMUP_TOTAL = WARMUP_TRAIN + WARMUP_REF

# W streamingu to oznacza: "Porównuj ostatnie 300 próbek strumienia z bazą referencyjną"
STREAM_WINDOW_SIZE = 300
ADWIN_DELTA = 0.002
EMBEDDING_DIM = 384  # Wymiar embeddingu


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
                    np.array(data["embedding"], dtype=np.float64),
                    data.get("date", "N/A"),
                    data.get("text", ""),
                )
            except json.JSONDecodeError:
                continue


def main():
    scaler = StandardScaler()
    # PCA tylko dla MMD (nie wpływa na SVM/ADWIN/KS)
    pca_reducer = PCA(n_components=PCA_COMPONENTS)

    # 1. ADWIN (Concept Drift - Zmiana średniej)
    adwin = ADWIN(config=ADWINConfig(delta=ADWIN_DELTA))

    # 2. Incremental KS Test (Data Drift - Zmiana rozkładu wyników 1D)
    # Porównuje histogram referencyjny z oknem przesuwnym ze strumienia
    ks_stream = IncrementalKSTest(window_size=STREAM_WINDOW_SIZE)

    # 3. Streaming MMD (Data Drift - Zmiana rozkładu embeddingów ND)
    # Porównuje kernel distance referencji z oknem przesuwnym
    mmd_stream = MMDStreaming(window_size=STREAM_WINDOW_SIZE)

    # Bufory
    step = 0
    warmup_buffer = []
    model_trained = False

    # Przechowywanie zredukowanych danych ref
    X_ref_pca = None

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
    print(f"   Mode: FULL STREAMING")
    print(f"   Detectors: ADWIN, IncrementalKSTest, StreamingMMD")

    try:
        for embedding, date_str, text in process_stream(str(FILE_PATH)):
            step += 1

            # ==========================================
            # FAZA 1: WARMUP (Zbieranie i Fit)
            # ==========================================
            if not model_trained:
                warmup_buffer.append(embedding)
                if len(warmup_buffer) >= WARMUP_TOTAL:
                    print(
                        f"\n🔧 Warmup zakończony. Przetwarzanie {WARMUP_TOTAL} próbek..."
                    )

                    X_all = np.array(warmup_buffer, dtype=np.float64)
                    X_train = X_all[:WARMUP_TRAIN]
                    X_ref_raw = X_all[WARMUP_TRAIN:]

                    # 1. Trenujemy Model Anomalii (na pełnych wymiarach)
                    print(f"   Trening modelu anomalii...")
                    scaler.fit(X_train)
                    X_train_scaled = scaler.transform(X_train)
                    model.fit(X_train_scaled)

                    # 2. Reference Scores dla KS
                    X_ref_scaled = scaler.transform(X_ref_raw)
                    if ANOMALY_MODEL_NAME == "svm":
                        raw_s = model.score_samples(X_ref_scaled)
                        scores_ref = [sigmoid(-s) for s in raw_s]
                    else:
                        raw_s = model.decision_function(X_ref_scaled)
                        scores_ref = [sigmoid(-s * 10) for s in raw_s]
                    scores_ref = np.array(scores_ref, dtype=np.float64)

                    # 3. FIT PCA I MMD (Optymalizacja)
                    print(f"   Trening PCA (384 -> {PCA_COMPONENTS})...")
                    # Uczymy PCA na danych referencyjnych, by "zrozumiał" normalną strukturę
                    pca_reducer.fit(X_ref_raw)
                    X_ref_pca = pca_reducer.transform(X_ref_raw)

                    # Fit detektorów
                    print("   Fitowanie detektorów strumieniowych...")
                    ks_stream.fit(X=scores_ref)
                    mmd_stream.fit(X=X_ref_pca)  # MMD uczy się na zredukowanych danych!

                    model_trained = True
                    warmup_buffer = []
                    print("✅ System gotowy. Przełączanie w tryb online.\n")
                continue

            # ==========================================
            # FAZA 2: DETEKCJA ONLINE
            # ==========================================

            # A. Scoring
            X_sample = embedding.reshape(1, -1)
            X_sample_scaled = scaler.transform(X_sample)

            if ANOMALY_MODEL_NAME == "svm":
                raw = model.score_samples(X_sample_scaled)[0]
                prob = sigmoid(-raw)
            else:
                raw = model.decision_function(X_sample_scaled)[0]
                prob = sigmoid(-raw * 10)

            # --- DETEKTOR 1: ADWIN (Concept Drift) ---
            adwin.update(value=prob)
            if adwin.drift:
                print(f"🚨 [ADWIN] Drift w kroku {step}! (Zmiana średniej anomalii)")
                print(f"    -> Tekst: {text[:60]}...")
                adwin.reset()

            # --- DETEKTOR 2: Incremental KS (Data Drift 1D) ---
            # Monitoruje rozkład wyników modelu (probability)
            ks_result, _ = ks_stream.update(value=prob)
            if (
                ks_result is not None
                and ks_result.p_value  # pyright: ignore[reportAttributeAccessIssue]
                < 0.001
            ):
                # Pobieramy p-value z obiektu testu
                # Frouros przechowuje statystyki w callbacks lub logs, ale flaga .drift wystarczy
                print(f"📉 [Inc-KS] Drift w kroku {step}! (Zmiana rozkładu wyników)")
                print(f"    -> Tekst: {text[:60]}...")
                ks_stream.reset()
                ks_stream.fit(
                    X=scores_ref  # pyright: ignore[reportPossiblyUnboundVariable]
                )

            # --- DETEKTOR 3: Streaming MMD (Data Drift ND) ---
            # Monitoruje surowe wektory embeddingów
            if step % MMD_CHECK_INTERVAL == 0:
                try:
                    emb_pca = pca_reducer.transform(embedding.reshape(1, -1))
                    # Flatten do 1D (wymóg niektórych wersji streaming MMD)
                    emb_pca_flat = emb_pca.flatten()
                    mmd_result, _ = mmd_stream.update(
                        value=emb_pca_flat  # pyright: ignore[reportArgumentType]
                    )

                    if (
                        mmd_result is not None
                        and mmd_result.distance  # pyright: ignore[reportAttributeAccessIssue]
                        > 0.025
                    ):
                        print(
                            f"🌌 [MMD] Drift w kroku {step}! (Zmiana geometrii embeddingów)"
                        )
                        print(f"    -> Tekst: {text[:60]}...")
                        mmd_stream.reset()
                        # MUSIMY UŻYĆ X_ref_pca (32 dim), NIE X_ref_raw (384 dim)
                        mmd_stream.fit(
                            X=X_ref_pca  # pyright: ignore[reportArgumentType]
                        )
                except Exception as e:
                    # Zabezpieczenie przed rzadkimi błędami algebry liniowej w update
                    print(f"⚠️ MMD Error: {e}")
                    pass

            # Logowanie postępu
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
