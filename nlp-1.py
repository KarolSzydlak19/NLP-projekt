import json
import math
import numpy as np
from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from frouros.detectors.concept_drift import DDM, DDMConfig

# KONFIGURACJA
FILE_PATH = (
    "./yelp_academic_dataset_review_embed/yelp_academic_dataset_review_embed.json"
)
CONTEXT_WINDOW = 30
WARMUP_SIZE = 200  # Liczba próbek do wstępnego nauczenia modelu
DRIFT_THRESHOLD = (
    0.5  # Próg prawdopodobieństwa, powyżej którego uznajemy punkt za anomalię
)


def sigmoid(x):
    """
    Mapuje wynik (-inf, +inf) na prawdopodobieństwo (0.0, 1.0).
    """
    if x < -500:
        return 0.0
    if x > 500:
        return 1.0
    return 1 / (1 + math.exp(-x))


def process_stream_features_only(file_path):
    """
    Generator czytający plik linia po linii (zakłada posortowane dane).
    """
    print(f"⏳ Otwieranie strumienia danych: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                # Pobieramy dane, obsługując ewentualne braki pól
                date_str = data.get("date", "N/A")
                text = data.get("text", "")
                stars = data.get("stars", "N/A")
                embedding = np.array(data["embedding"])

                yield embedding, date_str, text, stars
            except json.JSONDecodeError:
                continue


# 1. Model anomalii (SVM + Scaler)
scaler = StandardScaler()
# kernel="rbf" zazwyczaj lepiej mapuje nieliniowe embeddingi niż domyślny
anomaly_model = OneClassSVM(nu=0.1, kernel="rbf", gamma="scale")

# 2. Detektor driftu (Frouros DDM)
# Parametry: warning (ostrzeżenie), drift (alarm), min_instances (czas na stabilizację)
ddm_config = DDMConfig(warning_level=2.0, drift_level=3.0, min_num_instances=30)
drift_detector = DDM(config=ddm_config)

print("🚀 Rozpoczynam detekcję VIRTUAL DRIFT (Frouros DDM)")

# Zmienne stanu
step = 0
drifts_detected = 0
recent_reviews = deque(maxlen=CONTEXT_WINDOW)
warmup_data = []
model_trained = False

try:
    for embedding, date_str, text, stars in process_stream_features_only(FILE_PATH):
        step += 1

        # FAZA 1: WARMUP (Zbieranie danych do treningu SVM)
        if not model_trained:
            warmup_data.append(embedding)
            if len(warmup_data) >= WARMUP_SIZE:
                print(
                    f"\n🔧 Zakończono buforowanie. Trenuję model bazowy na {WARMUP_SIZE} próbkach..."
                )
                X_warmup = np.array(warmup_data)

                # Fit scalera i modelu
                scaler.fit(X_warmup)
                X_scaled = scaler.transform(X_warmup)
                anomaly_model.fit(X_scaled)

                model_trained = True
                print("✅ Model wytrenowany. Przełączanie w tryb detekcji online.\n")
            continue

        # FAZA 2: DETEKCJA ONLINE

        # A. Skalowanie pojedynczej próbki
        X_sample = embedding.reshape(1, -1)
        X_scaled = scaler.transform(X_sample)

        # B. Wynik surowy (Signed Distance) z SVM
        # W sklearn: wartości DODATNIE (+) = Wnętrze (Norma), UJEMNE (-) = Anomalia
        raw_score = anomaly_model.score_samples(X_scaled)[0]

        # C. Normalizacja i Inwersja
        # Chcemy, aby anomalia (ujemny raw_score) dawała wysokie prawdopodobieństwo (bliskie 1.0).
        # Dlatego używamy -raw_score.
        # Przykład: raw_score = -5 (anomalia) -> -(-5) = 5 -> sigmoid(5) ~= 0.99
        prob_anomaly = sigmoid(-raw_score)

        # D. Binaryzacja dla DDM
        # DDM działa na strumieniu błędów (0/1). Uznajemy > 0.5 za "błąd" (anomalię statystyczną).
        is_anomaly = 1 if prob_anomaly > DRIFT_THRESHOLD else 0

        # E. Aktualizacja detektora
        drift_detector.update(value=is_anomaly)

        # F. Logowanie kontekstu do bufora
        recent_reviews.append(
            {
                "step": step,
                "date": date_str,
                "text": text[:100],  # Skracamy tekst dla oszczędności
                "stars": stars,
                "prob_anomaly": prob_anomaly,
                "is_anomaly": is_anomaly,
            }
        )

        # G. Obsługa wykrytego Driftu
        if drift_detector.drift:
            drifts_detected += 1
            print(f"\n{'='*80}")
            # [Image of sigmoid function]- conceptual trigger
            print(
                f"🚨 [KROK {step} | {date_str}] WYKRYTO CONCEPT DRIFT #{drifts_detected}"
            )
            print(f"   -> Raw SVM Score: {raw_score:.4f} (ujemne wartości to anomalie)")
            print(f"   -> Anomaly Prob:  {prob_anomaly:.4f}")
            print(f"   -> Status:        DRIFT DETECTED")

            # Zapis kontekstu do pliku JSON
            drift_context = {
                "drift_id": drifts_detected,
                "step": step,
                "date": date_str,
                "trigger_prob": prob_anomaly,
                "context": list(recent_reviews),
            }

            fname = f"./logs/drift_{drifts_detected}.json"
            # Upewnij się, że katalog ./logs istnieje, lub zapisz w bieżącym
            try:
                with open(fname, "w", encoding="utf-8") as f:
                    json.dump(drift_context, f, indent=2, ensure_ascii=False)
                print(f"   💾 Zapisano log: {fname}")
            except FileNotFoundError:
                print(
                    f"   ⚠️ Nie można zapisać logu (sprawdź czy folder logs istnieje)."
                )

            # Reset detektora po wykryciu dryfu
            drift_detector.reset()

        # Logowanie postępu co 1000 kroków
        if step % 1000 == 0:
            status_msg = "WARNING" if drift_detector.warning else "OK"
            print(
                f"Krok {step} | {date_str} | Prob: {prob_anomaly:.2f} | DDM: {status_msg}"
            )

except KeyboardInterrupt:
    print("\nPrzerwano przez użytkownika.")
except Exception as e:
    print(f"\n❌ Błąd krytyczny: {e}")

print(f"\n{'='*30}")
print("KONIEC ANALIZY")
print(f"Całkowita liczba wykrytych zmian (Concept Drifts): {drifts_detected}")
