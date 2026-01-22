import sys
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from typing import Generator, Tuple, Dict, List

# --- IMPORTY SOTA DETECTORS ---
from alibi_detect.cd import LSDDDrift
from frouros.detectors.data_drift.batch import MMD, KSTest
from frouros.callbacks.batch import PermutationTestDistanceBased

# --- KONFIGURACJA ---
# FILE_PATH = "./yelp_sudden_embed/yelp_sudden_embed.json"
FILE_PATH = "./nlp/gradual/bert/bert/_0.json_embedded.jsonl"
DEFAULT_BATCH_SIZE = 300
REF_SIZE = 1000

# Parametry Czułości
# Utrzymujemy rygorystyczne p-value, bo nie mamy już "Sędziego" do weryfikacji.
# To zapewnia, że alarmy będą statystycznie istotne.
STATISTICAL_P_VAL = 0.005


# --- 1. MODEL UAE ---
class UAE(nn.Module):
    def __init__(self, input_dim=384, latent_dim=64):
        super(UAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, latent_dim)
        )
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.encoder(x)


# --- 2. GŁÓWNA KLASA DETEKTORA (BEZ SĘDZIEGO) ---
class PureStatisticalDetector:
    def __init__(self, x_ref: np.ndarray, p_val=STATISTICAL_P_VAL):
        self.p_val = p_val
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(
            f"\n[INIT] Inicjalizacja algorytmów (Ref: {len(x_ref)}, p-val: {p_val})..."
        )

        # --- A. LSDD ---
        print(" -> [1/3] LSDD (Density)...")
        self.lsdd = LSDDDrift(
            x_ref=x_ref, backend="pytorch", p_val=p_val, n_permutations=50
        )

        # --- B. BBSD (PCA + KS) ---
        print(" -> [2/3] PCA + KS (BBSD)...")
        self.n_pca_components = 32
        self.pca = PCA(n_components=self.n_pca_components)
        x_ref_pca = self.pca.fit_transform(x_ref)
        self.alpha_bonferroni = p_val / self.n_pca_components

        self.ks_detectors: List[KSTest] = []
        for i in range(self.n_pca_components):
            det = KSTest()
            det.fit(X=x_ref_pca[:, i])
            self.ks_detectors.append(det)

        # --- C. UAE + MMD ---
        print(" -> [3/3] UAE + MMD (Deep Kernel)...")
        self.uae = UAE(input_dim=x_ref.shape[1], latent_dim=64).to(self.device)
        self.uae.eval()

        x_ref_tensor = torch.tensor(x_ref, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            self.x_ref_uae = self.uae(x_ref_tensor).cpu().numpy()

        # Zostawiamy 500 permutacji dla wysokiej precyzji p-value
        self.mmd = MMD(
            callbacks=[
                PermutationTestDistanceBased(
                    num_permutations=500,
                    random_state=42,
                    name="perm_test",
                )
            ]
        )
        self.mmd.fit(X=self.x_ref_uae)
        print(" -> Gotowe.")

    def run_statistical_tests(self, x_batch: np.ndarray) -> Dict:
        """Uruchamia detektory i zwraca wyniki."""
        results = {}

        # 1. LSDD
        try:
            lsdd_res = self.lsdd.predict(x_batch)
            results["LSDD"] = {
                "drift": bool(lsdd_res["data"]["is_drift"]),
                "p_val": lsdd_res["data"]["p_val"],
            }
        except:
            results["LSDD"] = {"drift": False, "p_val": 1.0}

        # 2. PCA + KS
        x_batch_pca = self.pca.transform(x_batch)
        p_values = []
        for i in range(self.n_pca_components):
            test_res, _ = self.ks_detectors[i].compare(X=x_batch_pca[:, i])
            p_values.append(test_res.p_value)

        min_p = min(p_values) if p_values else 1.0
        results["KS_Bonf"] = {
            "drift": min_p < self.alpha_bonferroni,
            "p_val": min_p,
            "threshold": self.alpha_bonferroni,
        }

        # 3. UAE + MMD
        x_batch_tensor = torch.tensor(x_batch, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            x_batch_uae = self.uae(x_batch_tensor).cpu().numpy()

        _, cb = self.mmd.compare(X=x_batch_uae)
        p_mmd = cb["perm_test"]["p_value"]
        results["UAE_MMD"] = {"drift": p_mmd < self.p_val, "p_val": p_mmd}

        return results


# --- 3. GENERATOR DANYCH ---
def read_json_in_batches(file_path: str, batch_size: int):
    current_embeddings = []
    current_metadata = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                emb = data.get("embedding")
                if not emb or len(emb) != 384:
                    continue
                current_embeddings.append(emb)
                current_metadata.append(
                    {"date": data.get("date", "N/A"), "line_num": line_number}
                )
                if len(current_embeddings) == batch_size:
                    yield np.array(
                        current_embeddings, dtype=np.float32
                    ), current_metadata
                    current_embeddings = []
                    current_metadata = []
            except json.JSONDecodeError:
                continue
    if current_embeddings:
        yield np.array(current_embeddings, dtype=np.float32), current_metadata


def main():
    file_path = sys.argv[1] if len(sys.argv) > 1 else FILE_PATH
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_BATCH_SIZE

    print(f"File: {file_path} | Batch Size: {batch_size}")
    batch_generator = read_json_in_batches(file_path, batch_size)

    # --- FAZA 1: REFERENCJA ---
    ref_data = []
    print("\n[STEP 1] Zbieranie referencji...")
    try:
        while len(ref_data) < REF_SIZE:
            batch_emb, _ = next(batch_generator)
            ref_data.extend(batch_emb)
            if len(ref_data) >= REF_SIZE:
                X_ref = np.array(ref_data[:REF_SIZE], dtype=np.float32)
                break
    except StopIteration:
        print("ERROR: Za mało danych na referencję.")
        return

    # Inicjalizacja Detektora
    detector = PureStatisticalDetector(X_ref)

    # --- FAZA 2: ANALIZA STRUMIENIA ---
    print(f"\n[STEP 2] Analiza strumienia (Batch: {batch_size})...")
    batch_cnt = 0

    for X_batch, meta in batch_generator:
        batch_cnt += 1

        # Uruchomienie detektorów
        stats = detector.run_statistical_tests(X_batch)

        # Sprawdzamy, kto zgłosił alarm
        triggered_detectors = []
        if stats["LSDD"]["drift"]:
            triggered_detectors.append("LSDD")
        if stats["UAE_MMD"]["drift"]:
            triggered_detectors.append("UAE+MMD")
        if stats["KS_Bonf"]["drift"]:
            triggered_detectors.append("KS+Bonf")

        # Formatowanie nagłówka
        start_line = meta[0]["line_num"]
        end_line = meta[-1]["line_num"]
        date_str = meta[-1]["date"]

        print(
            f"Batch #{batch_cnt:<3} | Linie: {start_line}-{end_line} | Data: {date_str}"
        )
        print("   --- Wyniki Algorytmów ---")

        def print_stat_row(name, res):
            if res["drift"]:
                status = "🔴 DRIFT"
                p_info = f"(p: {res['p_val']:.6f}) <--- !!"
            else:
                status = "🟢 OK   "
                p_info = f"(p: {res['p_val']:.4f})"
            print(f"   | {name:<10} | {status} | {p_info}")

        print_stat_row("LSDD", stats["LSDD"])
        print_stat_row("UAE+MMD", stats["UAE_MMD"])
        print_stat_row("KS+Bonf", stats["KS_Bonf"])

        # --- LOGIKA STOPU: WYMAGANY KONSENSUS (3/3) ---

        drift_count = len(triggered_detectors)

        if drift_count == 3:
            # WSZYSTKIE 3 DETEKTORY KRZYCZĄ -> STOP
            print(f"\n   [!!!] KRYTYCZNY ALARM: Pełny konsensus detektorów!")
            print(f"   Zgłosili się: {', '.join(triggered_detectors)}")

            print("\n" + "=" * 60)
            print(f"⛔ STOP: Wykryto bezsprzeczny dryf w linii {end_line}.")
            print(
                "   Zatrzymuję przetwarzanie, ponieważ 3/3 algorytmy potwierdziły zmianę."
            )
            print("=" * 60)
            break

        elif drift_count > 0:
            # TYLKO 1 LUB 2 DETEKTORY -> OSTRZEŻENIE, ALE LECIMY DALEJ
            print(f"\n   [!] Ostrzeżenie: Wykryto sygnały dryfu ({drift_count}/3).")
            print(f"       Detektory: {', '.join(triggered_detectors)}")
            print("       >>> Kontynuuję analizę (wymagane 3/3 do zatrzymania)...")

        else:
            # 0 DETEKTORÓW -> CISZA
            pass

        print("-" * 60)


if __name__ == "__main__":
    main()
