"""
Detekcja dryfu danych w embeddingach wysokowymiarowych.

Metodologia zgodna z:
- Rabanser et al. (2019) "Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift"
- Gretton et al. (2012) "A Kernel Two-Sample Test"
- Lipton et al. (2018) "Detecting and Correcting for Label Shift with Black Box Predictors"
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional
from scipy.stats import combine_pvalues
from statsmodels.stats.multitest import multipletests

# --- IMPORTY SOTA DETECTORS ---
from alibi_detect.cd import LSDDDrift
from frouros.detectors.data_drift.batch import MMD, KSTest
from frouros.callbacks.batch import PermutationTestDistanceBased

# =============================================================================
# KONFIGURACJA - Parametry eksperymentu
# =============================================================================

FILE_PATH = "./nlp/gradual/bert/bert/_0.json_embedded.jsonl"
DEFAULT_BATCH_SIZE = 500  # Zwiększony dla stabilniejszych estymat
REF_SIZE = 1000

# --- Kontrola błędu wielokrotnego testowania (FWER) ---
# Zgodnie z Holm (1979) i praktyką w literaturze ML
ALPHA_GLOBAL = 0.05  # Globalny poziom istotności
N_TESTS = 3  # Liczba niezależnych testów (LSDD, KS, MMD)
ALPHA_PER_TEST = ALPHA_GLOBAL / N_TESTS  # ~0.0167 (korekta Bonferroniego)

# --- Reprodukowalność ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# --- Parametry testów permutacyjnych ---
# Minimum 500 dla p-value ~0.01, zalecane 1000+ dla większej precyzji
N_PERMUTATIONS_LSDD = 500
N_PERMUTATIONS_MMD = 500

# --- Parametry redukcji wymiarowości ---
N_PCA_COMPONENTS = 32
UAE_LATENT_DIM = 64


# =============================================================================
# MODEL UAE (Untrained AutoEncoder)
# =============================================================================


class UAE(nn.Module):
    """
    Untrained AutoEncoder do redukcji wymiarowości.

    Zgodnie z Rabanser et al. (2019), losowe projekcje mogą być skuteczne
    dla detekcji dryfu. Kluczowe jest ustawienie seeda dla reprodukowalności.

    Args:
        input_dim: Wymiar wejściowy (domyślnie 384 dla sentence-transformers)
        latent_dim: Wymiar przestrzeni ukrytej
        seed: Ziarno dla reprodukowalności inicjalizacji wag
    """

    def __init__(
        self,
        input_dim: int = 384,
        latent_dim: int = UAE_LATENT_DIM,
        seed: int = RANDOM_SEED,
    ):
        super(UAE, self).__init__()

        # Deterministyczna inicjalizacja wag
        torch.manual_seed(seed)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, latent_dim)
        )

        # Zamrożenie wag - to jest UNTRAINED autoencoder
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# =============================================================================
# GŁÓWNA KLASA DETEKTORA
# =============================================================================


class ScientificDriftDetector:
    """
    Detektor dryfu danych zgodny z najlepszymi praktykami naukowymi.

    Implementuje trzy komplementarne metody:
    1. LSDD (Least-Squares Density Difference) - detekcja zmian gęstości
    2. BBSD: PCA + KS z korektą Bonferroniego - zgodnie z Rabanser et al. (2019)
    3. UAE + MMD - detekcja zmian w przestrzeni ukrytej

    Wyniki są agregowane metodą Fishera (combined p-value).

    Attributes:
        alpha_per_test: Próg istotności dla pojedynczego testu
        alpha_global: Globalny próg istotności dla testu łączonego
    """

    def __init__(
        self,
        x_ref: np.ndarray,
        alpha_global: float = ALPHA_GLOBAL,
        alpha_per_test: float = ALPHA_PER_TEST,
        n_pca_components: int = N_PCA_COMPONENTS,
        random_seed: int = RANDOM_SEED,
    ):
        """
        Inicjalizuje detektor z danymi referencyjnymi.

        Args:
            x_ref: Dane referencyjne (n_samples, n_features)
            alpha_global: Globalny poziom istotności
            alpha_per_test: Poziom istotności dla pojedynczego testu
            n_pca_components: Liczba składowych PCA
            random_seed: Ziarno dla reprodukowalności
        """
        self.alpha_per_test = alpha_per_test
        self.alpha_global = alpha_global
        self.n_pca_components = min(n_pca_components, x_ref.shape[1] // 10)
        self.random_seed = random_seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Próg Bonferroniego dla BBSD (Rabanser et al., 2019)
        # α_bonf = α / k, gdzie k = liczba testów (składowych PCA)
        self.alpha_bonferroni = alpha_per_test / self.n_pca_components

        print(f"\n{'='*60}")
        print("INICJALIZACJA DETEKTORA DRYFU")
        print(f"{'='*60}")
        print(f"  Rozmiar referencji: {len(x_ref)} próbek")
        print(f"  Wymiar danych: {x_ref.shape[1]}")
        print(f"  α globalne: {alpha_global}")
        print(f"  α per test: {alpha_per_test:.4f}")
        print(f"  Liczba składowych PCA: {self.n_pca_components}")
        print(f"  α Bonferroni (BBSD): {self.alpha_bonferroni:.6f}")
        print(f"  Seed: {random_seed}")
        print(f"  Device: {self.device}")
        print(f"{'='*60}")

        self._init_lsdd(x_ref)
        self._init_pca_ks(x_ref)
        self._init_uae_mmd(x_ref)

        print(f"\n[✓] Inicjalizacja zakończona pomyślnie.\n")

    def _init_lsdd(self, x_ref: np.ndarray) -> None:
        """Inicjalizuje detektor LSDD."""
        print(" -> [1/3] Inicjalizacja LSDD (Density-based)...")

        self.lsdd = LSDDDrift(
            x_ref=x_ref,
            backend="pytorch",
            p_val=self.alpha_per_test,
            n_permutations=N_PERMUTATIONS_LSDD,
        )

    def _init_pca_ks(self, x_ref: np.ndarray) -> None:
        """Inicjalizuje PCA + wielowymiarowy test KS z korektą Bonferroniego."""
        print(" -> [2/3] Inicjalizacja PCA + KS (BBSD z korektą Bonferroniego)...")

        self.pca = PCA(
            n_components=self.n_pca_components, random_state=self.random_seed
        )
        x_ref_pca = self.pca.fit_transform(x_ref)

        # Wyjaśniona wariancja - informacja diagnostyczna
        explained_var = np.sum(self.pca.explained_variance_ratio_) * 100
        print(
            f"    PCA: {self.n_pca_components} składowych wyjaśnia {explained_var:.1f}% wariancji"
        )

        # Osobny detektor KS dla każdej składowej
        self.ks_detectors: List[KSTest] = []
        for i in range(self.n_pca_components):
            det = KSTest()
            det.fit(X=x_ref_pca[:, i])
            self.ks_detectors.append(det)

    def _init_uae_mmd(self, x_ref: np.ndarray) -> None:
        """Inicjalizuje UAE + MMD."""
        print(" -> [3/3] Inicjalizacja UAE + MMD (Deep Kernel)...")

        # UAE z deterministyczną inicjalizacją
        self.uae = UAE(
            input_dim=x_ref.shape[1], latent_dim=UAE_LATENT_DIM, seed=self.random_seed
        ).to(self.device)
        self.uae.eval()

        # Transformacja referencji przez UAE
        x_ref_tensor = torch.tensor(x_ref, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            self.x_ref_uae = self.uae(x_ref_tensor).cpu().numpy()

        # MMD z testem permutacyjnym
        self.mmd = MMD(
            callbacks=[
                PermutationTestDistanceBased(
                    num_permutations=N_PERMUTATIONS_MMD,
                    random_state=self.random_seed,
                    name="perm_test",
                )
            ]
        )
        self.mmd.fit(X=self.x_ref_uae)

    def detect(self, x_batch: np.ndarray) -> Dict:
        """
        Uruchamia wszystkie testy detekcji dryfu.

        Args:
            x_batch: Batch danych do testowania (n_samples, n_features)

        Returns:
            Słownik z wynikami wszystkich testów oraz testem łączonym
        """
        results = {}
        p_values = []

        # --- 1. LSDD ---
        p_lsdd = self._run_lsdd(x_batch)
        results["LSDD"] = {
            "drift": p_lsdd < self.alpha_per_test,
            "p_val": p_lsdd,
            "threshold": self.alpha_per_test,
        }
        p_values.append(p_lsdd)

        # --- 2. BBSD: PCA + KS z korektą Bonferroniego (Rabanser et al., 2019) ---
        ks_result = self._run_pca_ks_bonferroni(x_batch)
        results["BBSD_KS"] = ks_result
        # Używamy skorygowanego p-value dla Fisher's method
        p_values.append(ks_result["p_val_corrected"])

        # --- 3. UAE + MMD ---
        p_mmd = self._run_uae_mmd(x_batch)
        results["UAE_MMD"] = {
            "drift": p_mmd < self.alpha_per_test,
            "p_val": p_mmd,
            "threshold": self.alpha_per_test,
        }
        p_values.append(p_mmd)

        # --- 4. Meta-analiza: Fisher's Combined Probability Test ---
        _, combined_p = combine_pvalues(p_values, method="fisher")
        results["COMBINED_FISHER"] = {
            "drift": combined_p < self.alpha_global,
            "p_val": combined_p,
            "threshold": self.alpha_global,
            "individual_p_values": p_values.copy(),
        }

        # --- 5. Podsumowanie ---
        n_drifts = sum(
            1 for key in ["LSDD", "BBSD_KS", "UAE_MMD"] if results[key]["drift"]
        )
        results["SUMMARY"] = {
            "n_detectors_triggered": n_drifts,
            "consensus_drift": n_drifts == 3,
            "majority_drift": n_drifts >= 2,
            "fisher_drift": results["COMBINED_FISHER"]["drift"],
        }

        return results

    def _run_lsdd(self, x_batch: np.ndarray) -> float:
        """Uruchamia test LSDD."""
        try:
            lsdd_res = self.lsdd.predict(x_batch)
            return float(lsdd_res["data"]["p_val"])
        except Exception as e:
            print(f"    [WARN] LSDD error: {e}")
            return 1.0

    def _run_pca_ks_bonferroni(self, x_batch: np.ndarray) -> Dict:
        """
        BBSD: PCA + Multiple Univariate KS Tests z korektą Bonferroniego.

        Zgodnie z Rabanser et al. (2019) "Failing Loudly":
        - Redukcja wymiarowości przez PCA
        - Test KS dla każdej składowej głównej
        - Agregacja przez korektę Bonferroniego

        Procedura Bonferroniego:
        1. Wykonaj k testów, uzyskaj p-values: p_1, p_2, ..., p_k
        2. Skorygowany próg: α_bonf = α / k
        3. Odrzuć H0 jeśli JAKIEKOLWIEK p_i < α_bonf

        Alternatywnie (równoważne):
        - Skorygowane p-value: p_corrected = min(k * min(p_i), 1.0)
        - Odrzuć H0 jeśli p_corrected < α
        """
        x_batch_pca = self.pca.transform(x_batch)

        # Zbierz p-values ze wszystkich składowych
        raw_p_values = []
        for i in range(self.n_pca_components):
            test_res, _ = self.ks_detectors[i].compare(X=x_batch_pca[:, i])
            raw_p_values.append(test_res.p_value)

        # Korekta Bonferroniego
        min_p = min(raw_p_values)

        # Metoda 1: Porównanie z skorygowanym progiem (oryginalna z paper)
        drift_detected = min_p < self.alpha_bonferroni

        # Metoda 2: Skorygowane p-value (dla raportowania)
        # p_corrected = min(k * p_min, 1.0) - to jest równoważne
        p_corrected = min(self.n_pca_components * min_p, 1.0)

        # Liczba składowych, które indywidualnie odrzuciły H0
        n_rejected = sum(1 for p in raw_p_values if p < self.alpha_bonferroni)

        return {
            "drift": drift_detected,
            "p_val_min_raw": min_p,
            "p_val_corrected": p_corrected,  # Bonferroni-corrected p-value
            "n_rejected_components": n_rejected,
            "threshold_bonferroni": self.alpha_bonferroni,
            "threshold": self.alpha_per_test,
        }

    def _run_uae_mmd(self, x_batch: np.ndarray) -> float:
        """Uruchamia test MMD w przestrzeni UAE."""
        x_batch_tensor = torch.tensor(x_batch, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            x_batch_uae = self.uae(x_batch_tensor).cpu().numpy()

        _, callbacks_result = self.mmd.compare(X=x_batch_uae)
        return float(callbacks_result["perm_test"]["p_value"])


# =============================================================================
# GENERATOR DANYCH
# =============================================================================


def read_json_in_batches(
    file_path: str, batch_size: int, embedding_dim: int = 384
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Generator czytający embeddingi z pliku JSONL w batchach.

    Args:
        file_path: Ścieżka do pliku JSONL
        batch_size: Rozmiar batcha
        embedding_dim: Oczekiwany wymiar embeddingów

    Yields:
        Tuple[np.ndarray, List[Dict]]: Batch embeddingów i metadane
    """
    current_embeddings = []
    current_metadata = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                emb = data.get("embedding")

                if not emb or len(emb) != embedding_dim:
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

    # Ostatni niepełny batch
    if current_embeddings:
        yield np.array(current_embeddings, dtype=np.float32), current_metadata


# =============================================================================
# FUNKCJE POMOCNICZE DO WYŚWIETLANIA
# =============================================================================


def format_p_value(p: float) -> str:
    """Formatuje p-value z odpowiednią precyzją."""
    if p < 0.0001:
        return f"{p:.2e}"
    elif p < 0.01:
        return f"{p:.4f}"
    else:
        return f"{p:.3f}"


def print_test_result(name: str, result: Dict, indent: str = "   ") -> None:
    """Wyświetla wynik pojedynczego testu."""
    if result["drift"]:
        status = "🔴 DRIFT"
        marker = " <-- !"
    else:
        status = "🟢 OK   "
        marker = ""

    p_str = format_p_value(
        result["p_val"] if "p_val" in result else result.get("p_val_corrected", 1.0)
    )
    threshold = result.get("threshold", ALPHA_PER_TEST)

    print(
        f"{indent}| {name:<12} | {status} | p={p_str:<10} | α={threshold:.4f}{marker}"
    )


def print_batch_header(batch_num: int, meta: List[Dict]) -> None:
    """Wyświetla nagłówek batcha."""
    start_line = meta[0]["line_num"]
    end_line = meta[-1]["line_num"]
    start_date = meta[0]["date"]
    end_date = meta[-1]["date"]

    print(f"\n{'─'*70}")
    print(
        f"BATCH #{batch_num:<3} | Linie: {start_line:>6} - {end_line:<6} | Daty: {start_date} → {end_date}"
    )
    print(f"{'─'*70}")


# =============================================================================
# GŁÓWNA FUNKCJA
# =============================================================================


def main():
    """Główna funkcja uruchamiająca detekcję dryfu."""

    # Parsowanie argumentów
    file_path = sys.argv[1] if len(sys.argv) > 1 else FILE_PATH
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_BATCH_SIZE

    print("\n" + "=" * 70)
    print("DETEKCJA DRYFU DANYCH - ANALIZA STATYSTYCZNA")
    print("=" * 70)
    print(f"Plik: {file_path}")
    print(f"Rozmiar batcha: {batch_size}")
    print(f"Rozmiar referencji: {REF_SIZE}")
    print(f"α globalne (Fisher): {ALPHA_GLOBAL}")
    print(f"α per test: {ALPHA_PER_TEST:.4f}")
    print("=" * 70)

    # Generator danych
    batch_generator = read_json_in_batches(file_path, batch_size)

    # --- FAZA 1: Zbieranie danych referencyjnych ---
    print("\n[FAZA 1] Zbieranie danych referencyjnych...")

    ref_data = []
    ref_metadata = []

    try:
        while len(ref_data) < REF_SIZE:
            batch_emb, batch_meta = next(batch_generator)
            ref_data.extend(batch_emb)
            ref_metadata.extend(batch_meta)
            print(f"   Zebrano: {len(ref_data)}/{REF_SIZE} próbek")

            if len(ref_data) >= REF_SIZE:
                X_ref = np.array(ref_data[:REF_SIZE], dtype=np.float32)
                ref_metadata = ref_metadata[:REF_SIZE]
                break

    except StopIteration:
        print("[ERROR] Za mało danych do utworzenia referencji!")
        return

    print(f"\n[✓] Referencja utworzona:")
    print(f"    Linie: {ref_metadata[0]['line_num']} - {ref_metadata[-1]['line_num']}")
    print(f"    Daty: {ref_metadata[0]['date']} → {ref_metadata[-1]['date']}")

    # --- Inicjalizacja detektora ---
    detector = ScientificDriftDetector(X_ref)

    # --- FAZA 2: Analiza strumienia danych ---
    print("\n" + "=" * 70)
    print("[FAZA 2] ANALIZA STRUMIENIA DANYCH")
    print("=" * 70)

    batch_num = 0
    drift_history = []

    for X_batch, meta in batch_generator:
        batch_num += 1

        # Uruchom detekcję
        results = detector.detect(X_batch)
        drift_history.append(results)

        # Wyświetl nagłówek
        print_batch_header(batch_num, meta)

        # Wyświetl wyniki poszczególnych testów
        print("   ┌─────────────────────────────────────────────────────────────┐")
        print("   │                    WYNIKI TESTÓW                            │")
        print("   ├──────────────┬──────────┬────────────┬─────────────────────┤")
        print_test_result("LSDD", results["LSDD"])
        print_test_result("BBSD_KS", results["BBSD_KS"])
        print_test_result("UAE_MMD", results["UAE_MMD"])
        print("   ├──────────────┴──────────┴────────────┴─────────────────────┤")

        # Fisher's combined test
        fisher = results["COMBINED_FISHER"]
        fisher_status = "🔴 DRIFT" if fisher["drift"] else "🟢 OK   "
        print(
            f"   │ FISHER COMB. | {fisher_status} | p={format_p_value(fisher['p_val']):<10} | α={fisher['threshold']:.4f}"
        )
        print("   └─────────────────────────────────────────────────────────────┘")

        # Podsumowanie
        summary = results["SUMMARY"]
        n_triggered = summary["n_detectors_triggered"]

        if summary["consensus_drift"]:
            # Pełny konsensus (3/3)
            print(
                f"\n   ⛔ KRYTYCZNY ALARM: Pełny konsensus detektorów ({n_triggered}/3)"
            )
            print(f"   Fisher combined p-value: {format_p_value(fisher['p_val'])}")
            print("\n" + "=" * 70)
            print("⛔ STOP: Wykryto bezsprzeczny dryf danych!")
            print(f"   Linia końcowa: {meta[-1]['line_num']}")
            print(f"   Data: {meta[-1]['date']}")
            print("   Wszystkie 3 detektory potwierdziły zmianę dystrybucji.")
            print("=" * 70)
            break

        elif summary["fisher_drift"]:
            # Fisher test istotny, ale nie pełny konsensus
            print(
                f"\n   ⚠️  OSTRZEŻENIE: Fisher test istotny ({n_triggered}/3 detektorów)"
            )
            print(f"       Zalecenie: Monitoruj kolejne batche")

        elif summary["majority_drift"]:
            # Większość (2/3) bez istotności Fishera
            print(
                f"\n   ⚠️  UWAGA: Większość detektorów ({n_triggered}/3) sygnalizuje dryf"
            )
            print(
                f"       Fisher p-value: {format_p_value(fisher['p_val'])} (nieistotne)"
            )

        elif n_triggered > 0:
            # Pojedynczy detektor
            print(f"\n   ℹ️  INFO: {n_triggered}/3 detektorów sygnalizuje anomalię")

    # --- Podsumowanie końcowe ---
    print("\n" + "=" * 70)
    print("PODSUMOWANIE ANALIZY")
    print("=" * 70)
    print(f"Przeanalizowano batchy: {batch_num}")

    if drift_history:
        fisher_drifts = sum(1 for r in drift_history if r["COMBINED_FISHER"]["drift"])
        consensus_drifts = sum(
            1 for r in drift_history if r["SUMMARY"]["consensus_drift"]
        )

        print(f"Wykryto dryf (Fisher): {fisher_drifts} razy")
        print(f"Pełny konsensus (3/3): {consensus_drifts} razy")

        # Statystyki p-values
        fisher_p_vals = [r["COMBINED_FISHER"]["p_val"] for r in drift_history]
        print(f"\nStatystyki Fisher p-value:")
        print(f"   Min: {min(fisher_p_vals):.6f}")
        print(f"   Max: {max(fisher_p_vals):.6f}")
        print(f"   Mean: {np.mean(fisher_p_vals):.6f}")
        print(f"   Median: {np.median(fisher_p_vals):.6f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
