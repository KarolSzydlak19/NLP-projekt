import sys
import json
import math
import argparse
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from alibi_detect.cd import LSDDDrift

try:
    from frouros.detectors.data_drift import MMD, KSTest
except Exception:
    from frouros.detectors.data_drift.batch import MMD, KSTest

from frouros.callbacks.batch import PermutationTestDistanceBased

try:
    from scipy.stats import combine_pvalues

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# DEFAULT CONFIGURATION

DEFAULT_FILE_PATH = "./nlp/gradual/bert/bert/_8.json_embedded.jsonl"
DEFAULT_BATCH_SIZE = 50
DEFAULT_REF_SIZE = 1000

ALPHA_GLOBAL = 0.05
N_DETECTORS = 3  # LSDD, MKS, UAE(MMD)
ALPHA_PER_TEST = ALPHA_GLOBAL / N_DETECTORS

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

N_PERMUTATIONS_LSDD = 500
N_PERMUTATIONS_MMD = 500

N_PCA_COMPONENTS = 32
UAE_LATENT_DIM = 64


# UAE (Untrained AutoEncoder)


class UAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# META-TEST: ACAT (Cauchy Combination Test)


def combine_pvalues_acat(
    p_values: List[float], weights: Optional[List[float]] = None
) -> float:
    """
    ACAT: Aggregated Cauchy Association Test / Cauchy Combination Test.
    """
    if not p_values:
        return 1.0
    m = len(p_values)
    if weights is None:
        weights = [1.0 / m] * m
    else:
        s = sum(weights)
        weights = [w / s for w in weights]

    eps = 1e-15
    T = 0.0
    for p, w in zip(p_values, weights):
        p = float(p)
        p = min(max(p, eps), 1.0 - eps)
        T += w * math.tan((0.5 - p) * math.pi)

    p_comb = 0.5 - math.atan(T) / math.pi
    return float(min(max(p_comb, 0.0), 1.0))


# DETECTOR CLASS


@dataclass
class DetectorConfig:
    alpha_global: float = ALPHA_GLOBAL
    alpha_per_test: float = ALPHA_PER_TEST
    n_pca_components: int = N_PCA_COMPONENTS
    uae_latent_dim: int = UAE_LATENT_DIM
    n_perm_lsdd: int = N_PERMUTATIONS_LSDD
    n_perm_mmd: int = N_PERMUTATIONS_MMD
    random_seed: int = RANDOM_SEED
    meta_method: str = "acat"  # "acat" or "fisher"


class ScientificDriftDetector:
    def __init__(self, x_ref: np.ndarray, cfg: DetectorConfig):
        self.cfg = cfg
        self.alpha_global = cfg.alpha_global
        self.alpha_per_test = cfg.alpha_per_test
        self.random_seed = cfg.random_seed

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n_samples, n_features = x_ref.shape

        max_components = max(
            1, min(cfg.n_pca_components, n_features, max(1, n_samples - 1))
        )
        self.n_pca_components = max_components

        # Bonferroni correction (per component)
        self.alpha_bonferroni = self.alpha_per_test / self.n_pca_components

        print("\n" + "=" * 70)
        print("INICJALIZACJA DETEKTORA DRYFU")
        print("=" * 70)
        print(f"  Referencja: {n_samples} probek")
        print(f"  Wymiar: {n_features}")
        print(f"  alfa globalne (meta): {self.alpha_global}")
        print(f"  alfa per detektor:   {self.alpha_per_test:.6f}")
        print(f"  PCA components:   {self.n_pca_components}")
        print(f"  alfa Bonferroni BBSD:{self.alpha_bonferroni:.8f}")
        print(f"  Meta method:      {cfg.meta_method}")
        print(f"  Device:           {self.device}")
        print("=" * 70)

        self._init_lsdd(x_ref)
        self._init_pca_ks(x_ref)
        self._init_uae_mmd(x_ref)

        print("\nnicjalizacja zakończona.\n")

    def _init_lsdd(self, x_ref: np.ndarray) -> None:
        print(" -> [1/3] LSDD (Alibi Detect)")
        self.lsdd = LSDDDrift(
            x_ref=x_ref,
            backend="pytorch",
            p_val=self.alpha_per_test,
            n_permutations=self.cfg.n_perm_lsdd,
        )

    def _init_pca_ks(self, x_ref: np.ndarray) -> None:
        print(" -> [2/3] PCA + MKS (Bonferroni)")

        self.pca = PCA(
            n_components=self.n_pca_components, random_state=self.random_seed
        )
        x_ref_pca = self.pca.fit_transform(x_ref)

        explained = float(np.sum(self.pca.explained_variance_ratio_) * 100.0)
        print(
            f"    PCA: {self.n_pca_components} składowych, {explained:.1f}% wariancji"
        )

        self.ks_detectors: List[KSTest] = []
        for i in range(self.n_pca_components):
            det = KSTest()
            _ = det.fit(X=x_ref_pca[:, i])
            self.ks_detectors.append(det)

    def _init_uae_mmd(self, x_ref: np.ndarray) -> None:
        print(" -> [3/3] UAE + MMD")

        self.uae = UAE(
            input_dim=x_ref.shape[1],
            latent_dim=self.cfg.uae_latent_dim,
            seed=self.random_seed,
        ).to(self.device)
        self.uae.eval()

        x_ref_t = torch.tensor(x_ref, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            x_ref_uae = self.uae(x_ref_t).cpu().numpy()
        self.x_ref_uae = x_ref_uae

        self.mmd = MMD(
            callbacks=[
                PermutationTestDistanceBased(
                    num_permutations=self.cfg.n_perm_mmd,
                    random_state=self.random_seed,
                    name="permutation_test",
                )
            ]
        )
        _ = self.mmd.fit(X=self.x_ref_uae)

    def detect(self, x_batch: np.ndarray) -> Dict:
        results: Dict[str, Dict] = {}
        p_values_for_meta: List[float] = []

        # LSDD
        p_lsdd = self._run_lsdd(x_batch)
        results["LSDD"] = {
            "drift": p_lsdd < self.alpha_per_test,
            "p_val": p_lsdd,
            "threshold": self.alpha_per_test,
        }
        p_values_for_meta.append(p_lsdd)

        # PCA + KS + Bonferroni
        bbsd = self._run_pca_ks_bonferroni(x_batch)
        results["BBSD_KS"] = bbsd
        # do meta bierzemy p po Bonferronim
        p_values_for_meta.append(bbsd["p_val_corrected"])

        # UAE + MMD
        p_mmd = self._run_uae_mmd(x_batch)
        results["UAE_MMD"] = {
            "drift": p_mmd < self.alpha_per_test,
            "p_val": p_mmd,
            "threshold": self.alpha_per_test,
        }
        p_values_for_meta.append(p_mmd)

        # Meta test
        method = self.cfg.meta_method.lower().strip()
        if method == "fisher":
            if not _HAS_SCIPY:
                raise RuntimeError(
                    "Wybrano meta=fisher, ale brak SciPy (combine_pvalues)."
                )
            _, combined_p = combine_pvalues(p_values_for_meta, method="fisher")  # type: ignore
        else:
            combined_p = combine_pvalues_acat(p_values_for_meta)

        results["COMBINED_META"] = {
            "method": method,
            "drift": float(combined_p) < self.alpha_global,  # type: ignore
            "p_val": float(combined_p),  # type: ignore
            "threshold": self.alpha_global,
            "individual_p_values": p_values_for_meta.copy(),
        }

        n_drifts = sum(
            1 for key in ["LSDD", "BBSD_KS", "UAE_MMD"] if results[key]["drift"]
        )
        results["SUMMARY"] = {
            "n_detectors_triggered": n_drifts,
            "consensus_drift": n_drifts == 3,
            "majority_drift": n_drifts >= 2,
            "meta_drift": results["COMBINED_META"]["drift"],
        }
        return results

    def _run_lsdd(self, x_batch: np.ndarray) -> float:
        try:
            res = self.lsdd.predict(x_batch, return_p_val=True)
            p = res["data"].get("p_val", 1.0)  # type: ignore
            return float(np.asarray(p).reshape(-1)[0])
        except Exception as e:
            print(f"    [WARN] LSDD error: {e}")
            return 1.0

    def _run_pca_ks_bonferroni(self, x_batch: np.ndarray) -> Dict:
        x_batch_pca = self.pca.transform(x_batch)

        raw_p = []
        for i in range(self.n_pca_components):
            stat_arr, _ = self.ks_detectors[i].compare(X=x_batch_pca[:, i])
            stat = stat_arr[0] if isinstance(stat_arr, np.ndarray) else stat_arr
            raw_p.append(float(stat.p_value))

        min_p = float(np.min(raw_p))
        drift = min_p < self.alpha_bonferroni
        p_corrected = min(self.n_pca_components * min_p, 1.0)
        n_rejected = int(np.sum(np.array(raw_p) < self.alpha_bonferroni))

        return {
            "drift": drift,
            "p_val_min_raw": min_p,
            "p_val_corrected": float(p_corrected),
            "n_rejected_components": n_rejected,
            "threshold_bonferroni": self.alpha_bonferroni,
            "threshold": self.alpha_per_test,
        }

    def _run_uae_mmd(self, x_batch: np.ndarray) -> float:
        try:
            x_t = torch.tensor(x_batch, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                x_uae = self.uae(x_t).cpu().numpy()
            _, logs = self.mmd.compare(X=x_uae)
            return float(logs["permutation_test"]["p_value"])
        except Exception as e:
            print(f"    [WARN] MMD error: {e}")
            return 1.0


def read_json_in_batches(
    file_path: str,
    batch_size: int,
    embedding_dim: Optional[int] = None,
) -> Tuple[np.ndarray, List[Dict]]:  # type: ignore
    current_emb: List[np.ndarray] = []
    current_meta: List[Dict] = []

    inferred_dim = embedding_dim

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            emb = data.get("embedding")
            if not emb:
                continue

            if inferred_dim is None:
                inferred_dim = len(emb)

            if len(emb) != inferred_dim:
                continue

            arr = np.asarray(emb, dtype=np.float32)
            if not np.isfinite(arr).all():
                continue

            current_emb.append(arr)
            current_meta.append({"date": data.get("date", "N/A"), "line_num": line_num})

            if len(current_emb) == batch_size:
                yield np.vstack(current_emb).astype(np.float32), current_meta  # type: ignore
                current_emb, current_meta = [], []

    if current_emb:
        yield np.vstack(current_emb).astype(np.float32), current_meta  # type: ignore


def format_p_value(p: float) -> str:
    if p < 1e-4:
        return f"{p:.2e}"
    if p < 1e-2:
        return f"{p:.4f}"
    return f"{p:.3f}"


def print_batch_header(batch_num: int, meta: List[Dict], tested: bool) -> None:
    start_line, end_line = meta[0]["line_num"], meta[-1]["line_num"]
    start_date, end_date = meta[0]["date"], meta[-1]["date"]
    flag = "TEST" if tested else "SKIP"
    print(f"\n{'─'*74}")
    print(
        f"BATCH #{batch_num:<4} [{flag}] | Linie: {start_line:>7} - {end_line:<7} | Daty: {start_date} → {end_date}"
    )
    print(f"{'─'*74}")


def print_test_row(
    name: str, drift: bool, p: float, alpha: float, extra: str = ""
) -> None:
    status = "  DRIFT" if drift else "  OK   "
    mark = " <-- !" if drift else ""
    print(
        f"   | {name:<12} | {status} | p={format_p_value(p):<10} | α={alpha:.6f}{mark} {extra}"
    )


def print_results(results: Dict) -> None:
    print("   ┌──────────────────────────────────────────────────────────────────┐")
    print("   │                           WYNIKI                                  │")
    print("   ├──────────────┬──────────┬────────────┬────────────────────────────┤")

    lsdd = results["LSDD"]
    print_test_row("LSDD", lsdd["drift"], lsdd["p_val"], lsdd["threshold"])

    bbsd = results["BBSD_KS"]
    extra = f"(rej={bbsd['n_rejected_components']}, pmin={format_p_value(bbsd['p_val_min_raw'])})"
    print_test_row(
        "BBSD_KS",
        bbsd["drift"],
        bbsd["p_val_corrected"],
        bbsd["threshold_bonferroni"],
        extra=extra,
    )

    mmd = results["UAE_MMD"]
    print_test_row("UAE_MMD", mmd["drift"], mmd["p_val"], mmd["threshold"])

    meta = results["COMBINED_META"]
    print("   ├──────────────┴──────────┴────────────┴────────────────────────────┤")
    print_test_row(
        f"META({meta['method']})", meta["drift"], meta["p_val"], meta["threshold"]
    )
    print("   └──────────────────────────────────────────────────────────────────┘")


CSV_HEADER = [
    "batch_num",
    "tested",
    "start_line",
    "end_line",
    "start_date",
    "end_date",
    "lsdd_p",
    "lsdd_drift",
    "bbsd_pmin",
    "bbsd_pcorr",
    "bbsd_drift",
    "mmd_p",
    "mmd_drift",
    "meta_method",
    "meta_p",
    "meta_drift",
    "n_detectors_triggered",
    "consensus_drift",
    "majority_drift",
    "meta_hits_in_row",
    "stop_triggered",
]


def init_csv(path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)


def append_csv(path: str, row: Dict) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([row.get(k, "") for k in CSV_HEADER])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=DEFAULT_FILE_PATH)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--ref_size", type=int, default=DEFAULT_REF_SIZE)
    parser.add_argument("--embedding_dim", type=int, default=None)  # None = auto
    parser.add_argument("--alpha", type=float, default=ALPHA_GLOBAL)
    parser.add_argument("--meta", type=str, default="acat", choices=["acat", "fisher"])
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Ile kolejnych TESTOWANYCH batchy z meta-drift zanim przerwać (>=1).",
    )
    parser.add_argument(
        "--test_every",
        type=int,
        default=1,
        help="Testuj co k-ty batch (>=1). Np. 2 = test co drugi batch.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="drift_results.csv",
        help="Ścieżka do pliku CSV z logiem wyników.",
    )
    args = parser.parse_args()

    alpha_global = float(args.alpha)
    alpha_per_test = alpha_global / N_DETECTORS

    test_every = max(1, int(args.test_every))
    patience = max(1, int(args.patience))

    print("\n" + "=" * 72)
    print("DETEKCJA DRYFU (embeddingi) — LSDD + BBSD(KS) + UAE(MMD) + META")
    print("=" * 72)
    print(f"Plik:        {args.file}")
    print(f"Batch size:  {args.batch_size}")
    print(f"Ref size:    {args.ref_size}")
    print(f"Alpha meta:  {alpha_global}")
    print(f"Alpha/det:   {alpha_per_test:.6f}")
    print(f"Meta method: {args.meta}")
    print(f"Patience:    {patience}")
    print(f"Test every:  {test_every}")
    print(f"CSV out:     {args.out_csv}")
    print("=" * 72)

    init_csv(args.out_csv)

    gen = read_json_in_batches(
        args.file, args.batch_size, embedding_dim=args.embedding_dim
    )

    # referencje
    print("\n[FAZA 1] Buduję referencję...")
    ref_emb: List[np.ndarray] = []
    ref_meta: List[Dict] = []

    try:
        while len(ref_emb) < args.ref_size:
            X, meta = next(gen)  # type: ignore
            ref_emb.extend(list(X))
            ref_meta.extend(meta)
            print(f"  Zebrano: {len(ref_emb)}/{args.ref_size}")
    except StopIteration:
        print("[ERROR] Za mało danych do utworzenia referencji.")
        sys.exit(1)

    X_ref = np.asarray(ref_emb[: args.ref_size], dtype=np.float32)
    ref_meta = ref_meta[: args.ref_size]
    print(
        f"[✓] Referencja gotowa | Linie {ref_meta[0]['line_num']}–{ref_meta[-1]['line_num']} | Daty {ref_meta[0]['date']}→{ref_meta[-1]['date']}"
    )

    cfg = DetectorConfig(
        alpha_global=alpha_global,
        alpha_per_test=alpha_per_test,
        meta_method=args.meta,
    )
    detector = ScientificDriftDetector(X_ref, cfg)

    # Analiza streamu
    print("\n[FAZA 2] Analiza strumienia...")
    batch_num = 0
    meta_hits_in_row = 0
    stopped = False

    for X_batch, meta in gen:
        batch_num += 1
        tested = batch_num % test_every == 0

        start_line, end_line = meta[0]["line_num"], meta[-1]["line_num"]
        start_date, end_date = meta[0]["date"], meta[-1]["date"]

        row_base = {
            "batch_num": batch_num,
            "tested": int(tested),
            "start_line": start_line,
            "end_line": end_line,
            "start_date": start_date,
            "end_date": end_date,
            "meta_hits_in_row": meta_hits_in_row,
            "stop_triggered": 0,
        }

        if not tested:
            print_batch_header(batch_num, meta, tested=False)  # type: ignore
            append_csv(args.out_csv, row_base)
            continue

        results = detector.detect(X_batch)  # type: ignore

        print_batch_header(batch_num, meta, tested=True)  # type: ignore
        print_results(results)

        # update streak
        if results["COMBINED_META"]["drift"]:
            meta_hits_in_row += 1
        else:
            meta_hits_in_row = 0

        stop_now = meta_hits_in_row >= patience
        if stop_now:
            stopped = True

        # log row
        lsdd = results["LSDD"]
        bbsd = results["BBSD_KS"]
        mmd = results["UAE_MMD"]
        meta_res = results["COMBINED_META"]
        summ = results["SUMMARY"]

        row = {
            **row_base,
            "lsdd_p": lsdd["p_val"],
            "lsdd_drift": int(lsdd["drift"]),
            "bbsd_pmin": bbsd["p_val_min_raw"],
            "bbsd_pcorr": bbsd["p_val_corrected"],
            "bbsd_drift": int(bbsd["drift"]),
            "mmd_p": mmd["p_val"],
            "mmd_drift": int(mmd["drift"]),
            "meta_method": meta_res["method"],
            "meta_p": meta_res["p_val"],
            "meta_drift": int(meta_res["drift"]),
            "n_detectors_triggered": summ["n_detectors_triggered"],
            "consensus_drift": int(summ["consensus_drift"]),
            "majority_drift": int(summ["majority_drift"]),
            "meta_hits_in_row": meta_hits_in_row,
            "stop_triggered": int(stop_now),
        }
        append_csv(args.out_csv, row)

        if stop_now:
            print("\n" + "=" * 72)
            print("⛔ STOP: Meta-test wykrył dryf (spełniona reguła zatrzymania).")
            print(f"   Batch: {batch_num}")
            print(f"   Linia końcowa: {end_line}")
            print(f"   Data: {end_date}")
            print(
                f"   Meta p-value: {format_p_value(meta_res['p_val'])}  (α={meta_res['threshold']})"
            )
            print(
                f"   Streak (meta): {meta_hits_in_row}/{patience} (test_every={test_every})"
            )
            print("=" * 72)
            break

    print("\n" + "=" * 72)
    print("KONIEC")
    print(f"Przeanalizowano batchy: {batch_num}")
    print(f"Zatrzymano: {'TAK' if stopped else 'NIE'}")
    print(f"CSV: {args.out_csv}")
    print("=" * 72)


if __name__ == "__main__":
    main()
