import os
import json
import heapq
import tempfile
import pandas as pd
import random

INPUT_PATH = ""
BUSINESS_PATH = ""
OUTPUT_PATH = ""  
CATEGORY_A = "Restaurants"
CATEGORY_B = "Beauty & Spas"
dataset_size = 20_000
DRIFT_TYPE = "gradual"  # sudden, gradual or sudden_by_year

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Yelp Drift Dataset Generator")

    # Input review.json
    parser.add_argument(
        "--ip", "--input_path",
        dest="input_path",
        default=INPUT_PATH,
        type=str,
        required=True,
        help="Path to review.json (Yelp reviews)"
    )

    # Input business.json
    parser.add_argument(
        "--bp", "--business_path",
        dest="business_path",
        default=BUSINESS_PATH,
        type=str,
        required=True,
        help="Path to business.json (Yelp business info)"
    )

    # Output dataset
    parser.add_argument(
        "--op", "--output_path",
        dest="output_path",
        default=OUTPUT_PATH,
        type=str,
        required=True,
        help="Output path for drift dataset (JSONL)"
    )

    # Categories A & B
    parser.add_argument(
        "--ca", "--category_a",
        dest="category_a",
        type=str,
        default="Restaurants",
        help="Category A for drift generation"
    )

    parser.add_argument(
        "--cb", "--category_b",
        dest="category_b",
        type=str,
        default="Beauty & Spas",
        help="Category B for drift generation"
    )

    # Dataset size per class / segment
    parser.add_argument(
        "--ds", "--dataset_size",
        dest="dataset_size",
        type=int,
        default=20_000,
        help="Total samples needed for A and B (per segment for gradual)"
    )

    # Drift type
    parser.add_argument(
        "--dt", "--drift_type",
        dest="drift_type",
        type=str,
        default="gradual",
        choices=["sudden", "gradual", "sudden_by_year"],
        help="Type of drift: sudden, gradual, or sudden_by_year"
    )

    return parser.parse_args()



class YelpDataPreparer:
    def __init__(self, review_path: str, business_path: str):
        self.file_index = 100_000
        self.prev_file_index = 0
        self.review_path = review_path
        self.business_path = business_path
        self.df_reviews = self.load_reviews(0, self.file_index)
        print(f"Wczytano {len(self.df_reviews)} recenzji")
        self.df_business = self.load_business()
        print(f"Wczytano {len(self.df_business)} biznesów")
        self.df_merged = self.merge_reviews_with_business()
        print(f"Połączono do {len(self.df_merged)} rekordów")

    def load_reviews(self, start : int, end : int):
        """
        Wczytuje plik review.json do DataFrame.
        Konwertuje kolumnę 'date' na datetime.
        """
        batch = []
        with open(self.review_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < start:
                    continue
                if i >= end:
                    break
                batch.append(json.loads(line))
        df = pd.DataFrame(batch)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def load_business(self, nrows=None):
        """
        Wczytuje plik business.json do DataFrame.
        Zwraca tylko kolumny business_id, categories, city.
        """
        df = pd.read_json(self.business_path, lines=True, nrows=nrows)
        self.df_business = df[["business_id", "categories", "city"]]
        return self.df_business
    
    def merge_reviews_with_business(self):
        """
        Łączy review.json z business.json po business_id.
        Wymaga wcześniejszego uruchomienia load_reviews() i load_business().
        """

        if self.df_reviews is None:
            raise ValueError("No review data loaded")

        if self.df_business is None:
            raise ValueError("No business data loaded")

        business_cols = self.df_business[["business_id", "categories", "city"]]

        self.df_merged = self.df_reviews.merge(
            business_cols,
            on="business_id",
            how="left"
        )

        return self.df_merged

    def filter_by_year_and_category(self, year: int, category: str):
        """
        Zwraca rekordy, które:
        - mają w polu 'date' podany rok,
        - mają w 'categories' daną kategorię (case-insensitive).

        Wymaga wcześniejszego wykonania merge_reviews_with_business().
        """

        if self.df_merged is None:
            raise ValueError("Najpierw wywołaj merge_reviews_with_business()")

        df = self.df_merged

        mask_year = df["date"].dt.year == year

        mask_cat = df["categories"].fillna("").str.contains(category, case=False)

        result = df[mask_year & mask_cat].copy()

        return result
    
    def make_json_serializable(self, rec: dict):
        out = {}
        for k, v in rec.items():
            if isinstance(v, pd.Timestamp):
                out[k] = v.isoformat()  # <-- najważniejsze
            else:
                out[k] = v
        return out

    def filter_by_category(self, category: str):
        """
        Zwraca rekordy, które:
        - mają w polu 'date' podany rok,
        - mają w 'categories' daną kategorię (case-insensitive).

        Wymaga wcześniejszego wykonania merge_reviews_with_business().
        """

        if self.df_merged is None:
            raise ValueError("Najpierw wywołaj merge_reviews_with_business()")

        df = self.df_merged


        mask_cat = df["categories"].fillna("").str.contains(category, case=False)

        result = df[mask_cat].copy()

        return result
    
    def create_sudden_drift_data(self, dataset_size=dataset_size, output_path=OUTPUT_PATH):
        """
        Tworzy dane typu sudden drift:
        najpierw dataset_size próbek CATEGORY_A,
        potem dataset_size próbek CATEGORY_B.

        Dane są zapisywane jako JSONL do output_path.
        """

        collected_A = []
        collected_B = []

        batch_size = 100_000 

        print("=== Tworzenie sudden drift dataset ===")

        while self.file_index < 30_000_000:

            print(f"\nWczytywanie batchu od {self.file_index} do {self.file_index + batch_size}")

            
            self.df_reviews = self.load_reviews(self.file_index, self.file_index + batch_size)
            self.file_index += batch_size

            print(f"Wczytano {len(self.df_reviews)} recenzji")

           
            if self.df_business is None:
                self.df_business = self.load_business()
                print(f"Wczytano {len(self.df_business)} biznesów")

            self.df_merged = self.merge_reviews_with_business()
            print(f"Po scaleniu: {len(self.df_merged)} rekordów")

            A_part = self.filter_by_category(CATEGORY_A)
            B_part = self.filter_by_category(CATEGORY_B)
            if "date" in A_part.columns:
                A_part = A_part.copy()
                A_part["date"] = A_part["date"].astype(str)

            if "date" in B_part.columns:
                B_part = B_part.copy()
                B_part["date"] = B_part["date"].astype(str)

            collected_A.extend(A_part.to_dict("records"))
            collected_B.extend(B_part.to_dict("records"))

            print(f"Zebrano A: {len(collected_A)}, B: {len(collected_B)}")

            if len(collected_A) >= dataset_size and len(collected_B) >= dataset_size:
                break

        print("\n=== Kompletowanie sudden drift stream... ===")

        final_A = collected_A[:dataset_size]
        final_B = collected_B[:dataset_size]

        with open(output_path, "w", encoding="utf-8") as f:
            for rec in final_A:
                rec = self.make_json_serializable(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n",)
            for rec in final_B:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"\nZapisano sudden drift dataset → {output_path}")
        print(f"Rozmiar: {len(final_A)} A + {len(final_B)} B = {len(final_A) + len(final_B)} rekordów")

    def create_sudden_drift_data_by_year(self, year_a: int, year_b: int, dataset_size=dataset_size, output_path=OUTPUT_PATH):
        """
        Tworzy dane typu sudden drift:
        najpierw dataset_size próbek CATEGORY_A,
        potem dataset_size próbek CATEGORY_B.

        Dane są zapisywane jako JSONL do output_path.
        """

        collected_A = []
        collected_B = []

        batch_size = 100_000  # ile linii wczytywać per batch

        print("=== Tworzenie sudden drift dataset ===")

        while self.file_index < 30_000_000:

            print(f"\nWczytywanie batchu od {self.file_index} do {self.file_index + batch_size}")

            
            self.df_reviews = self.load_reviews(self.file_index, self.file_index + batch_size)
            self.file_index += batch_size

            print(f"Wczytano {len(self.df_reviews)} recenzji")

           
            if self.df_business is None:
                self.df_business = self.load_business()
                print(f"Wczytano {len(self.df_business)} biznesów")

            self.df_merged = self.merge_reviews_with_business()
            print(f"Po scaleniu: {len(self.df_merged)} rekordów")

            A_part = self.filter_by_year_and_category(year_a, CATEGORY_A)
            B_part = self.filter_by_year_and_category(year_b, CATEGORY_A)
            if "date" in A_part.columns:
                A_part = A_part.copy()
                A_part["date"] = A_part["date"].astype(str)

            if "date" in B_part.columns:
                B_part = B_part.copy()
                B_part["date"] = B_part["date"].astype(str)

            collected_A.extend(A_part.to_dict("records"))
            collected_B.extend(B_part.to_dict("records"))

            print(f"Zebrano A: {len(collected_A)}, B: {len(collected_B)}")

            if len(collected_A) >= dataset_size and len(collected_B) >= dataset_size:
                break

        print("\n=== Kompletowanie sudden drift stream... ===")

        final_A = collected_A[:dataset_size]
        final_B = collected_B[:dataset_size]

        with open(output_path, "w", encoding="utf-8") as f:
            for rec in final_A:
                rec = self.make_json_serializable(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n",)
            for rec in final_B:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"\nZapisano sudden drift dataset → {output_path}")
        print(f"Rozmiar: {len(final_A)} A + {len(final_B)} B = {len(final_A) + len(final_B)} rekordów")
        
    def create_gradual_drift_data(self, dataset_size_per_segment=dataset_size, output_path=OUTPUT_PATH):
        """
        Tworzy dane typu gradual drift:
        segment 1: 100% CATEGORY_A
        segment 2: 80% A / 20% B
        segment 3: 60% A / 40% B
        ...
        segment N: 0% A / 100% B

        Każdy segment ma dataset_size_per_segment próbek.
        Dane są zapisywane jako JSONL do output_path.
        """

        collected_A = []
        collected_B = []

        batch_size = 100_000 
        self.file_index = 0

        print("=== Tworzenie gradual drift dataset ===")
        ratios = [
            (1.0, 0.0),
            (0.8, 0.2),
            (0.6, 0.4),
            (0.4, 0.6),
            (0.2, 0.8),
            (0.0, 1.0),
        ]

        max_A_needed = int(sum(r[0] for r in ratios) * dataset_size_per_segment)
        max_B_needed = int(sum(r[1] for r in ratios) * dataset_size_per_segment)

        print(f"Docelowo potrzebne A: {max_A_needed}, B: {max_B_needed}")

        while self.file_index < 30_000_000:
            print(f"\nWczytywanie batchu od {self.file_index} do {self.file_index + batch_size}")

            self.df_reviews = self.load_reviews(self.file_index, self.file_index + batch_size)
            self.file_index += batch_size

            print(f"Wczytano {len(self.df_reviews)} recenzji")

            if self.df_business is None:
                self.df_business = self.load_business()
                print(f"Wczytano {len(self.df_business)} biznesów")

            self.df_merged = self.merge_reviews_with_business()
            print(f"Po scaleniu: {len(self.df_merged)} rekordów")

            A_part = self.filter_by_category(CATEGORY_A)
            B_part = self.filter_by_category(CATEGORY_B)

            if "date" in A_part.columns:
                A_part = A_part.copy()
                A_part["date"] = A_part["date"].astype(str)

            if "date" in B_part.columns:
                B_part = B_part.copy()
                B_part["date"] = B_part["date"].astype(str)

            collected_A.extend(A_part.to_dict("records"))
            collected_B.extend(B_part.to_dict("records"))

            print(f"Zebrano A: {len(collected_A)}, B: {len(collected_B)}")

            if len(collected_A) >= max_A_needed and len(collected_B) >= max_B_needed:
                print("Mamy wystarczająco danych z obu klas.")
                break

        if len(collected_A) < max_A_needed or len(collected_B) < max_B_needed:
            print("UWAGA: nie udało się zebrać wystarczającej liczby próbek A/B.")
            print(f"Mamy A={len(collected_A)} / B={len(collected_B)}, potrzebne A={max_A_needed}, B={max_B_needed}")

        print("\n=== Kompletowanie gradual drift stream... ===")

        idx_A = 0
        idx_B = 0
        stream = []

        for i, (ratio_A, ratio_B) in enumerate(ratios):
            n_A = int(round(dataset_size_per_segment * ratio_A))
            n_B = dataset_size_per_segment - n_A

            seg_A = collected_A[idx_A: idx_A + n_A]
            seg_B = collected_B[idx_B: idx_B + n_B]

            idx_A += n_A
            idx_B += n_B

            segment = seg_A + seg_B
            random.shuffle(segment) 

            print(f"Segment {i}: ratio A={ratio_A}, B={ratio_B} -> A={len(seg_A)}, B={len(seg_B)}, razem={len(segment)}")

            stream.extend(segment)

        print(f"\nŁączna liczba próbek w strumieniu: {len(stream)}")

        with open(output_path, "w", encoding="utf-8") as f:
            for rec in stream:
                rec = self.make_json_serializable(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"\nZapisano gradual drift dataset → {output_path}")
        print(f"Liczba segmentów: {len(ratios)}, rozmiar segmentu: {dataset_size_per_segment}, razem: {len(stream)} rekordów")
        
if __name__ == "__main__":
    args = parse_args()
    INPUT_PATH = args.input_path
    BUSINESS_PATH = args.business_path
    OUTPUT_PATH = args.output_path
    CATEGORY_A = args.category_a
    CATEGORY_B = args.category_b
    dataset_size = args.dataset_size
    DRIFT_TYPE = args.drift_type
    yd = YelpDataPreparer(INPUT_PATH, BUSINESS_PATH)
    if DRIFT_TYPE == "sudden":
        yd.create_sudden_drift_data()
    elif DRIFT_TYPE == "sudden_by_year":
        yd.create_sudden_drift_data_by_year(2010, 2021)
    elif DRIFT_TYPE == "gradual":
        yd.create_gradual_drift_data()