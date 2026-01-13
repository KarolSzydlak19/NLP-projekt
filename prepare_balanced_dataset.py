import os
import json
import heapq
import tempfile
import pandas as pd
import random

INPUT_PATH = "D:\\nlp\\data\\yelp_academic_dataset_review_sorted.json"
BUSINESS_PATH = "D:\\nlp\\data\\yelp_academic_dataset_business.json"
OUTPUT_PATH = "D:\\nlp\\datasets\\sudden_by_year\\sudden_by_year_drift"  
CATEGORY_A = "Restaurants"
CATEGORY_B = "Beauty & Spas"
DATASET_SIZE = 1000
DRIFT_TYPE = "sudden_by_year"  #sudden, gradual or sudden_by_year

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Yelp Drift Dataset Generator")

    # Input review.json
    parser.add_argument(
        "--ip", "--input_path",
        dest="input_path",
        default=INPUT_PATH,
        type=str,
        required=False,
        help="Path to review.json (Yelp reviews)"
    )

    # Input business.json
    parser.add_argument(
        "--bp", "--business_path",
        dest="business_path",
        default=BUSINESS_PATH,
        type=str,
        required=False,
        help="Path to business.json (Yelp business info)"
    )

    # Output dataset
    parser.add_argument(
        "--op", "--output_path",
        dest="output_path",
        default=OUTPUT_PATH,
        type=str,
        required=False,
        help="Output path for drift dataset (JSONL)"
    )

    # Categories A & B
    parser.add_argument(
        "--ca", "--category_a",
        dest="category_a",
        type=str,
        default=CATEGORY_A,
        help="Category A for drift generation"
    )

    parser.add_argument(
        "--cb", "--category_b",
        dest="category_b",
        type=str,
        default=CATEGORY_B,
        help="Category B for drift generation"
    )

    # Dataset size per class / segment
    parser.add_argument(
        "--ds", "--dataset_size",
        dest="dataset_size",
        type=int,
        default=DATASET_SIZE,
        help="Total samples needed for A and B (per segment for gradual)"
    )

    # Drift type
    parser.add_argument(
        "--dt", "--drift_type",
        dest="drift_type",
        type=str,
        default=DRIFT_TYPE,
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
        print(f"Reviews: {len(self.df_reviews)}")
        self.df_business = self.load_business()
        print(f"Businesses: {len(self.df_business)}")
        self.df_merged = self.merge_reviews_with_business()
        print(f"Records total: {len(self.df_merged)}")

    def load_reviews(self, start : int, end : int):
        #Read review.json into DataFrame
        #Convert 'date' -> datetime

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
        #Read business.json into DataFrame
        #returns columns: business_id, categories, city

        df = pd.read_json(self.business_path, lines=True, nrows=nrows)
        self.df_business = df[["business_id", "categories", "city"]]
        return self.df_business
    
    def merge_reviews_with_business(self):
        #Combine review.json with business.json by business_id

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

        if self.df_merged is None:
            raise ValueError("Missing: merge_reviews_with_business()")

        df = self.df_merged

        mask_year = df["date"].dt.year == year

        mask_cat = df["categories"].fillna("").str.contains(category, case=False)

        result = df[mask_year & mask_cat].copy()

        return result
    
    def make_json_serializable(self, rec: dict):
        out = {}
        for k, v in rec.items():
            if isinstance(v, pd.Timestamp):
                out[k] = v.isoformat()  #najważniejsze
            else:
                out[k] = v
        return out

    def filter_by_category(self, category: str):

        if self.df_merged is None:
            raise ValueError("Missing: merge_reviews_with_business()")

        df = self.df_merged


        mask_cat = df["categories"].fillna("").str.contains(category, case=False)

        result = df[mask_cat].copy()

        return result
    
    def create_sudden_drift_data(self, dataset_size=DATASET_SIZE, output_path=OUTPUT_PATH):
        #Tworzy dane typu sudden drift:
        #najpierw dataset_size próbek CATEGORY_A,
        #potem dataset_size próbek CATEGORY_B.
        #Dane są zapisywane jako JSONL do output_path.

        collected_A = []
        collected_B = []

        batch_size = 100_000 

        print("Sudden drift dataset")

        while self.file_index < 30_000_000:

            print(f"\nBatch {self.file_index} - {self.file_index + batch_size}")

            
            self.df_reviews = self.load_reviews(self.file_index, self.file_index + batch_size)
            self.file_index += batch_size

            print(f"Reviews: {len(self.df_reviews)}")

           
            if self.df_business is None:
                self.df_business = self.load_business()
                print(f"Businesses: {len(self.df_business)}")

            self.df_merged = self.merge_reviews_with_business()
            print(f"Records: {len(self.df_merged)}")

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

            print(f"A: {len(collected_A)}, B: {len(collected_B)}")

            if len(collected_A) >= dataset_size and len(collected_B) >= dataset_size:
                break

        final_A = collected_A[:dataset_size]
        final_B = collected_B[:dataset_size]

        with open(output_path, "w", encoding="utf-8") as f:
            for rec in final_A:
                rec = self.make_json_serializable(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n",)
            for rec in final_B:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"\nOutput path: {output_path}")
        print(f"Records total: {len(final_A)} A + {len(final_B)} B = {len(final_A) + len(final_B)}")

    def create_sudden_drift_data_by_year(self, year_a: int, year_b: int, dataset_size=DATASET_SIZE, output_path=OUTPUT_PATH):
        # Tworzy serię plików sudden drift, wykorzystując tyle danych, ile się da.
        # Każdy plik ma rozmiar dataset_size (A) + dataset_size (B).
        
        collected_A = []
        collected_B = []
        batch_size = 100_000  
        file_counter = 0 # Licznik utworzonych plików

        print(f"Generowanie wielu plików Sudden Drift ({year_a} vs {year_b})...")

        while self.file_index < 60_000_000:

            print(f"\nProcessing Batch {self.file_index} - {self.file_index + batch_size}")
            
            # 1. Wczytanie danych
            try:
                self.df_reviews = self.load_reviews(self.file_index, self.file_index + batch_size)
            except Exception as e:
                print(f"Error loading reviews: {e}")
                break
                
            self.file_index += batch_size
            print(f"Reviews in batch: {len(self.df_reviews)}")

            # 2. Ładowanie biznesów (tylko raz)
            if self.df_business is None:
                self.df_business = self.load_business()
                print(f"Businesses loaded: {len(self.df_business)}")

            # 3. Merge
            self.df_merged = self.merge_reviews_with_business()
            
            # 4. Filtrowanie
            A_part = self.filter_by_year_and_category(year_a, CATEGORY_A)
            B_part = self.filter_by_year_and_category(year_b, CATEGORY_A) # UWAGA: Tu chyba miało być CATEGORY_B? Zostawiam jak w oryginale, ale sprawdź to.
            
            # Konwersja dat na string (dla JSON)
            if "date" in A_part.columns:
                A_part = A_part.copy()
                A_part["date"] = A_part["date"].astype(str)
            if "date" in B_part.columns:
                B_part = B_part.copy()
                B_part["date"] = B_part["date"].astype(str)

            # 5. Dodanie do bufora
            collected_A.extend(A_part.to_dict("records"))
            collected_B.extend(B_part.to_dict("records"))

            print(f"Buffer status -> A: {len(collected_A)}, B: {len(collected_B)}")

            # 6. ZAPISYWANIE PLIKÓW (Pętla "opróżniająca" bufor)
            # Dopóki mamy wystarczająco danych na pełny plik, zapisujemy i tniemy listy
            while len(collected_A) >= dataset_size and len(collected_B) >= dataset_size:
                file_counter += 1
                
                # Wycinamy porcję do zapisu
                chunk_A = collected_A[:dataset_size]
                chunk_B = collected_B[:dataset_size]
                
                # Usuwamy zapisaną porcję z bufora (zostawiamy resztę na później)
                collected_A = collected_A[dataset_size:]
                collected_B = collected_B[dataset_size:]
                
                # Tworzenie nazwy pliku wg Twojego wzoru
                # Zakładamy, że output_path kończy się np. "_0.json" (6 znaków do ucięcia)
                current_filename = output_path[:-6] + str(file_counter) + ".json"
                
                print(f"--> Zapisywanie pliku nr {file_counter}: {current_filename}")
                
                with open(current_filename, "w", encoding="utf-8") as f:
                    for rec in chunk_A:
                        rec = self.make_json_serializable(rec)
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    for rec in chunk_B:
                        rec = self.make_json_serializable(rec)
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Podsumowanie po przejściu całego pliku
        print("\n--- Zakończono przetwarzanie ---")
        print(f"Utworzono łącznie plików: {file_counter}")
        print(f"Odrzucone resztki (za mało na pełny plik): A={len(collected_A)}, B={len(collected_B)}")
        
    def create_gradual_drift_data(self, dataset_size_per_segment=DATASET_SIZE, output_path=OUTPUT_PATH):
        #Tworzy dane typu gradual drift:
        #segment 1: 100% CATEGORY_A
        #segment 2: 80% A / 20% B
        #segment 3: 60% A / 40% B
        #...
        #segment N: 0% A / 100% B

        #Każdy segment ma dataset_size_per_segment próbek.
        #Dane są zapisywane jako JSONL do output_path.

        collected_A = []
        collected_B = []

        batch_size = 100_000 

        print("Gradual drift dataset")
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

        print(f"Need A: {max_A_needed}, B: {max_B_needed}")

        while self.file_index < 30_000_000:
            print(f"\nBatch {self.file_index} - {self.file_index + batch_size}")

            self.df_reviews = self.load_reviews(self.file_index, self.file_index + batch_size)
            self.file_index += batch_size

            print(f"Reviews: {len(self.df_reviews)}")

            if self.df_business is None:
                self.df_business = self.load_business()
                print(f"Businesses: {len(self.df_business)}")

            self.df_merged = self.merge_reviews_with_business()
            print(f"Records total: {len(self.df_merged)}")

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

            print(f"Samples A: {len(collected_A)}, B: {len(collected_B)}")

            if len(collected_A) >= max_A_needed and len(collected_B) >= max_B_needed:
                break

        if len(collected_A) < max_A_needed or len(collected_B) < max_B_needed:
            print("WARNING: Not enough samples")
            print(f"A={len(collected_A)} / B={len(collected_B)}, need A={max_A_needed}, B={max_B_needed}")

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

            print(f"Segment {i}: ratio A={ratio_A}, B={ratio_B} -> A={len(seg_A)}, B={len(seg_B)}, total={len(segment)}")

            stream.extend(segment)

        print(f"\nSamples total: {len(stream)}")

        with open(output_path, "w", encoding="utf-8") as f:
            for rec in stream:
                rec = self.make_json_serializable(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"\nOutput path: {output_path}")
        print(f"Segments: {len(ratios)}, segment size: {dataset_size_per_segment}, records total: {len(stream)}")
        
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
    current_output = OUTPUT_PATH + "_0.json"
    if DRIFT_TYPE == "sudden":
        for i in range(10):
            yd.create_sudden_drift_data(output_path=current_output)
            current_output = current_output[:-6]
            current_output += str(i+1) + ".json"
    elif DRIFT_TYPE == "sudden_by_year":
        for i in range(10):
            yd.create_sudden_drift_data_by_year(output_path=current_output, year_a=2010, year_b= 2021)
            current_output = current_output[:-6] + str(i+1) + ".json"
    elif DRIFT_TYPE == "gradual":
        yd.create_gradual_drift_data(output_path=current_output)
            