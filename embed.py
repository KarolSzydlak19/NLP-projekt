import json
from sentence_transformers import SentenceTransformer
import os
import argparse
import sys

INPUT_PATH = "" 
OUTPUT_PATH = ""   
BATCH_SIZE = 256                         
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def parse_args():
    parser = argparse.ArgumentParser(description="Embedding generator with configurable parameters.")

    parser.add_argument(
        "--ip", "--input_path",
        dest="input_path",
        default=INPUT_PATH,
        type=str,
        required=True,
        help="Path to the input dataset file"
    )

    parser.add_argument(
        "--op", "--output_path",
        dest="output_path",
        default=OUTPUT_PATH,
        type=str,
        required=True,
        help="Path to save the output file"
    )

    parser.add_argument(
        "--b", "--batch_size",
        dest="batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for the embedding model"
    )

    parser.add_argument(
        "--m", "--model_name",
        dest="model_name",
        type=str,
        default=MODEL_NAME,
        help="Name of the embedding model"
    )

    return parser.parse_args()

def batch_reader(path, batch_size):
    #Generator zwracający batche linii z pliku JSONL.
    batch = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                batch.append(line)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def main():
    print(f"Embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    out = open(OUTPUT_PATH, "w", encoding="utf-8")

    total_lines = 0

    print(f"Input file path: {INPUT_PATH}")
    for batch_idx, batch in enumerate(batch_reader(INPUT_PATH, BATCH_SIZE)):
        texts = []
        objs = []
        for line in batch:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            objs.append(obj)
            texts.append(obj["text"])

        embeddings = model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        for obj, emb in zip(objs, embeddings):
            obj["embedding"] = emb.tolist()
            out.write(json.dumps(obj) + "\n")

        total_lines += len(batch)
        print(f"Lines processed: {total_lines}", end="\r")

    out.close()
    print(f"\nOutput path: {OUTPUT_PATH}")


if __name__ == "__main__":
    args = parse_args()
    INPUT_PATH = args.input_path
    OUTPUT_PATH = args.output_path
    BATCH_SIZE = args.batch_size
    MODEL_NAME = args.model_name
    main()
