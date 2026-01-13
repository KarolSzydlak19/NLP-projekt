import json
from sentence_transformers import SentenceTransformer
import os
import argparse

# Domyślne wartości (zostaną nadpisane przez argumenty)
INPUT_PATH = "D:\\nlp\\datasets\\sudden" 
OUTPUT_PATH = "D:\\nlp\\datasets\\embed\\sudden"   
BATCH_SIZE = 256                         
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def parse_args():
    parser = argparse.ArgumentParser(description="Embedding generator with configurable parameters.")

    parser.add_argument(
        "--ip", "--input_path",
        dest="input_path",
        default=INPUT_PATH,
        type=str,
        required=False,
        help="Path to the input dataset file or directory"
    )

    parser.add_argument(
        "--op", "--output_path",
        dest="output_path",
        default=OUTPUT_PATH,
        type=str,
        required=False,
        help="Path to save the output file or directory"
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
    """Generator zwracający batche linii z pliku JSONL."""
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

def process_file(model, input_file, output_file, batch_size):
    """Funkcja przetwarzająca jeden konkretny plik."""
    output_file = input_file + "_embedded.jsonl"
    
    # Jeśli plik wyjściowy istnieje, usuń go przed zapisem
    if os.path.exists(output_file):
        os.remove(output_file)

    # Upewnij się, że katalog dla pliku wyjściowego istnieje
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Processing: {input_file} -> {output_file}")
    
    out = open(output_file, "w", encoding="utf-8")
    total_lines = 0

    for batch_idx, batch in enumerate(batch_reader(input_file, batch_size)):
        texts = []
        objs = []
        for line in batch:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Zakładamy, że pole tekstowe nazywa się "text"
            if "text" in obj:
                objs.append(obj)
                texts.append(obj["text"])

        if not texts:
            continue

        embeddings = model.encode(
            texts,
            batch_size=64, # Batch size wewnętrzny modelu (nie musi być taki sam jak batch czytania)
            convert_to_numpy=True,
            show_progress_bar=False
        )

        for obj, emb in zip(objs, embeddings):
            obj["embedding"] = emb.tolist()
            out.write(json.dumps(obj) + "\n")

        total_lines += len(batch)
        print(f"   Lines processed: {total_lines}", end="\r")
    
    out.close()
    print(f"\n   Finished file: {output_file}")

def main():
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Sprawdzamy czy INPUT_PATH to katalog czy plik
    if os.path.isdir(INPUT_PATH):
        print(f"Input is a directory: {INPUT_PATH}")
        
        # Jeśli input to katalog, output też traktujemy jako katalog
        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)

        # Pobieramy listę plików w katalogu
        files = [f for f in os.listdir(INPUT_PATH) if os.path.isfile(os.path.join(INPUT_PATH, f))]
        
        for filename in files:
            # Pomiń pliki systemowe lub ukryte (opcjonalnie)
            if filename.startswith("."):
                continue
                
            full_input_path = os.path.join(INPUT_PATH, filename)
            full_output_path = os.path.join(OUTPUT_PATH, filename)
            
            process_file(model, full_input_path, full_output_path, BATCH_SIZE)

    else:
        # Tryb pojedynczego pliku (stare zachowanie)
        print(f"Input is a single file: {INPUT_PATH}")
        process_file(model, INPUT_PATH, OUTPUT_PATH, BATCH_SIZE)

if __name__ == "__main__":
    args = parse_args()
    
    INPUT_PATH = args.input_path
    OUTPUT_PATH = args.output_path
    BATCH_SIZE = args.batch_size
    MODEL_NAME = args.model_name
    
    main()