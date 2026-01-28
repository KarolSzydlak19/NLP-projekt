import os
import json
import heapq
import tempfile

INPUT_PATH = "../../nlp-datasets/yelp_dataset/yelp_academic_dataset_review.json" 
OUTPUT_PATH = "../../nlp-datasets/yelp_dataset/yelp_academic_dataset_review_sorted.json"   
CHUNK_SIZE = 300_000 


def make_chunks(apath):
    chunk_paths = []
    chunk = []
    count = 0
    idx = 0
    num_chunks = 0

    with open(apath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if num_chunks >= 10:
                break
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "date" not in obj:
                continue

            chunk.append((obj["date"], line))
            count += 1

            if count >= CHUNK_SIZE:
                num_chunks += 1
                chunk.sort(key=lambda x: x[0])

                fd, temp_path = tempfile.mkstemp(prefix=f"chunk_{idx}_", suffix=".jsonl")
                os.close(fd)

                with open(temp_path, "w", encoding="utf-8") as temp:
                    for _, l in chunk:
                        temp.write(l + "\n")

                chunk_paths.append(temp_path)

                chunk.clear()
                count = 0
                idx += 1

    # ostatni chunk
    if chunk:
        chunk.sort(key=lambda x: x[0])
        fd, temp_path = tempfile.mkstemp(prefix=f"chunk_{idx}_", suffix=".jsonl")
        os.close(fd)
        with open(temp_path, "w", encoding="utf-8") as temp:
            for _, l in chunk:
                temp.write(l + "\n")
        chunk_paths.append(temp_path)

    return chunk_paths


def merge_chunks(chunk_paths, output_path):

    # otwieramy wszystkie chunk pliki
    files = [open(p, "r", encoding="utf-8") for p in chunk_paths]
    heap = []

    # inicjalizacja kopca
    for idx, f in enumerate(files):
        line = f.readline().strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except:
            continue
        heapq.heappush(heap, (obj["date"], idx, line))

    with open(output_path, "w", encoding="utf-8") as out:
        written = 0

        while heap:
            date, idx, line = heapq.heappop(heap)
            out.write(line + "\n")
            written += 1

            nxt = files[idx].readline().strip()
            if nxt:
                try:
                    obj = json.loads(nxt)
                    heapq.heappush(heap, (obj["date"], idx, nxt))
                except:
                    pass

    for f in files:
        f.close()


def cleanup(chunk_paths):
    for p in chunk_paths:
        try:
            os.remove(p)
        except:
            print("  [WARN] ", p)


def main():
    chunks = make_chunks(INPUT_PATH)
    merge_chunks(chunks, OUTPUT_PATH)
    cleanup(chunks)


if __name__ == "__main__":
    main()