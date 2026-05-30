import os
import re
import json
import math
import time
import heapq
import warnings
from collections import defaultdict
from urllib.parse import urldefrag

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning, MarkupResemblesLocatorWarning
from nltk.stem import PorterStemmer

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


class Indexer:
    def __init__(
        self,
        data_dir="./DEV",
        partial_dir="partial_indexes",
        output_dir="final_index",
        flush_limit=5000,
    ):
        self.data_dir = data_dir
        self.partial_dir = partial_dir
        self.output_dir = output_dir
        self.flush_limit = flush_limit

        self.stemmer = PorterStemmer()
        self.index = defaultdict(lambda: defaultdict(float))

        self.doc_id_map = {}
        self.doc_lengths = {}
        self.document_count = 0
        self.partial_count = 0

        os.makedirs(self.partial_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def tokenize(self, text):
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return [self.stemmer.stem(token) for token in tokens]

    def extract_weighted_tokens(self, html):
        soup = BeautifulSoup(html, "lxml")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        all_text = soup.get_text(separator=" ")
        normal_tokens = self.tokenize(all_text)

        important_text = ""
        for tag in soup.find_all(["title", "h1", "h2", "h3", "b", "strong"]):
            important_text += " " + tag.get_text(separator=" ")

        important_tokens = self.tokenize(important_text)

        return normal_tokens, important_tokens

    def process_file(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)

            url = data.get("url", "")
            html = data.get("content", "")

            if not url or not html:
                return

            url, _ = urldefrag(url)

            self.document_count += 1
            doc_id = str(self.document_count)
            self.doc_id_map[doc_id] = url

            normal_tokens, important_tokens = self.extract_weighted_tokens(html)

            if not normal_tokens and not important_tokens:
                return

            tf = defaultdict(float)

            for token in normal_tokens:
                tf[token] += 1.0

            # Important words are counted more heavily.
            for token in important_tokens:
                tf[token] += 3.0

            self.doc_lengths[doc_id] = max(len(normal_tokens), 1)

            for token, freq in tf.items():
                self.index[token][doc_id] += freq

        except Exception as e:
            print(f"Skipping {file_path}: {e}")

    def build_partial_indexes(self):
        start = time.time()

        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".json"):
                    path = os.path.join(root, file)
                    self.process_file(path)

                    if self.document_count > 0 and self.document_count % self.flush_limit == 0:
                        self.flush_partial_index()

        if self.index:
            self.flush_partial_index()

        elapsed = time.time() - start
        print(f"Finished building partial indexes in {elapsed:.2f} seconds.")

    def flush_partial_index(self):
        self.partial_count += 1
        output_path = os.path.join(self.partial_dir, f"partial_{self.partial_count}.jsonl")

        with open(output_path, "w", encoding="utf-8") as f:
            for term in sorted(self.index.keys()):
                postings = self.index[term]
                record = {
                    "term": term,
                    "postings": dict(postings),
                }
                f.write(json.dumps(record) + "\n")

        print(f"Saved {output_path}")
        self.index.clear()

    def _partial_reader(self, path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    yield record["term"], record["postings"]

    def merge_partial_indexes(self):
        """
        K-way merge of sorted partial indexes.

        This avoids loading the full final index into memory.
        It writes:
        - postings.jsonl: one JSON record per term
        - lexicon.json: term -> byte offset + byte length in postings.jsonl
        """
        partial_files = [
            os.path.join(self.partial_dir, file)
            for file in os.listdir(self.partial_dir)
            if file.endswith(".jsonl")
        ]

        readers = []
        heap = []

        for i, path in enumerate(sorted(partial_files)):
            reader = self._partial_reader(path)
            readers.append(reader)

            try:
                term, postings = next(reader)
                heapq.heappush(heap, (term, i, postings))
            except StopIteration:
                pass

        postings_path = os.path.join(self.output_dir, "postings.jsonl")
        lexicon_path = os.path.join(self.output_dir, "lexicon.json")

        lexicon = {}
        unique_tokens = 0

        with open(postings_path, "w", encoding="utf-8") as out:
            while heap:
                current_term = heap[0][0]
                merged_postings = defaultdict(float)

                while heap and heap[0][0] == current_term:
                    _, reader_id, postings = heapq.heappop(heap)

                    for doc_id, tf in postings.items():
                        merged_postings[doc_id] += float(tf)

                    try:
                        next_term, next_postings = next(readers[reader_id])
                        heapq.heappush(heap, (next_term, reader_id, next_postings))
                    except StopIteration:
                        pass

                postings_list = [
                    {"doc_id": doc_id, "tf": tf}
                    for doc_id, tf in merged_postings.items()
                ]

                record = {
                    "term": current_term,
                    "df": len(postings_list),
                    "postings": postings_list,
                }

                line = json.dumps(record, separators=(",", ":")) + "\n"
                encoded = line.encode("utf-8")

                offset = out.tell()
                out.write(line)
                length = len(encoded)

                lexicon[current_term] = {
                    "offset": offset,
                    "length": length,
                    "df": len(postings_list),
                }

                unique_tokens += 1

        with open(lexicon_path, "w", encoding="utf-8") as f:
            json.dump(lexicon, f)

        self.unique_tokens = unique_tokens
        print(f"Saved postings to {postings_path}")
        print(f"Saved lexicon to {lexicon_path}")

    def save_metadata(self):
        doc_map_path = os.path.join(self.output_dir, "doc_id_map.json")
        lengths_path = os.path.join(self.output_dir, "doc_lengths.json")
        stats_path = os.path.join(self.output_dir, "stats.json")

        with open(doc_map_path, "w", encoding="utf-8") as f:
            json.dump(self.doc_id_map, f)

        with open(lengths_path, "w", encoding="utf-8") as f:
            json.dump(self.doc_lengths, f)

        postings_path = os.path.join(self.output_dir, "postings.jsonl")
        lexicon_path = os.path.join(self.output_dir, "lexicon.json")

        postings_size = os.path.getsize(postings_path) if os.path.exists(postings_path) else 0
        lexicon_size = os.path.getsize(lexicon_path) if os.path.exists(lexicon_path) else 0

        stats = {
            "documents_indexed": self.document_count,
            "partial_indexes_created": self.partial_count,
            "unique_tokens": self.unique_tokens,
            "postings_size_bytes": postings_size,
            "postings_size_mb": round(postings_size / (1024 * 1024), 2),
            "lexicon_size_bytes": lexicon_size,
            "lexicon_size_mb": round(lexicon_size / (1024 * 1024), 2),
        }

        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4)

        print("\nFinal Stats")
        print("-----------")
        print(f"Documents indexed: {self.document_count}")
        print(f"Partial indexes created: {self.partial_count}")
        print(f"Unique tokens: {self.unique_tokens}")
        print(f"Postings size: {stats['postings_size_mb']} MB")
        print(f"Lexicon size: {stats['lexicon_size_mb']} MB")

    def run(self):
        print("Building partial indexes...")
        self.build_partial_indexes()

        print("Merging partial indexes...")
        self.merge_partial_indexes()

        print("Saving metadata...")
        self.save_metadata()

        print("Done.")


if __name__ == "__main__":
    indexer = Indexer(
        data_dir="./DEV",
        partial_dir="partial_indexes",
        output_dir="final_index",
        flush_limit=5000,
    )

    indexer.run()