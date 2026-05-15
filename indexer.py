import os
import re
import json
import math
import time
import warnings

from collections import defaultdict
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from nltk.stem import PorterStemmer

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

class Indexer:
    def __init__(self, data_dir="./DEV", partial_dir="partial_indexes", output_dir="final_index", flush_limit=5000):
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

        all_text = soup.get_text(separator=" ")
        normal_tokens = self.tokenize(all_text)

        important_text = ""
        # important text would be big/emphasized texts (title/h1/h2/h3/b/strong)
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

            self.document_count += 1
            # increment document_count to ensure uniqueness
            doc_id = str(self.document_count)
            self.doc_id_map[doc_id] = url

            normal_tokens, important_tokens = self.extract_weighted_tokens(html)
            # token frequency dict
            tf = defaultdict(float)

            for token in normal_tokens:
                tf[token] += 1.0

            # Important words count more
            for token in important_tokens:
                tf[token] += 3.0

            self.doc_lengths[doc_id] = len(normal_tokens)
            # add to index
            for token, freq in tf.items():
                self.index[token][doc_id] += freq

        except Exception as e:
            print(f"Skipping {file_path}: {e}")

    def build_partial_indexes(self):
        start = time.time()
        # go through all website data
        # process token frequencies for a file
        # if reached flush limit (5000), flush by dumping into disk + clearing index 
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

    # dump current partial index into json
    def flush_partial_index(self):
        self.partial_count += 1
        output_path = os.path.join(self.partial_dir, f"partial_{self.partial_count}.json")

        serializable = {
            term: dict(postings)
            for term, postings in self.index.items()
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f)

        print(f"Saved {output_path}")

        self.index.clear()
    # combine all partials at the end by merging frequencies.
    def merge_partial_indexes(self):
        merged = defaultdict(lambda: defaultdict(float))

        for file in os.listdir(self.partial_dir):
            if not file.endswith(".json"):
                continue

            path = os.path.join(self.partial_dir, file)

            with open(path, "r", encoding="utf-8") as f:
                partial = json.load(f)

            for term, postings in partial.items():
                for doc_id, freq in postings.items():
                    merged[term][doc_id] += freq

        final_index = {}

        for term, postings in merged.items():
            # document frequency or # documents with word
            df = len(postings)

            final_index[term] = {
                "df": df,
                "postings": [
                    {
                        "doc_id": doc_id,
                        # use tf-idf in the future
                        "tf": freq
                    }
                    for doc_id, freq in postings.items()
                ]
            }

        output_path = os.path.join(self.output_dir, "inverted_index.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_index, f)
        
        self.unique_tokens = len(final_index)
        print(f"Saved merged final index to {output_path}")

    def save_metadata(self):
        doc_map_path = os.path.join(self.output_dir, "doc_id_map.json")
        lengths_path = os.path.join(self.output_dir, "doc_lengths.json")
        stats_path = os.path.join(self.output_dir, "stats.json")

        with open(doc_map_path, "w", encoding="utf-8") as f:
            json.dump(self.doc_id_map, f)

        with open(lengths_path, "w", encoding="utf-8") as f:
            json.dump(self.doc_lengths, f)

        index_path = os.path.join(self.output_dir, "inverted_index.json")
        index_size = os.path.getsize(index_path) if os.path.exists(index_path) else 0

        stats = {
            "documents_indexed": self.document_count,
            "partial_indexes_created": self.partial_count,
            "unique_tokens": self.unique_tokens,
            "final_index_size_bytes": index_size,
            "final_index_size_mb": round(index_size / (1024 * 1024), 2)
        }

        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4)

        print("\nM1 Stats")
        print("--------")
        print(f"Documents indexed: {self.document_count}")
        print(f"Partial indexes created: {self.partial_count}")
        print(f"Unique tokens: {self.unique_tokens}")
        print(f"Final index size: {stats['final_index_size_mb']} MB")

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
        flush_limit=5000
    )

    indexer.run()