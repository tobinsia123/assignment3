import os
import re
import json
import math
import time
import heapq
from collections import defaultdict
from nltk.stem import PorterStemmer


class SearchEngine:
    def __init__(self, index_dir="final_index"):
        self.index_dir = index_dir
        self.stemmer = PorterStemmer()

        self.lexicon_path = os.path.join(index_dir, "lexicon.json")
        self.postings_path = os.path.join(index_dir, "postings.jsonl")
        self.doc_map_path = os.path.join(index_dir, "doc_id_map.json")
        self.doc_lengths_path = os.path.join(index_dir, "doc_lengths.json")
        self.stats_path = os.path.join(index_dir, "stats.json")

        with open(self.lexicon_path, "r", encoding="utf-8") as f:
            self.lexicon = json.load(f)

        with open(self.doc_map_path, "r", encoding="utf-8") as f:
            self.doc_id_map = json.load(f)

        with open(self.doc_lengths_path, "r", encoding="utf-8") as f:
            self.doc_lengths = json.load(f)

        with open(self.stats_path, "r", encoding="utf-8") as f:
            self.stats = json.load(f)

        self.total_docs = self.stats["documents_indexed"]

    def tokenize(self, text):
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return [self.stemmer.stem(token) for token in tokens]

    def get_postings(self, term):
        """
        Reads only the postings for one term from disk using the lexicon.
        This avoids loading the full postings file into memory.
        """
        if term not in self.lexicon:
            return None

        entry = self.lexicon[term]
        offset = entry["offset"]
        length = entry["length"]

        with open(self.postings_path, "r", encoding="utf-8") as f:
            f.seek(offset)
            line = f.read(length)

        return json.loads(line)

    def url_quality_multiplier(self, url):
        """
        General URL-quality heuristic.

        Penalizes low-value crawler artifacts such as image detail pages,
        slide pages, photo pages, and malformed encoded URLs.

        Slightly rewards official/current-looking ICS, CS, Informatics,
        graduate, undergraduate, and UCI ML Repository pages.
        """
        url_lower = url.lower()
        multiplier = 1.0

        bad_patterns = [
            "detail.php",
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            "/pix/",
            "/photos/",
            "sld",
            "tsld",
            "slide",
            "%3c",
            "%22",
            "%3e",
        ]

        for pattern in bad_patterns:
            if pattern in url_lower:
                multiplier *= 0.75

        good_patterns = [
            "ics.uci.edu/ugrad",
            "ics.uci.edu/grad",
            "cs.uci.edu",
            "informatics.uci.edu",
            "mcs.ics.uci.edu",
            "mswe.ics.uci.edu",
            "archive.ics.uci.edu/ml",
            "cyberclub.ics.uci.edu",
            "wics.ics.uci.edu",
        ]

        for pattern in good_patterns:
            if pattern in url_lower:
                multiplier *= 1.10
                break

        return multiplier

    def search(self, query, top_k=10):
        start = time.time()

        query_tokens = self.tokenize(query)

        if not query_tokens:
            return [], 0.0

        # Remove duplicate query tokens while preserving order.
        seen = set()
        unique_query_tokens = []

        for token in query_tokens:
            if token not in seen:
                seen.add(token)
                unique_query_tokens.append(token)

        scores = defaultdict(float)
        matched_terms = defaultdict(set)

        for term in unique_query_tokens:
            term_data = self.get_postings(term)

            if term_data is None:
                continue

            df = term_data["df"]
            idf = math.log((self.total_docs + 1) / (df + 1)) + 1

            for posting in term_data["postings"]:
                doc_id = posting["doc_id"]
                raw_tf = float(posting["tf"])

                if raw_tf <= 0:
                    continue

                # Heuristic 1: log-scaled TF.
                # Prevents repeated terms from dominating too much.
                tf_weight = 1 + math.log(raw_tf)

                # Heuristic 2: document length normalization.
                # Prevents huge pages from always ranking highest.
                doc_len = float(self.doc_lengths.get(doc_id, 1))
                length_norm = math.sqrt(doc_len)

                scores[doc_id] += (tf_weight * idf) / length_norm
                matched_terms[doc_id].add(term)

        if not scores:
            return [], time.time() - start

        query_term_count = len(unique_query_tokens)
        final_scores = {}

        for doc_id, score in scores.items():
            # Heuristic 3: query coverage boost.
            # Pages matching more query terms get rewarded.
            coverage = len(matched_terms[doc_id]) / query_term_count
            coverage_boost = 1.0 + coverage

            # Heuristic 4: URL quality multiplier.
            # Penalizes low-value crawler artifacts and rewards official pages.
            url = self.doc_id_map.get(doc_id, "")
            url_multiplier = self.url_quality_multiplier(url)

            final_scores[doc_id] = score * coverage_boost * url_multiplier

        # Get more than top_k first so duplicate filtering still leaves enough results.
        candidates = heapq.nlargest(top_k * 5, final_scores.items(), key=lambda item: item[1])

        results = []
        seen_urls = set()

        for doc_id, score in candidates:
            url = self.doc_id_map.get(doc_id, "UNKNOWN_URL")

            # Heuristic 5: exact duplicate URL filtering.
            if url in seen_urls:
                continue

            seen_urls.add(url)

            results.append(
                {
                    "doc_id": doc_id,
                    "url": url,
                    "score": score,
                    "matched_terms": sorted(matched_terms[doc_id]),
                }
            )

            if len(results) == top_k:
                break

        elapsed = time.time() - start
        return results, elapsed

    def interactive_loop(self):
        while True:
            query = input("\nEnter your search query or 'exit': ").strip()

            if query.lower() == "exit":
                break

            results, elapsed = self.search(query, top_k=10)

            print(f"\nSearch time: {elapsed * 1000:.2f} ms")

            if not results:
                print("No results found.")
                continue

            for rank, result in enumerate(results, start=1):
                print(
                    f"{rank}. {result['url']}\n"
                    f"   score={result['score']:.6f}, "
                    f"matched_terms={result['matched_terms']}"
                )


if __name__ == "__main__":
    engine = SearchEngine(index_dir="final_index")
    engine.interactive_loop()